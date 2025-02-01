import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans

class ColorshiftColorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "color_count": ("INT", {"default": 8, "min": 2, "max": 64}),
            },
            "optional": {
                "lock_masks": ("MASK", {"default": None}),
                "palette_override": ("PALETTE", {"default": None}),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 50}),
                "sampling_rate": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.5}),
                "n_init": ("INT", {"default": 3, "min": 1, "max": 10}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PALETTE", "MASK", "IMAGE")
    RETURN_NAMES = ("images", "palette", "index_maps", "palette_preview")
    FUNCTION = "process"
    CATEGORY = "Image/Color"

    def process(self, images, color_count, lock_masks=None, palette_override=None, font_size=20, sampling_rate=0.25, n_init=3):
        batch_size, height, width, channels = images.shape
        pixels = images.reshape(batch_size, -1, 3)

        if palette_override is not None:
            palette = palette_override
            labels = self._match_palette_batch(pixels, palette)
        else:
            # パレット生成のために画像をリサイズ
            resized_images = F.interpolate(images.permute(0, 3, 1, 2), scale_factor=0.5, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            pixels_for_clustering = resized_images.reshape(batch_size, -1, 3).cpu().numpy().reshape(-1, 3)
            random_indices = np.random.choice(pixels_for_clustering.shape[0], size=int(pixels_for_clustering.shape[0] * sampling_rate), replace=False)
            pixels_for_clustering = pixels_for_clustering[random_indices]

            kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=n_init)
            kmeans.fit(pixels_for_clustering)
            # 全画素に対してクラスタ中心からの最近傍を求める
            labels, dist = vq(pixels.cpu().numpy().reshape(-1, 3), kmeans.cluster_centers_)
            labels = torch.from_numpy(labels.reshape(batch_size, -1)).long()
            palette = torch.from_numpy(kmeans.cluster_centers_).float()

            # ----- パレットの並び順を各クラスタの占有率が高い順に変更 -----
            # 全ラベルを1次元にまとめ、各クラスタの出現回数を計算
            all_labels = labels.flatten()
            counts = torch.bincount(all_labels, minlength=color_count)
            # 出現回数の降順に並べ替えたときの各クラスタの元のインデックス
            sorted_order = torch.argsort(counts, descending=True)
            # mapping: 旧インデックス -> 新インデックス を作成
            mapping = torch.empty_like(sorted_order)
            mapping[sorted_order] = torch.arange(sorted_order.size(0), device=sorted_order.device)
            # パレットを並び替え、各画素のラベルを新しい順序に置換
            palette = palette[sorted_order]
            labels = mapping[labels]
            # ---------------------------------------------------------------

        preview_img = self.generate_palette_preview(palette.cpu().numpy(), font_size)
        index_maps = labels.reshape(batch_size, height, width)

        if lock_masks is not None:
            lock_masks_batch = []
            for i in range(batch_size):
                lock_mask = lock_masks[i] if i < lock_masks.shape[0] else lock_masks[-1]
                lock_masks_batch.append(lock_mask)
            lock_masks = torch.stack(lock_masks_batch)

            pixel_masks = torch.zeros_like(index_maps, dtype=torch.bool)
            for i in range(batch_size):
                for j, is_locked in enumerate(lock_masks[i]):
                    if is_locked:
                        pixel_masks[i][index_maps[i] == j] = True

            original_pixels = pixels.clone()
            new_pixels = palette[labels]
            pixels[~pixel_masks] = new_pixels[~pixel_masks]
            pixels[pixel_masks] = original_pixels[pixel_masks]
        else:
            pixels = palette[labels]

        processed_images_tensor = pixels.reshape(batch_size, height, width, 3)
        return (processed_images_tensor, palette, index_maps, preview_img)

    def generate_palette_preview(self, palette, font_size):
        if isinstance(palette, torch.Tensor):
            colors = palette.numpy()
        else:
            colors = palette

        patch_size = 100
        cols = 8
        rows = (len(colors) + cols - 1) // cols

        img = Image.new("RGB", (cols * patch_size, rows * patch_size), (40, 40, 40))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

        for i, color in enumerate(colors):
            x = (i % cols) * patch_size
            y = (i // cols) * patch_size

            draw.rectangle([x, y, x + patch_size, y + patch_size], fill=tuple((color * 255).astype(int)))
            text = str(i)
            bbox = draw.textbbox((x, y), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x + (patch_size - text_w) // 2, y + (patch_size - text_h) // 2), text, fill="white", font=font)

        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return img_tensor

    def _match_palette_batch(self, pixels, palette):
        distances = torch.cdist(pixels, palette)
        return torch.argmin(distances, dim=2)

class PaletteEditorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "palette": ("PALETTE",),
                "index_maps": ("MASK",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "hue_random_enable": ("BOOLEAN", {"default": False}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                "saturation_random_enable": ("BOOLEAN", {"default": False}),
                "saturation_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "value_random_enable": ("BOOLEAN", {"default": False}),
                "value_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0}),
                "lock_color_num": ("STRING", {"default": "0,", "placeholder": "例: 0,1,2,3,4"}),
                "mask_enable": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "operations": ("STRING", {"default": "[]", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PALETTE", "MASK")
    RETURN_NAMES = ("images", "palette", "mask")
    FUNCTION = "process_palette"
    CATEGORY = "Image/Color"

    def rgb_to_hsv(self, rgb):
        r, g, b = rgb.unbind(-1)
        max_rgb, argmax_rgb = rgb.max(-1)
        min_rgb, _ = rgb.min(-1)
        delta = max_rgb - min_rgb

        h = torch.zeros_like(r)
        s = torch.zeros_like(r)
        v = max_rgb

        non_zero_delta = delta != 0
        h[non_zero_delta] = torch.where(
            argmax_rgb[non_zero_delta] == 0,
            (g[non_zero_delta] - b[non_zero_delta]) / delta[non_zero_delta],
            torch.where(
                argmax_rgb[non_zero_delta] == 1,
                2 + (b[non_zero_delta] - r[non_zero_delta]) / delta[non_zero_delta],
                4 + (r[non_zero_delta] - g[non_zero_delta]) / delta[non_zero_delta],
            ),
        )
        h = (h / 6.0) % 1.0
        s[non_zero_delta] = delta[non_zero_delta] / max_rgb[non_zero_delta]

        return torch.stack((h, s, v), dim=-1)

    def hsv_to_rgb(self, hsv):
        h, s, v = hsv.unbind(-1)

        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c

        rgb_prime = torch.zeros_like(hsv)

        h_category = (h * 6).long() % 6

        rgb_prime[..., 0] = torch.where(
            (h_category == 0) | (h_category == 5),
            c,
            torch.where((h_category == 1) | (h_category == 4), x, 0),
        )
        rgb_prime[..., 1] = torch.where(
            (h_category == 1) | (h_category == 2),
            c,
            torch.where((h_category == 0) | (h_category == 3), x, 0),
        )
        rgb_prime[..., 2] = torch.where(
            (h_category == 3) | (h_category == 4),
            c,
            torch.where((h_category == 2) | (h_category == 5), x, 0),
        )

        rgb = rgb_prime + m[..., None]
        return rgb

    def process_palette(self, images, palette, index_maps, seed, hue_random_enable, hue_shift, saturation_random_enable, saturation_scale, value_random_enable, value_scale, lock_color_num="0,", mask_enable=False, invert_mask=False, operations="[]"):
        palette = palette
        index_maps = index_maps

        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        if not mask_enable:
            color_mask = torch.zeros(len(palette), dtype=torch.bool)
        else:
            try:
                indices = [int(idx.strip()) for idx in lock_color_num.split(",") if idx.strip()]
                color_mask = torch.zeros(len(palette), dtype=torch.bool)
                for idx in indices:
                    if 0 <= idx < len(palette):
                        color_mask[idx] = True
                if invert_mask:
                    color_mask = ~color_mask
            except ValueError:
                color_mask = torch.zeros(len(palette), dtype=torch.bool)

        current_hue = random.uniform(-180, 180) if hue_random_enable else hue_shift
        current_saturation = random.uniform(0, 5) if saturation_random_enable else saturation_scale
        current_value = random.uniform(0, 5) if value_random_enable else value_scale

        modified_palette = self.edit_palette(palette, operations, color_mask, current_hue, current_saturation, current_value)

        pixels = images.reshape(batch_size, -1, 3)
        labels = index_maps.reshape(batch_size, -1)
        new_pixels = modified_palette[labels.long()]
        original_pixels = pixels.clone()

        pixel_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        for i, is_masked in enumerate(color_mask):
            pixel_mask[index_maps == i] = 1.0 if is_masked else 0.0

        pixels_mask = pixel_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3)
        processed_pixels = torch.where(pixels_mask > 0.5, original_pixels, new_pixels)
        processed_images = processed_pixels.reshape(images.shape)

        random.seed(None)
        return (processed_images, modified_palette, pixel_mask)

    def edit_palette(self, palette, operations, mask, hue_shift, saturation_scale, value_scale):
        hsv_palette = self.rgb_to_hsv(palette)

        hsv_palette[~mask, 0] = (hsv_palette[~mask, 0] + hue_shift / 360.0) % 1.0
        hsv_palette[~mask, 1] = torch.clamp(hsv_palette[~mask, 1] * saturation_scale, 0, 1)
        hsv_palette[~mask, 2] = torch.clamp(hsv_palette[~mask, 2] * value_scale, 0, 1)

        try:
            ops = json.loads(operations)
            if isinstance(ops, list):
                for op in ops:
                    if isinstance(op, dict) and "index" in op:
                        idx = op["index"]
                        if 0 <= idx < len(palette):
                            if "color" in op and len(op["color"]) == 3:
                                new_color = torch.tensor(op["color"], dtype=torch.float32)
                                palette[idx] = new_color
                                hsv_palette[idx] = self.rgb_to_hsv(new_color.reshape(1, 3))[0]
                            if "hsv" in op and len(op["hsv"]) == 3:
                                hsv_palette[idx] = torch.clamp(torch.tensor(op["hsv"]), 0, 1)
        except Exception as e:
            print(f"Operation Error: {str(e)}")

        modified_palette = self.hsv_to_rgb(hsv_palette)
        return modified_palette

class CsCFill:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "palette": ("PALETTE",),
                "index_maps": ("MASK",),
            },
            "optional": {
                "operations": ("STRING", {"default": "[]", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("filled_image", "mask_image")
    FUNCTION = "process"
    CATEGORY = "Image/Color"

    def process(self, images, palette, index_maps, operations="[]"):
        batch_size, height, width, channels = images.shape

        pairs = None
        try:
            ops = json.loads(operations)
            if isinstance(ops, list) and len(ops) > 0:
                manual_pairs = []
                for op in ops:
                    if isinstance(op, dict) and "A" in op and "B" in op:
                        try:
                            a_idx = int(op["A"])
                            b_idx = int(op["B"])
                            if 0 <= a_idx < palette.shape[0] and 0 <= b_idx < palette.shape[0]:
                                brightness_a = palette[a_idx].sum()
                                brightness_b = palette[b_idx].sum()
                                if brightness_a < brightness_b:
                                    a_idx, b_idx = b_idx, a_idx
                                manual_pairs.append((a_idx, b_idx))
                        except Exception:
                            continue
                if len(manual_pairs) > 0:
                    pairs = manual_pairs
        except Exception as e:
            pairs = None

        if pairs is None:
            pairs = self._compute_auto_pairs(palette)

        filled_image = images.clone()
        mask_image = torch.zeros(batch_size, height, width, 4, dtype=images.dtype)

        for (a_idx, b_idx) in pairs:
            mask = (index_maps == b_idx)
            mask_expanded_rgb = mask.unsqueeze(-1).expand(-1, -1, -1, 3)

            filled_color = palette[a_idx].view(1, 1, 1, 3).expand_as(filled_image)
            filled_image = torch.where(mask_expanded_rgb, filled_color, filled_image)

            b_color = palette[b_idx].view(1, 1, 1, 3).expand(batch_size, height, width, 3)
            current_rgb = mask_image[..., :3]
            updated_rgb = torch.where(mask_expanded_rgb, b_color, current_rgb)
            mask_image[..., :3] = updated_rgb

            current_alpha = mask_image[..., 3:4]
            new_alpha = torch.where(mask.unsqueeze(-1), torch.tensor(1.0, dtype=images.dtype), current_alpha)
            mask_image[..., 3:4] = new_alpha

        return (filled_image, mask_image)

    def _compute_auto_pairs(self, palette):
        palette_np = palette.detach().cpu().numpy()
        n = palette_np.shape[0]
        pair_list = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(palette_np[i] - palette_np[j])
                pair_list.append((i, j, dist))
        pair_list.sort(key=lambda x: x[2])
        
        paired = set()
        auto_pairs = []
        for i, j, dist in pair_list:
            if i in paired or j in paired:
                continue
            brightness_i = palette_np[i].sum()
            brightness_j = palette_np[j].sum()
            if brightness_i >= brightness_j:
                auto_pairs.append((i, j))
            else:
                auto_pairs.append((j, i))
            paired.add(i)
            paired.add(j)
        return auto_pairs


NODE_CLASS_MAPPINGS = {
    "ColorshiftColor": ColorshiftColorNode,
    "CsCPaletteEditor": PaletteEditorNode,
    "CsCFill": CsCFill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorshiftColor": "ColorshiftColor",
    "CsCPaletteEditor": "CsCPaletteEditor",
    "CsCFill": "CsCFill",
}
