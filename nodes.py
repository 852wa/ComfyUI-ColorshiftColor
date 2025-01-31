# nodes.py
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

    def process(
        self,
        images,
        color_count,
        lock_masks=None,
        palette_override=None,
        font_size=20,
        sampling_rate=0.25,
        n_init=3,
    ):
        # デバイスを取得
        device = images.device

        batch_size, height, width, channels = images.shape  # imagesの形状を使用
        pixels = images.reshape(batch_size, -1, 3)

        if palette_override is not None:
            palette = palette_override.to(device)
            # バッチ全体に対して一回の距離計算でラベルを取得
            labels = self._match_palette_batch(pixels, palette)
        else:
            # 複数の画像に対して、それぞれでクラスタリングをすると、計算量も増えるし、パレットがずれる。
            # 解決策として、複数の画像を混ぜてクラスタリングを行う
            # 混ぜてクラスタリング

            # パレット生成時にのみリサイズ
            resized_images = F.interpolate(
                images.permute(0, 3, 1, 2),
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

            pixels_for_clustering = (
                resized_images.reshape(batch_size, -1, 3).cpu().numpy().reshape(-1, 3)
            )

            random_indices = np.random.choice(
                pixels_for_clustering.shape[0],
                size=int(pixels_for_clustering.shape[0] * sampling_rate),
                replace=False,
            )
            pixels_for_clustering = pixels_for_clustering[random_indices]

            kmeans = KMeans(n_clusters=color_count, random_state=42, n_init=n_init)
            kmeans.fit(pixels_for_clustering)
            # バッチ内の各画像に対してラベルを取得 (scipy.cluster.vq.vq を使用)
            labels, dist = vq(
                pixels.cpu().numpy().reshape(-1, 3), kmeans.cluster_centers_
            )
            labels = torch.from_numpy(labels.reshape(batch_size, -1)).long().to(device)
            palette = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

        # Generate palette preview (最初の画像のみ, CPUで処理)
        preview_img = self.generate_palette_preview(palette.cpu().numpy(), font_size)

        # バッチ全体のインデックスマップを取得
        index_maps = labels.reshape(batch_size, height, width)

        # GPUで一括処理
        if lock_masks is not None:
            lock_masks_batch = []
            for i in range(batch_size):
                lock_mask = lock_masks[i] if i < lock_masks.shape[0] else lock_masks[-1]
                lock_masks_batch.append(lock_mask)
            lock_masks = torch.stack(lock_masks_batch).to(device)

            pixel_masks = torch.zeros_like(index_maps, dtype=torch.bool)
            for i in range(batch_size):
                for j, is_locked in enumerate(lock_masks[i]):
                    if is_locked:
                        pixel_masks[i][index_maps[i] == j] = True

            original_pixels = pixels.clone()  # ディープコピー
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

            draw.rectangle(
                [x, y, x + patch_size, y + patch_size],
                fill=tuple((color * 255).astype(int)),
            )

            text = str(i)
            bbox = draw.textbbox((x, y), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(
                (x + (patch_size - text_w) // 2, y + (patch_size - text_h) // 2),
                text,
                fill="white",
                font=font,
            )

        img_tensor = torch.from_numpy(
            np.array(img).astype(np.float32) / 255.0
        ).unsqueeze(0)
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
                "lock_color_num": (
                    "STRING",
                    {"default": "0,", "placeholder": "例: 0,1,2,3,4"},
                ),
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

    def process_palette(
        self,
        images,
        palette,
        index_maps,
        seed,
        hue_random_enable,
        hue_shift,
        saturation_random_enable,
        saturation_scale,
        value_random_enable,
        value_scale,
        lock_color_num="0,",
        mask_enable=False,
        invert_mask=False,
        operations="[]",
    ):
        device = images.device
        palette = palette.to(device)
        index_maps = index_maps.to(device)

        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]

        if not mask_enable:
            color_mask = torch.zeros(len(palette), dtype=torch.bool, device=device)
        else:
            try:
                indices = [
                    int(idx.strip()) for idx in lock_color_num.split(",") if idx.strip()
                ]
                color_mask = torch.zeros(len(palette), dtype=torch.bool, device=device)
                for idx in indices:
                    if 0 <= idx < len(palette):
                        color_mask[idx] = True
                if invert_mask:
                    color_mask = ~color_mask
            except ValueError:
                color_mask = torch.zeros(len(palette), dtype=torch.bool, device=device)

        current_hue = random.uniform(-180, 180) if hue_random_enable else hue_shift
        current_saturation = (
            random.uniform(0, 5) if saturation_random_enable else saturation_scale
        )
        current_value = random.uniform(0, 5) if value_random_enable else value_scale

        modified_palette = self.edit_palette(
            palette,
            operations,
            color_mask,
            current_hue,
            current_saturation,
            current_value,
        )

        # バッチ処理
        pixels = images.reshape(batch_size, -1, 3)
        labels = index_maps.reshape(batch_size, -1)
        new_pixels = modified_palette[labels.long()]
        original_pixels = pixels.clone()

        # ピクセルマスクの生成（2D形式）
        pixel_mask = torch.zeros(
            (batch_size, height, width), dtype=torch.float32, device=device
        )
        for i, is_masked in enumerate(color_mask):
            pixel_mask[index_maps == i] = 1.0 if is_masked else 0.0

        # 画像の処理
        pixels_mask = pixel_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3)
        processed_pixels = torch.where(pixels_mask > 0.5, original_pixels, new_pixels)
        processed_images = processed_pixels.reshape(images.shape)

        random.seed(None)
        return (processed_images, modified_palette, pixel_mask)

    def edit_palette(
        self, palette, operations, mask, hue_shift, saturation_scale, value_scale
    ):
        hsv_palette = self.rgb_to_hsv(palette)

        hsv_palette[~mask, 0] = (hsv_palette[~mask, 0] + hue_shift / 360.0) % 1.0
        hsv_palette[~mask, 1] = torch.clamp(
            hsv_palette[~mask, 1] * saturation_scale, 0, 1
        )
        hsv_palette[~mask, 2] = torch.clamp(hsv_palette[~mask, 2] * value_scale, 0, 1)

        try:
            ops = json.loads(operations)
            if isinstance(ops, list):
                for op in ops:
                    if isinstance(op, dict) and "index" in op:
                        idx = op["index"]
                        if 0 <= idx < len(palette):
                            if "color" in op and len(op["color"]) == 3:
                                new_color = torch.tensor(
                                    op["color"],
                                    dtype=torch.float32,
                                    device=palette.device,
                                )
                                palette[idx] = new_color
                                hsv_palette[idx] = self.rgb_to_hsv(
                                    new_color.reshape(1, 3)
                                )[0]
                            if "hsv" in op and len(op["hsv"]) == 3:
                                hsv_palette[idx] = torch.clamp(
                                    torch.tensor(op["hsv"], device=palette.device), 0, 1
                                )
        except Exception as e:
            print(f"Operation Error: {str(e)}")

        modified_palette = self.hsv_to_rgb(hsv_palette)
        return modified_palette


NODE_CLASS_MAPPINGS = {
    "ColorshiftColor": ColorshiftColorNode,
    "CsCPaletteEditor": PaletteEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorshiftColor": "ColorshiftColor",
    "CsCPaletteEditor": "CsCPaletteEditor",
}
