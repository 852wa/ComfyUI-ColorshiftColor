import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


def _validate_images(images):
    if (not isinstance(images, torch.Tensor) or images.ndim != 4
            or images.shape[-1] != 3 or any(size == 0 for size in images.shape)):
        raise ValueError("images must have shape [batch, height, width, 3] (RGB).")
    if not images.is_floating_point() or not torch.isfinite(images).all():
        raise ValueError("images must contain finite floating-point RGB values.")


def _validate_palette(palette):
    if (not isinstance(palette, torch.Tensor) or palette.ndim != 2
            or palette.shape[0] == 0 or palette.shape[1] != 3):
        raise ValueError("palette must have shape [color_count, 3] and contain at least one color.")
    if (not torch.isfinite(palette).all()
            or ((palette < 0) | (palette > 1)).any()):
        raise ValueError("palette RGB values must be finite and between 0 and 1.")


def _prepare_indexed_inputs(images, palette, index_maps):
    _validate_images(images)
    _validate_palette(palette)
    if not isinstance(index_maps, torch.Tensor) or index_maps.shape != images.shape[:3]:
        raise ValueError("index_maps must match images: [batch, height, width].")
    if (not torch.isfinite(index_maps).all() or (index_maps < 0).any()
            or (index_maps >= len(palette)).any()
            or (index_maps.is_floating_point() and (index_maps != index_maps.round()).any())):
        raise ValueError("index_maps must contain integer palette indices; connect ColorshiftColor's index_maps output.")
    return (palette.to(device=images.device, dtype=torch.float32),
            index_maps.to(device=images.device, dtype=torch.long))


def _parse_operations(operations):
    if not isinstance(operations, str):
        raise ValueError("operations must be a JSON list.")
    try:
        # Shipped workflows disable example operations with a leading ## line comment.
        active_lines = "\n".join(line for line in operations.splitlines()
                                 if not line.lstrip().startswith("##"))
        result = json.loads(active_lines.strip() or "[]")
    except ValueError as exc:
        raise ValueError(f"operations contains invalid JSON: {exc}") from exc
    if not isinstance(result, list) or any(not isinstance(op, dict) for op in result):
        raise ValueError("operations must be a JSON list of objects.")
    return result


def _color_index(value, count, name):
    if type(value) is not int or not 0 <= value < count:
        raise ValueError(f"{name} must be an integer between 0 and {count - 1}.")
    return value


def _color_triplet(value, palette, name):
    if (not isinstance(value, list) or len(value) != 3
            or any(type(component) not in (int, float) for component in value)):
        raise ValueError(f"{name} must contain three numbers between 0 and 1.")
    color = palette.new_tensor(value)
    if not torch.isfinite(color).all() or ((color < 0) | (color > 1)).any():
        raise ValueError(f"{name} must contain three finite numbers between 0 and 1.")
    return color


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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "PALETTE", "MASK", "IMAGE")
    RETURN_NAMES = ("images", "palette", "index_maps", "palette_preview")
    FUNCTION = "process"
    CATEGORY = "Image/Color"

    @torch.no_grad()
    def process(self, images, color_count, lock_masks=None, palette_override=None, font_size=20, sampling_rate=0.25, n_init=3, seed=0):
        _validate_images(images)
        if type(color_count) is not int or not 2 <= color_count <= 64:
            raise ValueError("color_count must be an integer between 2 and 64.")
        if not 0.01 <= sampling_rate <= 0.5:
            raise ValueError("sampling_rate must be between 0.01 and 0.5.")
        if type(n_init) is not int or not 1 <= n_init <= 10:
            raise ValueError("n_init must be an integer between 1 and 10.")
        batch_size, height, width, channels = images.shape
        pixels = images.reshape(batch_size, -1, 3)

        if palette_override is not None:
            _validate_palette(palette_override)
            palette = palette_override.to(device=images.device, dtype=torch.float32).clone()
            labels = self._match_palette_batch(pixels, palette)
        else:
            # Sample original pixels so resizing cannot invent intermediate colors.
            rng = np.random.default_rng(seed)
            flat_pixels = pixels.reshape(-1, 3)
            sample_count = min(len(flat_pixels), 262144, max(color_count, int(len(flat_pixels) * sampling_rate)))
            random_indices = torch.as_tensor(
                rng.choice(len(flat_pixels), size=sample_count, replace=False), device=images.device)
            samples = flat_pixels[random_indices].float().clamp(0, 1).cpu().numpy()
            unique_colors, weights = np.unique(samples, axis=0, return_counts=True)
            cluster_count = min(color_count, len(unique_colors))
            if cluster_count == len(unique_colors):
                centers = unique_colors
            else:
                kmeans = KMeans(n_clusters=cluster_count, random_state=int(seed) % (2**32), n_init=n_init)
                kmeans.fit(unique_colors, sample_weight=weights)
                centers = kmeans.cluster_centers_
            palette = torch.as_tensor(centers, device=images.device, dtype=torch.float32).clamp(0, 1)
            labels = self._match_palette_batch(pixels, palette)

            # ----- パレットの並び順を各クラスタの占有率が高い順に変更 -----
            # 全ラベルを1次元にまとめ、各クラスタの出現回数を計算
            all_labels = labels.flatten()
            counts = torch.bincount(all_labels, minlength=len(palette))
            # 出現回数の降順に並べ替えたときの各クラスタの元のインデックス
            sorted_order = torch.argsort(counts, descending=True, stable=True)
            # mapping: 旧インデックス -> 新インデックス を作成
            mapping = torch.empty_like(sorted_order)
            mapping[sorted_order] = torch.arange(sorted_order.size(0), device=sorted_order.device)
            # パレットを並び替え、各画素のラベルを新しい順序に置換
            palette = palette[sorted_order]
            labels = mapping[labels]
            # ---------------------------------------------------------------

        preview_img = self.generate_palette_preview(palette, font_size)
        index_maps = labels.reshape(batch_size, height, width)
        processed_images_tensor = palette[labels].reshape(images.shape).to(images.dtype)
        if lock_masks is not None:
            if not isinstance(lock_masks, torch.Tensor) or lock_masks.ndim not in (2, 3):
                raise ValueError("lock_masks must have shape [height, width] or [batch, height, width].")
            if any(size == 0 for size in lock_masks.shape) or not torch.isfinite(lock_masks).all():
                raise ValueError("lock_masks must be nonempty and contain finite values.")
            masks = lock_masks.to(device=images.device, dtype=torch.float32)
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if masks.shape[1:] != (height, width):
                masks = F.interpolate(masks.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False).squeeze(1)
            batch_indices = torch.arange(batch_size, device=images.device).clamp(max=len(masks) - 1)
            masks = masks[batch_indices].clamp(0, 1).unsqueeze(-1).to(images.dtype)
            processed_images_tensor = images * masks + processed_images_tensor * (1 - masks)
        return (processed_images_tensor, palette, index_maps, preview_img)

    def generate_palette_preview(self, palette, font_size):
        if isinstance(palette, torch.Tensor):
            colors = palette.detach().float().cpu().numpy()
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

            draw.rectangle([x, y, x + patch_size - 1, y + patch_size - 1], fill=tuple((color * 255).clip(0, 255).astype(int)))
            text = str(i)
            bbox = draw.textbbox((x, y), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_color = "black" if np.dot(color, [0.2126, 0.7152, 0.0722]) > 0.5 else "white"
            draw.text((x + (patch_size - text_w) // 2 - bbox[0], y + (patch_size - text_h) // 2 - bbox[1]), text, fill=text_color, font=font)

        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return img_tensor

    def _match_palette_batch(self, pixels, palette):
        # Bound the temporary distance matrix, including for large override palettes.
        flat_pixels = pixels.reshape(-1, 3)
        palette = palette.to(device=pixels.device, dtype=torch.float32)
        chunk_size = max(1, min(65536, 4_194_304 // len(palette)))
        labels = torch.empty(len(flat_pixels), device=pixels.device, dtype=torch.long)
        for start in range(0, len(flat_pixels), chunk_size):
            chunk = flat_pixels[start:start + chunk_size].float()
            distances = torch.cdist(chunk, palette, compute_mode="donot_use_mm_for_euclid_dist")
            labels[start:start + chunk_size] = distances.argmin(dim=-1)
        return labels.reshape(pixels.shape[:-1])

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

    @torch.no_grad()
    def process_palette(self, images, palette, index_maps, seed=0, hue_random_enable=False, hue_shift=0.0, saturation_random_enable=False, saturation_scale=1.0, value_random_enable=False, value_scale=1.0, lock_color_num="0,", mask_enable=False, invert_mask=False, operations="[]"):
        palette, index_maps = _prepare_indexed_inputs(images, palette, index_maps)
        color_mask = torch.zeros(len(palette), dtype=torch.bool, device=images.device)
        if mask_enable:
            try:
                indices = [int(idx.strip()) for idx in lock_color_num.split(",") if idx.strip()]
            except ValueError as exc:
                raise ValueError("lock_color_num must contain comma-separated integer color indices.") from exc
            for idx in indices:
                color_mask[_color_index(idx, len(palette), "lock_color_num")] = True
            if invert_mask:
                color_mask = ~color_mask

        rng = random.Random(seed)
        current_hue = rng.uniform(-180, 180) if hue_random_enable else hue_shift
        current_saturation = rng.uniform(0, 5) if saturation_random_enable else saturation_scale
        current_value = rng.uniform(0, 5) if value_random_enable else value_scale

        modified_palette = self.edit_palette(palette, operations, color_mask, current_hue, current_saturation, current_value)

        pixel_mask = color_mask[index_maps]
        new_pixels = modified_palette[index_maps].to(images.dtype)
        processed_images = torch.where(pixel_mask.unsqueeze(-1), images, new_pixels)
        return (processed_images, modified_palette, pixel_mask.float())

    def edit_palette(self, palette, operations, mask, hue_shift, saturation_scale, value_scale):
        _validate_palette(palette)
        palette = palette.clone()
        mask = mask.to(device=palette.device, dtype=torch.bool)
        hsv_palette = self.rgb_to_hsv(palette)

        hsv_palette[~mask, 0] = (hsv_palette[~mask, 0] + hue_shift / 360.0) % 1.0
        hsv_palette[~mask, 1] = torch.clamp(hsv_palette[~mask, 1] * saturation_scale, 0, 1)
        hsv_palette[~mask, 2] = torch.clamp(hsv_palette[~mask, 2] * value_scale, 0, 1)

        modified_palette = self.hsv_to_rgb(hsv_palette).clamp(0, 1)
        modified_palette[mask] = palette[mask]
        for op in _parse_operations(operations):
            idx = _color_index(op.get("index"), len(palette), "operations.index")
            if "color" not in op and "hsv" not in op:
                raise ValueError("Each palette operation needs color or hsv.")
            rgb = _color_triplet(op["color"], palette, "operations.color") if "color" in op else None
            hsv = _color_triplet(op["hsv"], palette, "operations.hsv") if "hsv" in op else None
            if rgb is not None:
                modified_palette[idx] = rgb
            if hsv is not None:
                modified_palette[idx] = self.hsv_to_rgb(hsv.reshape(1, 3))[0]

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

    @torch.no_grad()
    def process(self, images, palette, index_maps, operations="[]"):
        palette, index_maps = _prepare_indexed_inputs(images, palette, index_maps)
        batch_size, height, width, channels = images.shape
        ops = _parse_operations(operations)
        pairs = [(_color_index(op.get("A"), len(palette), "operations.A"),
                  _color_index(op.get("B"), len(palette), "operations.B")) for op in ops]
        if not ops:
            pairs = self._compute_auto_pairs(palette)

        # Resolve the palette mapping first; duplicate B entries use the last operation.
        replacements = torch.arange(len(palette), device=images.device)
        selected_colors = torch.zeros(len(palette), device=images.device, dtype=torch.bool)
        for a_idx, b_idx in pairs:
            replacements[b_idx] = a_idx
            selected_colors[b_idx] = a_idx != b_idx
        selected = selected_colors[index_maps]
        filled_image = torch.where(selected.unsqueeze(-1),
                                   palette[replacements[index_maps]].to(images.dtype), images)
        mask_image = images.new_zeros((batch_size, height, width, 4))
        mask_image[..., :3] = torch.where(selected.unsqueeze(-1), palette[index_maps].to(images.dtype), 0)
        mask_image[..., 3] = selected.to(images.dtype)
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
