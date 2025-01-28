# nodes.py
import torch
import numpy as np
from sklearn.cluster import KMeans
import colorsys
import json
import random
from PIL import Image, ImageDraw, ImageFont
import io
import time

class ColorshiftColorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_count": ("INT", {"default": 8, "min": 2, "max": 32}),
            },
            "optional": {
                "lock_mask": ("MASK", {"default": None}),
                "palette_override": ("PALETTE", {"default": None}),
                "font_size": ("INT", {"default": 20, "min": 10, "max": 50}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "PALETTE", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "palette", "index_map", "palette_preview")
    FUNCTION = "process"
    CATEGORY = "Image/Color"

    def process(self, image, color_count, lock_mask=None, palette_override=None, font_size=20):
        # Convert torch tensor to numpy array
        img_np = image.numpy().squeeze()
        height, width = img_np.shape[:2]
        pixels = img_np.reshape(-1, 3)

        if palette_override is not None:
            if hasattr(palette_override, 'numpy'):
                palette = palette_override.numpy()
            else:
                palette = palette_override
            labels = self._match_palette(pixels, palette)
        else:
            kmeans = KMeans(n_clusters=color_count, random_state=42)
            labels = kmeans.fit_predict(pixels)
            palette = kmeans.cluster_centers_

        # Generate palette preview
        preview_img = self.generate_palette_preview(palette, font_size)

        # Get the unique index map first
        index_map = labels.reshape(height, width)
        index_map_tensor = torch.from_numpy(index_map.astype(np.float32))

        if lock_mask is not None:
            if hasattr(lock_mask, 'numpy'):
                mask = lock_mask.numpy()
            else:
                mask = lock_mask
            
            pixel_mask = np.zeros_like(labels, dtype=bool)
            for i, is_locked in enumerate(mask):
                if is_locked:
                    pixel_mask[labels == i] = True
            
            original_pixels = pixels.copy()
            new_pixels = palette[labels]
            pixels[~pixel_mask] = new_pixels[~pixel_mask]
            pixels[pixel_mask] = original_pixels[pixel_mask]
        else:
            pixels = palette[labels]

        processed_image = torch.from_numpy(pixels.reshape((1, height, width, 3))).float()
        return (processed_image, torch.from_numpy(palette), index_map_tensor, preview_img)

    def generate_palette_preview(self, palette, font_size):
        if isinstance(palette, torch.Tensor):
            colors = palette.numpy()
        else:
            colors = palette
            
        patch_size = 100
        cols = 8
        rows = (len(colors) + cols - 1) // cols
        
        img = Image.new("RGB", (cols*patch_size, rows*patch_size), (40,40,40))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for i, color in enumerate(colors):
            x = (i % cols) * patch_size
            y = (i // cols) * patch_size
            
            draw.rectangle([x, y, x+patch_size, y+patch_size], 
                         fill=tuple((color * 255).astype(int)))
            
            text = str(i)
            bbox = draw.textbbox((x, y), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x + (patch_size-text_w)//2, y + (patch_size-text_h)//2),
                    text, fill="white", font=font)
        
        img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return img_tensor

    def _match_palette(self, pixels, palette):
        distances = np.linalg.norm(pixels[:, np.newaxis] - palette, axis=2)
        return np.argmin(distances, axis=1)

class PaletteEditorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette": ("PALETTE",),
                "index_map": ("MASK",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
            }
        }
    
    RETURN_TYPES = ("IMAGE", "PALETTE", "MASK")
    RETURN_NAMES = ("image", "palette", "mask")
    FUNCTION = "process_palette"
    CATEGORY = "Image/Color"

    def rgb_to_hsv(self, rgb):
        return np.array([colorsys.rgb_to_hsv(*c) for c in rgb])

    def hsv_to_rgb(self, hsv):
        return np.array([colorsys.hsv_to_rgb(*c) for c in hsv])
        
    def process_palette(self, image, palette, index_map,
                   seed,
                   hue_random_enable, hue_shift,
                   saturation_random_enable, saturation_scale,
                   value_random_enable, value_scale,
                   lock_color_num="0,", mask_enable=False, 
                   invert_mask=False, mask_type="lock", operations="[]"):
        # If mask is not enabled, apply changes to the entire palette
        if not mask_enable:
            mask = np.zeros(len(palette), dtype=bool)
        else:
            try:
                indices = [int(idx.strip()) for idx in lock_color_num.split(",") if idx.strip()]
                mask = np.zeros(len(palette), dtype=bool)
                for idx in indices:
                    if 0 <= idx < len(palette):
                        mask[idx] = True
                if invert_mask:
                    mask = ~mask
            except:
                mask = np.zeros(len(palette), dtype=bool)

        current_hue = random.uniform(-180, 180) if hue_random_enable else hue_shift
        current_saturation = random.uniform(0, 5) if saturation_random_enable else saturation_scale
        current_value = random.uniform(0, 5) if value_random_enable else value_scale

        modified_palette = self.edit_palette(palette, operations, mask,
                                          current_hue, current_saturation, current_value)
        
        # Apply the modified palette to the image
        processed_image = self.apply_palette(image, modified_palette, index_map, mask, mask_type)
        
        random.seed(None)
        return (processed_image, modified_palette, mask)

    def edit_palette(self, palette, operations, mask,
                    hue_shift, saturation_scale, value_scale):
        palette_np = palette.numpy()
        hsv_palette = self.rgb_to_hsv(palette_np)

        # Apply HSV adjustments to unmasked colors
        hsv_palette[~mask, 0] = (hsv_palette[~mask, 0] + hue_shift/360.0) % 1.0
        hsv_palette[~mask, 1] = np.clip(hsv_palette[~mask, 1] * saturation_scale, 0, 1)
        hsv_palette[~mask, 2] = np.clip(hsv_palette[~mask, 2] * value_scale, 0, 1)

        try:
            ops = json.loads(operations)
            if isinstance(ops, list):
                for op in ops:
                    if isinstance(op, dict) and "index" in op:
                        idx = op["index"]
                        if 0 <= idx < len(palette_np):
                            if "color" in op and len(op["color"]) == 3:
                                new_color = np.array(op["color"], dtype=np.float32)
                                palette_np[idx] = new_color
                                hsv_palette[idx] = self.rgb_to_hsv(new_color.reshape(1,3))[0]
                            if "hsv" in op and len(op["hsv"]) == 3:
                                hsv_palette[idx] = np.clip(op["hsv"], [0,0,0], [1,1,1])
        except Exception as e:
            print(f"Operation Error: {str(e)}")

        modified_palette = self.hsv_to_rgb(hsv_palette)
        return torch.from_numpy(modified_palette).float()

    def apply_palette(self, image, palette, index_map, mask, mask_type="lock"):
        img_np = image.numpy().squeeze()
        height, width = img_np.shape[:2]
        pixels = img_np.reshape(-1, 3)
        
        if hasattr(palette, 'numpy'):
            palette = palette.numpy()
        
        index_map = index_map.numpy() if hasattr(index_map, 'numpy') else index_map
        color_mask = mask.numpy() if hasattr(mask, 'numpy') else mask
        
        labels = index_map.reshape(-1)
        new_pixels = palette[labels.astype(int)]
        original_pixels = pixels.copy()
        
        # Check if mask is entirely False (mask disabled)
        if np.all(~color_mask):
            # Apply new_pixels to all pixels
            pixels = new_pixels
        else:
            # Create pixel_mask based on color_mask
            pixel_mask = np.zeros_like(labels, dtype=bool)
            for i, is_masked in enumerate(color_mask):
                if is_masked:
                    pixel_mask[labels == i] = True
            
            if mask_type == "change":
                # Apply new_pixels to masked areas
                pixels = np.where(pixel_mask[:, np.newaxis], new_pixels, original_pixels)
            else:
                # Apply new_pixels to non-masked areas
                pixels = np.where(pixel_mask[:, np.newaxis], original_pixels, new_pixels)

        processed_image = torch.from_numpy(pixels.reshape((1, height, width, 3))).float()
        return processed_image

NODE_CLASS_MAPPINGS = {
    "ColorshiftColor": ColorshiftColorNode,
    "CsCPaletteEditor": PaletteEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorshiftColor": "ColorshiftColor",
    "CsCPaletteEditor": "CsCPaletteEditor",
}