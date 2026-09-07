import random
import colorsys
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nodes import ColorshiftColorNode, CsCFill, PaletteEditorNode


torch.set_num_threads(1)


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.palette = torch.tensor([[1., 0., 0.], [0., 0., 1.]])
        self.labels = torch.tensor([[[0, 1], [1, 0]]])
        self.images = self.palette[self.labels]

    def edit(self, **kwargs):
        options = dict(seed=123, hue_random_enable=False, hue_shift=0.,
                       saturation_random_enable=False, saturation_scale=1.,
                       value_random_enable=False, value_scale=1.)
        options.update(kwargs)
        return PaletteEditorNode().process_palette(
            self.images, self.palette, self.labels, **options)

    def test_edit_does_not_mutate_upstream_palette(self):
        original = self.palette.clone()
        self.edit(operations='[{"index": 0, "color": [0, 1, 0]}]')
        torch.testing.assert_close(self.palette, original)
        torch.testing.assert_close(self.edit()[0], self.images)

    def test_random_edit_repeats_with_seed(self):
        first = self.edit(hue_random_enable=True)[0]
        second = self.edit(hue_random_enable=True)[0]
        torch.testing.assert_close(first, second)

    def test_random_edit_preserves_global_rng(self):
        state = random.getstate()
        self.edit(hue_random_enable=True)
        self.assertEqual(random.getstate(), state)

    def test_tiny_image_can_use_large_palette_request(self):
        image = torch.tensor([[[[.2, .4, .6]]]])
        output, palette, labels, _ = ColorshiftColorNode().process(image, 64)
        torch.testing.assert_close(output, image)
        self.assertEqual(palette.shape, (1, 3))
        self.assertEqual(labels.item(), 0)

    def test_spatial_mask_preserves_original_without_mutation(self):
        image = self.images * .7
        original = image.clone()
        mask = torch.tensor([[[1., 0.], [.5, 0.]]])
        output, _, _, _ = ColorshiftColorNode().process(
            image, 2, lock_masks=mask, palette_override=self.palette)
        expected = image * mask[..., None] + self.images * (1 - mask[..., None])
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(image, original)

    def test_fill_honors_explicit_direction_even_when_target_is_darker(self):
        palette = torch.tensor([[.1, .1, .1], [.9, .9, .9]])
        images = palette[self.labels]
        filled, shadow = CsCFill().process(
            images, palette, self.labels, '[{"A": 0, "B": 1}]')
        torch.testing.assert_close(filled, palette[0].expand_as(images))
        torch.testing.assert_close(shadow[..., 3], (self.labels == 1).float())

    def test_random_edit_changes_with_seed(self):
        self.assertFalse(torch.equal(self.edit(seed=1, hue_random_enable=True)[1],
                                     self.edit(seed=2, hue_random_enable=True)[1]))

    def test_mask_inversion_and_explicit_palette_override(self):
        for invert in (False, True):
            with self.subTest(invert=invert):
                result, palette, mask = self.edit(mask_enable=True, lock_color_num="0,",
                                                 invert_mask=invert, hue_shift=120)
                expected = (self.labels == (1 if invert else 0))
                torch.testing.assert_close(mask, expected.float())
                torch.testing.assert_close(result[expected], self.images[expected])
                torch.testing.assert_close(palette[int(invert)], self.palette[int(invert)])
        result, palette, mask = self.edit(mask_enable=True,
            operations='[{"index": 0, "color": [0, 1, 0]}]')
        torch.testing.assert_close(result[self.labels == 0], self.images[self.labels == 0])
        torch.testing.assert_close(palette[0], torch.tensor([0., 1., 0.]))

    def test_hsv_conversion_matches_standard_library(self):
        colors = torch.cat((torch.rand(100, 3, generator=torch.Generator().manual_seed(9)),
                            torch.tensor([[0., 0., 0.], [1., 1., 1.], [.5, .5, .5]])))
        editor = PaletteEditorNode()
        hsv = editor.rgb_to_hsv(colors)
        expected = torch.tensor([colorsys.rgb_to_hsv(*color.tolist()) for color in colors])
        torch.testing.assert_close(hsv, expected)
        torch.testing.assert_close(editor.hsv_to_rgb(hsv), colors)

    def test_hsv_operations_and_rgb_precedence(self):
        _, palette, _ = self.edit(operations='[{"index": 0, "color": [1, 0, 0], "hsv": [0.5, 1, 1]}]')
        torch.testing.assert_close(palette[0], torch.tensor([0., 1., 1.]))

    def test_invalid_operations_raise_without_mutation(self):
        invalid = ['oops', '{}', '[null]', '[{}]', '[{"index": -1, "color": [0,0,0]}]',
                   '[{"index": 0.5, "color": [0,0,0]}]', '[{"index": true, "color": [0,0,0]}]',
                   '[{"index": 0, "color": [2,0,0]}]', '[{"index": 0, "color": [NaN,0,0]}]',
                   '[{"index": 0, "color": [0,0]}]', '[{"index": 0, "color": ["0",0,0]}]',
                   '[{"index": 0, "hsv": null}]', '[{"index": 0}]',
                   '[{"index": 0, "color": [0,1,0]}, {"index": 999, "color": [0,0,0]}]']
        for operation in invalid:
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'operations|palette operation'):
                self.edit(operations=operation)
        torch.testing.assert_close(self.images, self.palette[self.labels])

    def test_invalid_locks_are_actionable(self):
        for locks in ('x', '-1', '999'):
            with self.subTest(locks=locks), self.assertRaisesRegex(ValueError, 'lock_color_num'):
                self.edit(mask_enable=True, lock_color_num=locks)

    def test_bundled_editor_workflows_and_comments(self):
        for path in (Path(__file__).resolve().parents[1] / 'example').glob('*.json'):
            for node in json.loads(path.read_text(encoding='utf8'))['nodes']:
                if node['type'] == 'CsCPaletteEditor':
                    with self.subTest(workflow=path.name):
                        # The supplied commented example must behave exactly like [].
                        torch.testing.assert_close(self.edit(operations=node['widgets_values'][-1])[0], self.images)
        torch.testing.assert_close(self.edit(operations='  ')[0], self.images)

    def test_quantization_seed_and_global_numpy_rng(self):
        image = torch.rand(2, 32, 32, 3, generator=torch.Generator().manual_seed(10))
        state = np.random.get_state()
        first = ColorshiftColorNode().process(image, 4, seed=2**64 - 1)
        second = ColorshiftColorNode().process(image, 4, seed=2**64 - 1)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_solid_and_thin_images(self):
        for shape in ((1, 1, 20, 3), (1, 20, 1, 3), (2, 4, 4, 3)):
            with self.subTest(shape=shape):
                image = torch.full(shape, .25)
                output, palette, labels, _ = ColorshiftColorNode().process(image, 64, sampling_rate=.01)
                torch.testing.assert_close(output, image)
                self.assertEqual(len(palette), 1)
                self.assertEqual(labels.max().item(), 0)

    def test_palette_is_sorted_by_coverage(self):
        image = torch.tensor([[[[1., 0., 0.]] * 3 + [[0., 0., 1.]]]])
        output, palette, labels, _ = ColorshiftColorNode().process(image, 64)
        torch.testing.assert_close(output, image)
        torch.testing.assert_close(palette[0], torch.tensor([1., 0., 0.]))
        self.assertEqual(labels.tolist(), [[[0, 0, 0, 1]]])

    def test_palette_override_order_and_float16_images(self):
        image = self.images.half().transpose(1, 2)
        result, palette, labels, _ = ColorshiftColorNode().process(image, 8, palette_override=self.palette)
        torch.testing.assert_close(result, image)
        torch.testing.assert_close(palette, self.palette)
        self.assertEqual(labels.dtype, torch.long)

    def test_palette_matching_is_chunked_and_exact(self):
        pixels = torch.rand(2, 40000, 3, generator=torch.Generator().manual_seed(12))
        original_cdist = torch.cdist
        with patch('nodes.torch.cdist', wraps=original_cdist) as calls:
            labels = ColorshiftColorNode()._match_palette_batch(pixels, self.palette)
        self.assertGreater(calls.call_count, 1)
        self.assertLessEqual(max(call.args[0].shape[0] for call in calls.call_args_list), 65536)
        distances = ((pixels.unsqueeze(-2) - self.palette) ** 2).sum(-1)
        torch.testing.assert_close(labels, distances.argmin(-1))

    def test_mask_broadcast_resize_and_repeat_last(self):
        images = (self.images * .7).repeat(3, 1, 1, 1)
        masks = torch.tensor([[[1.]], [[0.]]])
        output = ColorshiftColorNode().process(images, 2, lock_masks=masks, palette_override=self.palette)[0]
        torch.testing.assert_close(output[0], images[0])
        torch.testing.assert_close(output[1:], self.images.expand(2, -1, -1, -1))
        output = ColorshiftColorNode().process(images, 2, lock_masks=torch.ones(2, 2), palette_override=self.palette)[0]
        torch.testing.assert_close(output, images)

    def test_invalid_spatial_masks(self):
        for mask in (torch.empty(0, 2, 2), torch.ones(2), torch.full((2, 2), float('nan'))):
            with self.subTest(shape=mask.shape), self.assertRaisesRegex(ValueError, 'lock_masks'):
                ColorshiftColorNode().process(self.images, 2, palette_override=self.palette, lock_masks=mask)

    def test_invalid_images_and_palettes(self):
        for image in (torch.ones(1, 2, 2, 4), torch.empty(0, 2, 2, 3), self.images.long(), self.images * float('nan')):
            with self.subTest(shape=image.shape), self.assertRaisesRegex(ValueError, 'images'):
                ColorshiftColorNode().process(image, 2)
        for palette in (torch.empty(0, 3), torch.ones(3), self.palette * float('inf'), self.palette * 2):
            with self.subTest(palette=palette), self.assertRaisesRegex(ValueError, 'palette'):
                ColorshiftColorNode().process(self.images, 2, palette_override=palette)

    def test_invalid_index_maps_in_both_consumers(self):
        for labels in (self.labels - 1, self.labels + 2, self.labels.float() + .5,
                       self.labels.float() * float('nan'), self.labels[..., :1]):
            for process in (PaletteEditorNode().process_palette, CsCFill().process):
                with self.subTest(labels=labels, process=process), self.assertRaisesRegex(ValueError, 'index_maps'):
                    process(self.images, self.palette, labels)

    def test_fill_rejects_invalid_manual_pairs_instead_of_auto_filling(self):
        for operation in ('bad json', '{}', '[{}]', '[{"A": 0, "B": 999}]', '[{"A": 0.5, "B": 1}]'):
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, 'operations'):
                CsCFill().process(self.images, self.palette, self.labels, operation)

    def test_fill_shadow_rgb_and_transparent_background(self):
        _, shadow = CsCFill().process(self.images, self.palette, self.labels, '[{"A": 0, "B": 1}]')
        selected = self.labels == 1
        torch.testing.assert_close(shadow[..., :3][selected], self.images[selected])
        self.assertEqual(torch.count_nonzero(shadow[~selected]).item(), 0)

    def test_fill_auto_pairs_odd_palette_and_self_pair(self):
        palette = torch.tensor([[.1, .1, .1], [.2, .2, .2], [1., 0., 0.]])
        self.assertEqual(CsCFill()._compute_auto_pairs(palette), [(1, 0)])
        output, shadow = CsCFill().process(self.images, self.palette, self.labels,
                                           '[{"A": 0, "B": 1}, {"A": 1, "B": 1}]')
        torch.testing.assert_close(output, self.images)
        self.assertEqual(torch.count_nonzero(shadow).item(), 0)

    def test_preview_has_readable_text_on_white(self):
        preview = ColorshiftColorNode().generate_palette_preview(torch.ones(1, 3), 20)
        self.assertEqual(preview.shape, (1, 100, 800, 3))
        self.assertLess(preview[0, 25:75, 25:75].min().item(), .5)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is not available')
    def test_all_nodes_on_cuda_with_cpu_auxiliary_inputs(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                images = self.images.to(device='cuda', dtype=dtype)
                output, palette, labels, _ = ColorshiftColorNode().process(
                    images, 2, palette_override=self.palette, lock_masks=torch.ones(2, 2))
                torch.testing.assert_close(output, images)
                result, edited, mask = PaletteEditorNode().process_palette(
                    images, self.palette, self.labels, operations='[{"index": 0, "color": [0, 1, 0]}]')
                filled, shadow = CsCFill().process(images, self.palette, self.labels, '[{"A": 0, "B": 1}]')
                for value in (output, palette, labels, result, edited, mask, filled, shadow):
                    self.assertEqual(value.device.type, 'cuda')
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(shadow.dtype, dtype)
        generated = ColorshiftColorNode().process(self.images.cuda(), 64)
        torch.testing.assert_close(generated[0], self.images.cuda())


if __name__ == '__main__':
    unittest.main()
