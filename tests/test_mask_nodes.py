import importlib.util
import pathlib
import unittest
from unittest import mock

import torch


MODULE_PATH = pathlib.Path(__file__).parents[1] / "nodes" / "mask_nodes.py"
SPEC = importlib.util.spec_from_file_location("stdismas_mask_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def reference_expand(mask, top, bottom, left, right):
    """Simple shift-and-maximum reference with fixed canvas dimensions."""
    source = mask.clamp(0.0, 1.0)
    if source.ndim == 2:
        source = source.unsqueeze(0)

    batch, height, width = source.shape
    result = torch.zeros_like(source)
    for y_shift in range(-top, bottom + 1):
        src_y0 = max(0, -y_shift)
        src_y1 = min(height, height - y_shift)
        dst_y0 = src_y0 + y_shift
        dst_y1 = src_y1 + y_shift
        for x_shift in range(-left, right + 1):
            src_x0 = max(0, -x_shift)
            src_x1 = min(width, width - x_shift)
            dst_x0 = src_x0 + x_shift
            dst_x1 = src_x1 + x_shift
            result[:, dst_y0:dst_y1, dst_x0:dst_x1] = torch.maximum(
                result[:, dst_y0:dst_y1, dst_x0:dst_x1],
                source[:, src_y0:src_y1, src_x0:src_x1],
            )
    return result


class ExpandMaskBySidesTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.ExpandMaskBySides()

    def run_node(self, mask, top=0, bottom=0, left=0, right=0):
        return self.node.expand_mask(mask, top, bottom, left, right)[0]

    def test_each_control_expands_the_named_side(self):
        mask = torch.zeros((1, 7, 9), dtype=torch.float32)
        mask[0, 3, 4] = 1.0
        cases = {
            "top": ((2, 0, 0, 0), (1, 3, 4, 4)),
            "bottom": ((0, 2, 0, 0), (3, 5, 4, 4)),
            "left": ((0, 0, 2, 0), (3, 3, 2, 4)),
            "right": ((0, 0, 0, 2), (3, 3, 4, 6)),
        }

        for name, (amounts, expected_bbox) in cases.items():
            with self.subTest(side=name):
                result = self.run_node(mask, *amounts)[0]
                ys, xs = torch.where(result > 0)
                bbox = (
                    int(ys.min()),
                    int(ys.max()),
                    int(xs.min()),
                    int(xs.max()),
                )
                self.assertEqual(bbox, expected_bbox)
                self.assertEqual(float(result.sum()), 3.0)

    def test_asymmetric_expansion_matches_exact_requested_bbox(self):
        mask = torch.zeros((1, 9, 11), dtype=torch.float32)
        mask[0, 4, 5] = 1.0

        result = self.run_node(mask, top=1, bottom=2, left=3, right=1)[0]
        ys, xs = torch.where(result > 0)

        self.assertEqual(
            (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            (3, 6, 2, 6),
        )
        self.assertEqual(float(result.sum()), 20.0)

    def test_equal_expansion_adds_the_same_distance_on_every_side(self):
        mask = torch.zeros((1, 51, 51), dtype=torch.float32)
        mask[0, 25, 25] = 1.0

        result = self.run_node(mask, top=20, bottom=20, left=20, right=20)[0]
        ys, xs = torch.where(result > 0)

        self.assertEqual(
            (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            (5, 45, 5, 45),
        )
        self.assertEqual(float(result.sum()), 41.0 * 41.0)

    def test_binary_and_soft_batches_match_reference(self):
        mask = torch.zeros((3, 8, 10), dtype=torch.float32)
        mask[0, 2:4, 3:6] = 1.0
        mask[1, 1, 1] = 0.25
        mask[1, 5, 7] = 0.8
        mask[2, 0, 9] = 1.0

        result = self.run_node(mask, top=2, bottom=1, left=1, right=3)
        expected = reference_expand(mask, 2, 1, 1, 3)

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, mask.shape)
        self.assertEqual(result.dtype, mask.dtype)

    def test_expansion_is_clipped_to_canvas_edges(self):
        mask = torch.zeros((1, 4, 5), dtype=torch.float32)
        mask[0, 0, 0] = 1.0

        result = self.run_node(mask, top=3, bottom=2, left=4, right=2)
        expected = reference_expand(mask, 3, 2, 4, 2)

        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, mask.shape)

    def test_noop_does_not_mutate_input(self):
        mask = torch.tensor([[[-1.0, 0.5, 2.0]]], dtype=torch.float32)
        original = mask.clone()

        result = self.run_node(mask)

        self.assertTrue(torch.equal(mask, original))
        self.assertTrue(torch.equal(result, torch.tensor([[[0.0, 0.5, 1.0]]])))

    def test_two_axes_use_only_one_dimensional_pooling_kernels(self):
        mask = torch.zeros((4, 16, 16), dtype=torch.float32)
        mask[:, 8, 8] = 1.0

        with mock.patch.object(
            MODULE.F,
            "max_pool2d",
            wraps=MODULE.F.max_pool2d,
        ) as pool:
            self.run_node(mask, top=2, bottom=3, left=4, right=5)

        kernels = [call.kwargs["kernel_size"] for call in pool.call_args_list]
        self.assertEqual(kernels, [(1, 10), (6, 1)])


if __name__ == "__main__":
    unittest.main()
