import importlib.util
import pathlib
import unittest
from unittest import mock

import torch


MODULE_PATH = (
    pathlib.Path(__file__).parents[1]
    / "nodes"
    / "batch_crop_uncrop_by_mask_advanced_optimized.py"
)
SPEC = importlib.util.spec_from_file_location("stdismas_crop_nodes", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def run_crop(images, crop_mask, **overrides):
    options = dict(
        aspect_ratio="1:1",
        output_long_side=32,
        use_long_side=True,
        use_custom_resolution=True,
        width=32,
        height=32,
        margin_scale=1.0,
        smooth_center=True,
        center_smoothing_strength=0.25,
        smooth_zoom=True,
        zoom_smoothing_strength=0.25,
        offset_x=0,
        offset_y=0,
        min_zoom=0.01,
        max_zoom=20.0,
        interpolation="bilinear",
        fit_frame_bounds=False,
        divisible_by=1,
        enable_visualize=False,
        crop_chunk_size=2,
        tracking_mode="mask",
        smoothing_method="none",
        center_smooth_window=1,
        size_smooth_window=1,
        size_metric="bbox_fit",
        resolution_mode="manual",
        auto_resolution_cap=768,
    )
    options.update(overrides)
    return MODULE.BatchImageCropByMaskAdvanced_StDismas().crop(
        images=images,
        crop_mask=crop_mask,
        **options,
    )


class CropUncropTests(unittest.TestCase):
    def setUp(self):
        self.images = torch.zeros((3, 64, 64, 3), dtype=torch.float32)
        self.masks = torch.zeros((3, 64, 64), dtype=torch.float32)
        self.masks[0, 20:30, 10:20] = 1.0
        self.masks[2, 20:30, 30:40] = 1.0

    def test_gap_is_interpolated_from_both_sides(self):
        result = run_crop(self.images, self.masks)
        metadata = result[4]
        centers = [frame["center"][0] for frame in metadata["frames"]]
        self.assertAlmostEqual(centers[0], 15.0)
        self.assertAlmostEqual(centers[1], 25.0)
        self.assertAlmostEqual(centers[2], 35.0)
        self.assertFalse(metadata["frames"][1]["valid"])

    def test_tracking_mode_is_the_first_setting(self):
        required = MODULE.BatchImageCropByMaskAdvanced_StDismas.INPUT_TYPES()["required"]
        widget_names = [name for name in required if name not in {"images", "crop_mask"}]
        self.assertEqual(widget_names[0], "tracking_mode")

    def test_legacy_strength_controls_every_smoothing_method(self):
        values = MODULE.np.asarray([10.0, 20.0, 5.0, 30.0, 15.0])
        for method in ("gaussian", "savgol", "moving_average", "ema"):
            with self.subTest(method=method):
                locked = MODULE._smooth_trajectory(values, 5, method, 0.0)
                following = MODULE._smooth_trajectory(values, 5, method, 1.0)
                self.assertTrue(MODULE.np.allclose(locked, values[0]))
                self.assertFalse(MODULE.np.allclose(following, locked))

        disabled = MODULE._smooth_trajectory(values, 5, "none", 0.0)
        self.assertTrue(MODULE.np.allclose(disabled, values))

    def test_zero_strength_freezes_center_and_zoom_in_crop_pipeline(self):
        masks = torch.zeros_like(self.masks)
        masks[0, 10:20, 10:20] = 1.0
        masks[1, 20:40, 20:40] = 1.0
        masks[2, 30:42, 40:56] = 1.0
        result = run_crop(
            self.images,
            masks,
            smoothing_method="gaussian",
            center_smooth_window=3,
            size_smooth_window=3,
            center_smoothing_strength=0.0,
            zoom_smoothing_strength=0.0,
        )
        frames = result[4]["frames"]
        self.assertTrue(all(frame["center"] == frames[0]["center"] for frame in frames))
        self.assertTrue(all(frame["S"] == frames[0]["S"] for frame in frames))

    def test_auto_capped_resolution_and_report_outputs(self):
        result = run_crop(
            self.images,
            self.masks,
            margin_scale=20.0,
            resolution_mode="auto_capped",
            auto_resolution_cap=128,
            divisible_by=32,
        )
        self.assertLessEqual(max(result[7], result[8]), 128)
        self.assertIn("magnification", result[6])
        self.assertIn("magnification", result[4])

    def test_uncrop_skip_uses_tracking_validity(self):
        crops = run_crop(self.images, self.masks)
        white_crops = torch.ones_like(crops[0])
        output = MODULE.BatchImageUncropByMaskAdvanced_StDismas().uncrop(
            cropped_images=white_crops,
            crop_metadata=crops[4],
            mode="overlay_full",
            blend=1.0,
            base_images=self.images,
            undetected_frames="skip",
            uncrop_chunk_size=2,
            uncrop_memory_limit_mb=64,
        )[0]
        self.assertGreater(float(output[0].sum()), 0.0)
        self.assertEqual(float(output[1].sum()), 0.0)
        self.assertGreater(float(output[2].sum()), 0.0)

    def test_color_match_modes_can_be_disabled_or_selected(self):
        patch = torch.full((1, 4, 4, 3), 0.8)
        base = torch.full((1, 4, 4, 3), 0.2)
        alpha = torch.ones((1, 4, 4, 1))
        disabled = MODULE._color_match_patch(patch, base, alpha, "off", 1.0)
        matched = MODULE._color_match_patch(patch, base, alpha, "mean", 1.0)
        luminance = MODULE._color_match_patch(patch, base, alpha, "luminance", 1.0)
        self.assertTrue(torch.allclose(disabled, patch))
        self.assertAlmostEqual(float(matched.mean()), 0.2, places=5)
        self.assertAlmostEqual(float(luminance.mean()), 0.2, places=5)

    def test_optional_face_mode_does_not_require_a_crop_mask(self):
        prediction = ([[10.0, 12.0, 22.0, 28.0]], [0.9], [0])
        with (
            mock.patch.object(MODULE, "_load_face_detector", return_value=object()),
            mock.patch.object(MODULE, "_predict_boxes", return_value=prediction),
            mock.patch.object(MODULE, "_release_optional_face_models"),
        ):
            result = run_crop(
                self.images,
                None,
                tracking_mode="face_detection",
                identity_track=False,
                size_metric="height",
            )
        self.assertEqual(result[4]["tracking_mode"], "face_detection")
        self.assertGreater(float(result[1].sum()), 0.0)
        self.assertIn("face tracking", result[6])

    def test_identity_reference_selects_the_matching_face(self):
        two_faces = (
            [[8.0, 10.0, 20.0, 26.0], [40.0, 10.0, 52.0, 26.0]],
            [0.95, 0.90],
            [0, 0],
        )

        def fake_embeddings(_app, bgr):
            if float(bgr.mean()) > 200.0:
                return [([0.0, 0.0, 10.0, 10.0], MODULE.np.asarray([1.0, 0.0], dtype=MODULE.np.float32))]
            return [
                (two_faces[0][0], MODULE.np.asarray([0.0, 1.0], dtype=MODULE.np.float32)),
                (two_faces[0][1], MODULE.np.asarray([1.0, 0.0], dtype=MODULE.np.float32)),
            ]

        reference = torch.ones((1, 32, 32, 3), dtype=torch.float32)
        with (
            mock.patch.object(MODULE, "_load_face_detector", return_value=object()),
            mock.patch.object(MODULE, "_predict_boxes", return_value=two_faces),
            mock.patch.object(MODULE, "_face_recogniser", return_value=object()),
            mock.patch.object(MODULE, "_embed_faces", side_effect=fake_embeddings),
            mock.patch.object(MODULE, "_release_optional_face_models"),
        ):
            result = run_crop(
                self.images,
                None,
                tracking_mode="face_detection",
                identity_reference=reference,
                identity_track=True,
                identity_threshold=0.28,
            )
        self.assertAlmostEqual(result[4]["frames"][0]["center"][0], 46.0)
        self.assertIn("identity=1", result[6])

    def test_fp16_inputs_keep_fp32_geometry_but_return_original_dtype(self):
        result = run_crop(self.images.half(), self.masks.half())
        self.assertEqual(result[0].dtype, torch.float16)
        output = MODULE.BatchImageUncropByMaskAdvanced_StDismas().uncrop(
            cropped_images=result[0],
            crop_metadata=result[4],
            mode="overlay_full",
            blend=1.0,
            base_images=self.images.half(),
            uncrop_chunk_size=1,
            uncrop_memory_limit_mb=64,
        )[0]
        self.assertEqual(output.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
