import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


# Load the node in isolation so the unit suite also runs outside a ComfyUI server.
package = types.ModuleType("_stdismas_tile_preview_tests")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "nodes")]
sys.modules[package.__name__] = package
preview = importlib.import_module(f"{package.__name__}.mmh3_tile_preview")
render = importlib.import_module(f"{package.__name__}._mmh3_tile_preview_render")
video_preview = importlib.import_module(f"{package.__name__}._mmh3_tile_preview_video")


SPATIAL = {
    "tile_width": 1920, "tile_height": 1088,
    "spatial_w_overlap": 256, "spatial_h_overlap": 256,
    "fade_width": 64, "fade_height": 64, "min_tile_size": 256,
    "overlap_mode": "earlier", "overlap_blend": "linear",
}
TEMPORAL = {"chunk_length": 102, "temporal_overlap": 17, "anchor_strength": 1.0}


def latent(tokens, height, width):
    samples = types.SimpleNamespace(is_nested=True, tensors=[
        torch.empty(1, 24, tokens, height, width, device="meta"),
        torch.empty(1, 32, 2, 1, device="meta"),
    ])
    return {"samples": samples}


class MMH3TilePreviewTests(unittest.TestCase):
    def test_screen_settings_with_4k_target(self):
        layout = preview.build_layout(
            latent=latent(107, 44, 80), latent_upscale_param={"width": 3840, "height": 2176},
            temporal_split_param=TEMPORAL, spatial_split_param=SPATIAL,
        )
        self.assertEqual(layout.columns, [(0, 120, 0), (104, 120, 16), (208, 32, 16)])
        self.assertEqual(layout.rows, [(0, 68, 0), (52, 68, 16), (104, 32, 16)])
        self.assertEqual(layout.frames, 362)
        self.assertEqual(layout.chunks, [(0, 0, 30, 102), (25, 85, 55, 187), (50, 170, 80, 272),
                                         (75, 255, 105, 357), (100, 340, 107, 362)])
        self.assertIn("45 sampler calls", preview.format_layout(layout))

    def test_edge_tile_expands_and_uses_realized_overlap(self):
        layout = preview.build_layout(
            latent_upscale_param={"width": 1952, "height": 1088},
            spatial_split_param={**SPATIAL, "spatial_w_overlap": 128},
        )
        self.assertEqual(layout.columns, [(0, 120, 0), (106, 16, 14)])
        self.assertIn("x=1696, y=0, 256 x 1088; overlap 224/0", preview.format_layout(layout))
        self.assertIsNone(layout.frames)

    def test_latest_resolved_equal_grid_uses_solved_overlap(self):
        layout = preview.build_layout(
            latent_upscale_param={"width": 3840, "height": 2176},
            spatial_split_param={**SPATIAL, "tile_width": 2048, "tile_height": 1216,
                "tile_size_mode": "rows_cols", "grid_cols": 2, "grid_rows": 2},
            planned_frames=360,
        )
        self.assertEqual(layout.columns, [(0, 128, 0), (112, 128, 16)])
        self.assertEqual(layout.rows, [(0, 76, 0), (60, 76, 16)])
        self.assertEqual(layout.frames, 362)
        self.assertEqual(len(layout.chunks), 1)

    def test_no_upscale_and_no_spatial_use_original_latent_shape(self):
        layout = preview.build_layout(latent=latent(3, 41, 79), planned_frames=362)
        self.assertEqual(layout.columns, [(0, 79, 0)])
        self.assertEqual(layout.rows, [(0, 41, 0)])
        self.assertEqual(layout.chunks, [(0, 0, 3, 9)])
        self.assertIn("1264 x 656 px", preview.format_layout(layout))
        self.assertIn("model patch footprint 1280 x 672", preview.format_layout(layout))

    def test_tiles_and_chunks_use_latent_shape_without_upscale_params(self):
        layout = preview.build_layout(
            latent=latent(32, 44, 80), temporal_split_param=TEMPORAL, planned_frames=360,
            spatial_split_param={**SPATIAL, "tile_width": 768, "tile_height": 512,
                                 "spatial_w_overlap": 128, "spatial_h_overlap": 128},
        )
        self.assertEqual(layout.columns, [(0, 48, 0), (40, 40, 8)])
        self.assertEqual(layout.rows, [(0, 32, 0), (24, 20, 8)])
        self.assertEqual(layout.chunks, [(0, 0, 30, 102), (25, 85, 32, 107)])
        self.assertIn("1280 x 704 px", preview.format_layout(layout))

    def test_input_validation_matches_sampling_constraints(self):
        with self.assertRaisesRegex(ValueError, "target size"):
            preview.build_layout(spatial_split_param=SPATIAL)
        with self.assertRaisesRegex(ValueError, "at least the configured tile size"):
            preview.build_layout(latent_upscale_param={"width": 1280, "height": 1088}, spatial_split_param=SPATIAL)
        with self.assertRaisesRegex(ValueError, "MiniMax H3 AV latent"):
            preview.build_layout(latent={"samples": torch.empty(1, 4, 64, 64)})

    def test_stitch_blends_left_then_top_and_mask_keeps_lower_weight(self):
        spatial = {
            **SPATIAL, "tile_width": 128, "tile_height": 128, "min_tile_size": 0,
            "spatial_w_overlap": 64, "spatial_h_overlap": 64,
        }
        layout = preview.build_layout(latent=latent(2, 12, 12), spatial_split_param=spatial)
        mask, stitched = render.spatial_maps(layout, 16)
        self.assertAlmostEqual(mask[5, 6], 1 / 3)
        self.assertEqual(mask[0, 4], 0)
        self.assertEqual(mask[0, 10], 1)
        c1, c2, c3, c4 = np.array(render.TILE_COLORS[:4], dtype=np.float32)
        # At the first shared corner cell both axes have 1/3 weight for every new tile.
        top = c1 * (2 / 3) + c2 * (1 / 3)
        prior = top * (2 / 3) + c3 * (1 / 3)
        np.testing.assert_allclose(stitched[5, 5], prior * (4 / 9) + c4 * (5 / 9), atol=1e-5)
        np.testing.assert_array_equal(render.stitch_weights(3, "midpoint", "later"), [1, 0, 0])
        np.testing.assert_array_equal(render.stitch_weights(3, "overwrite", "earlier"), [0, 0, 0])
        np.testing.assert_allclose(render.stitch_weights(5, "smoothstep", "earlier"), [0, .15625, .5, .84375, 1])

    def test_node_socket_types_registration_and_preview_payload(self):
        cls = preview.NODE_CLASS_MAPPINGS["StDismasMMH3TilePreview"]
        inputs = cls.INPUT_TYPES()["optional"]
        self.assertEqual(inputs["latent_upscale_param"][0], "H3_UPSCALE_PARAM")
        self.assertEqual(inputs["temporal_split_param"][0], "H3_TEMPORAL_PARAM")
        self.assertEqual(inputs["spatial_split_param"][0], "H3_SPATIAL_PARAM")
        self.assertEqual(inputs["show_overlap_and_fade"][0], "BOOLEAN")
        self.assertIs(inputs["show_overlap_and_fade"][1]["default"], True)
        self.assertEqual(cls.RETURN_TYPES, ("IMAGE", "STRING", "IMAGE"))
        self.assertEqual(cls.RETURN_NAMES[2], "preview_video")
        ui = {"images": [{"filename": "test.png", "type": "temp", "subfolder": ""}]}

        class PreviewImage:
            def save_images(self, image, *_args):
                return {"ui": ui, "result": (image,)}

        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(PreviewImage=PreviewImage)}):
            result = cls().preview(latent_upscale_param={"width": 3840, "height": 2176}, planned_frames=30,
                                   spatial_split_param=SPATIAL, background_image=torch.zeros(1, 48, 64, 3))
        image, report, video = result["result"]
        self.assertIsNone(video.message)
        self.assertEqual(result["ui"], ui)
        self.assertEqual((image.ndim, image.shape[0], image.shape[-1]), (4, 1, 3))
        self.assertEqual((image.device.type, image.dtype), ("cpu", torch.float32))
        self.assertTrue(0.0 <= image.min() <= image.max() <= 1.0)
        self.assertIn("39 frames", report)

    def test_overlap_switch_keeps_report_and_skips_spatial_maps(self):
        class PreviewImage:
            def save_images(self, image, *_args):
                return {"ui": {}, "result": (image,)}

        frame_color = [17, 33, 55]
        background = torch.tensor(frame_color).float().view(1, 1, 1, 3).expand(1, 48, 64, 3) / 255
        inputs = dict(latent_upscale_param={"width": 3840, "height": 2176}, planned_frames=200,
                      temporal_split_param=TEMPORAL, spatial_split_param=SPATIAL, background_image=background)
        node = preview.MMH3TilePreview()
        with patch.dict(sys.modules, {"nodes": types.SimpleNamespace(PreviewImage=PreviewImage)}):
            detailed, report, _ = node.preview(**inputs)["result"]
            with patch.object(render, "spatial_maps", side_effect=AssertionError("Unexpected seam map")):
                compact, compact_report, _ = node.preview(**inputs, show_overlap_and_fade=False)["result"]

        self.assertEqual(compact_report, report)
        self.assertLess(compact.shape[2], detailed.shape[2])
        compact_rgb = np.rint(compact.numpy() * 255).astype(np.uint8)
        detailed_rgb = np.rint(detailed.numpy() * 255).astype(np.uint8)
        amber = list(bytes.fromhex(render.AMBER[1:]))
        self.assertTrue((detailed_rgb == amber).all(axis=-1).any())
        self.assertFalse((compact_rgb == amber).all(axis=-1).any())
        self.assertTrue((compact_rgb == frame_color).all(axis=-1).any())

    def test_video_mode_requires_frames_and_returns_image_batch(self):
        node = preview.MMH3TilePreview()
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            node.preview(preview_mode="video", background_image=torch.zeros((1, 48, 64, 3)))
        frames = torch.zeros((2, 48, 64, 3))
        video_frames = torch.zeros((2, 480, 640, 3))
        ui = {"images": [{"filename": "test.mp4", "type": "temp", "subfolder": ""}], "animated": (True,)}
        with patch.object(preview, "save_video_preview", return_value=(video_frames, ui)) as save:
            result = node.preview(preview_mode="video", preview_fps=30, show_overlap_and_fade=False,
                                  background_image=frames, latent_upscale_param={"width": 640, "height": 480})
        self.assertIs(result["result"][2], video_frames)
        self.assertEqual(result["ui"], ui)
        actual_frames, fps, layout, pixels_per_cell, show_overlap = save.call_args.args
        self.assertIs(actual_frames, frames)
        self.assertEqual((fps, layout.width, layout.height, pixels_per_cell, show_overlap), (30, 40, 30, 16, False))

    def test_video_encoder_preserves_frame_sequence_and_draws_grid(self):
        import av

        sp = {**SPATIAL, "tile_width": 128, "tile_height": 128, "min_tile_size": 0,
              "spatial_w_overlap": 64, "spatial_h_overlap": 64}
        layout = preview.build_layout(latent=latent(2, 12, 12), spatial_split_param=sp)
        seams = render.video_overlay(layout, (192, 192), 16, True)
        edges = render.video_overlay(layout, (192, 192), 16, False)
        self.assertGreater(seams.getpixel((70, 55))[3], 0)
        self.assertEqual(edges.getpixel((70, 55))[3], 0)
        frames = torch.zeros((2, 48, 48, 3))
        frames[0, ..., 0] = 0.7
        frames[1, ..., 2] = 0.7
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.mp4"
            images = video_preview.encode_video_preview(path, frames, 30, layout, 16, True)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                self.assertEqual((stream.width, stream.height, stream.average_rate), (192, 192, 30))
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        self.assertEqual(len(decoded), 2)
        self.assertEqual((images.shape, images.dtype, images.device.type), ((2, 192, 192, 3), torch.float32, "cpu"))
        self.assertTrue(0 <= images.min() <= images.max() <= 1)
        torch.testing.assert_close(images[0, 50, 40], torch.tensor([178 / 255, 0, 0]))
        torch.testing.assert_close(images[1, 50, 40], torch.tensor([0, 0, 178 / 255]))
        self.assertTrue((images[0, 50, 64] > 0.9).all())
        self.assertGreater(decoded[0][50, 40, 0], 140)
        self.assertGreater(decoded[1][50, 40, 2], 140)
        self.assertTrue((decoded[0][50, 64] > 170).all())  # Start of column 2.


if __name__ == "__main__":
    unittest.main()
