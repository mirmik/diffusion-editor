from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CanonicalPointmapHeadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from diffusion_editor.training.canonical_pointmap_head import (
            CanonicalPointmapHead,
            MultiscaleCanonicalPointmapHead,
            build_canonical_pointmap_head,
        )

        cls.local_head = CanonicalPointmapHead
        cls.multiscale_head = MultiscaleCanonicalPointmapHead
        cls.build_head = staticmethod(build_canonical_pointmap_head)

    @staticmethod
    def inputs(grid: int = 16):
        generator = torch.Generator().manual_seed(17)
        features = torch.randn(2, 4, 32, grid, grid, generator=generator)
        camera = torch.randn(2, 16, generator=generator)
        rays = torch.randn(2, 6, grid, grid, generator=generator)
        timestep = torch.tensor([0.25, 0.75])
        return features, camera, rays, timestep

    def test_factory_preserves_local_v1_checkpoint_shape(self) -> None:
        original = self.local_head(4, 32, 8, 32, 6, 8)
        restored = self.build_head("local-v1", 4, 32, 8, 32, 6, 8)
        restored.load_state_dict(original.state_dict())

        features, camera, rays, timestep = self.inputs(grid=8)
        with torch.no_grad():
            expected = original(features, camera, rays, 16, timestep=timestep)
            actual = restored(features, camera, rays, 16, timestep=timestep)

        self.assertEqual(actual.shape, (2, 5, 16, 16))
        torch.testing.assert_close(actual, expected)

    def test_multiscale_v2_returns_native_pyramid(self) -> None:
        model = self.multiscale_head(4, 32, 8, 32, 6, 8)
        features, camera, rays, timestep = self.inputs()

        result = model.forward_pyramid(
            features,
            camera,
            rays,
            output_size=64,
            timestep=timestep,
        )

        self.assertEqual(result["prediction"].shape, (2, 5, 64, 64))
        self.assertEqual(
            [item.shape for item in result["auxiliary"]],
            [(2, 5, 16, 16), (2, 5, 32, 32)],
        )

    def test_multiscale_v2_conditions_and_backpropagates(self) -> None:
        model = self.multiscale_head(4, 32, 8, 32, 6, 8)
        features, camera, rays, timestep = self.inputs(grid=8)
        first = model(features, camera, rays, 32, timestep=timestep)
        second = model(
            features,
            camera + 0.25,
            rays,
            32,
            timestep=1.0 - timestep,
        )

        self.assertFalse(torch.allclose(first, second))
        first.square().mean().backward()
        self.assertIsNotNone(model.projectors[0][0].weight.grad)
        self.assertIsNotNone(model.attention.attention.in_proj_weight.grad)
        self.assertIsNotNone(model.output.xyz[0].weight.grad)
        self.assertIsNotNone(model.output.mask[0].weight.grad)
        self.assertIsNotNone(model.output.uncertainty[0].weight.grad)

    def test_multiscale_v2_requires_timestep_when_configured(self) -> None:
        model = self.multiscale_head(4, 32, 8, 32, 6, 8)
        features, camera, rays, _timestep = self.inputs(grid=8)

        with self.assertRaisesRegex(ValueError, "requires timestep"):
            model(features, camera, rays, 32)

    def test_multiscale_vl_v3_queries_front_back_context(self) -> None:
        model = self.build_head(
            "multiscale-vl-v3",
            4,
            32,
            8,
            32,
            6,
            8,
            context_layers=2,
            context_channels=48,
        )
        features, camera, rays, timestep = self.inputs(grid=8)
        generator = torch.Generator().manual_seed(19)
        context = torch.randn(
            2, 2, 2, 4, 4, 48, generator=generator
        )
        with torch.no_grad():
            model.vl_context.gate.fill_(1.0)
        first = model(
            features,
            camera,
            rays,
            32,
            timestep=timestep,
            context=context,
        )
        second = model(
            features,
            camera,
            rays,
            32,
            timestep=timestep,
            context=context.roll(1, dims=3),
        )

        self.assertEqual(first.shape, (2, 5, 32, 32))
        self.assertFalse(torch.allclose(first, second))
        first.square().mean().backward()
        self.assertIsNotNone(
            model.vl_context.context_projectors[0][1].weight.grad
        )
        self.assertIsNotNone(model.vl_context.attention.in_proj_weight.grad)

    def test_multiscale_vl_v3_upgrade_preserves_v2_prediction(self) -> None:
        baseline = self.build_head("multiscale-v2", 4, 32, 8, 32, 6, 8)
        fusion = self.build_head(
            "multiscale-vl-v3",
            4,
            32,
            8,
            32,
            6,
            8,
            context_layers=2,
            context_channels=48,
        )
        incompatible = fusion.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(
            all(key.startswith("vl_context.") for key in incompatible.missing_keys)
        )
        features, camera, rays, timestep = self.inputs(grid=8)
        context = torch.randn(2, 2, 2, 4, 4, 48)

        with torch.no_grad():
            expected = baseline(
                features, camera, rays, 32, timestep=timestep
            )
            actual = fusion(
                features,
                camera,
                rays,
                32,
                timestep=timestep,
                context=context,
            )

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
