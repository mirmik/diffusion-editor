#!/usr/bin/env python3
"""Extract deterministic Qwen readout features for exact Rain views."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.generation.image_edit_profiles import (  # noqa: E402
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
)
from diffusion_editor.workers.ml_backend import RealMlBackend  # noqa: E402


AZIMUTH_DESCRIPTORS = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}
PRODUCTION_RING_AZIMUTHS = tuple(AZIMUTH_DESCRIPTORS)
PRODUCTION_ELEVATION_DESCRIPTORS = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--conditioning-dataset",
        type=Path,
        help=(
            "Optional exact-view dataset supplying front/back conditioning; "
            "the positional dataset still supplies all target views."
        ),
    )
    parser.add_argument("--blocks", default="15,30,45,59")
    parser.add_argument("--view", action="append", dest="views")
    parser.add_argument(
        "--production-ring",
        action="store_true",
        help=(
            "Select the eight eye-level azimuths supported by the production "
            "Multiple Angles prompt contract."
        ),
    )
    parser.add_argument(
        "--production-elevation",
        action="append",
        type=int,
        dest="production_elevations",
        choices=tuple(PRODUCTION_ELEVATION_DESCRIPTORS),
        help=(
            "Select production elevation rings; repeat as needed. Implies "
            "the eight supported azimuths."
        ),
    )
    parser.add_argument("--output-resolution", type=int, default=512)
    parser.add_argument("--readout-sigma", type=float, default=0.0)
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=0,
        help=(
            "Extract teacher-forced noisy exact-target features at every step "
            "of the production FlowMatch schedule."
        ),
    )
    parser.add_argument(
        "--noise-replicas",
        type=int,
        default=1,
        help="Independent noise trajectories per view in denoising mode.",
    )
    parser.add_argument(
        "--schedule-step",
        action="append",
        type=int,
        dest="schedule_steps",
        help=(
            "Keep only this zero-based step from --denoising-steps while "
            "preserving the production schedule; repeat as needed."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--prompt-azimuth-offset",
        type=float,
        default=0.0,
        help=(
            "Diagnostic control: rotate only the semantic view prompt while "
            "keeping the exact target render and camera unchanged."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output_resolution <= 0 or args.output_resolution % 16:
        parser.error("--output-resolution must be a positive multiple of 16")
    if not 0.0 <= args.readout_sigma <= 1.0:
        parser.error("--readout-sigma must be between zero and one")
    if args.denoising_steps < 0:
        parser.error("--denoising-steps must be non-negative")
    if args.noise_replicas <= 0:
        parser.error("--noise-replicas must be positive")
    if args.denoising_steps and args.readout_sigma:
        parser.error("--denoising-steps and --readout-sigma are mutually exclusive")
    if args.schedule_steps and not args.denoising_steps:
        parser.error("--schedule-step requires --denoising-steps")
    if args.schedule_steps and any(
        step < 0 or step >= args.denoising_steps for step in args.schedule_steps
    ):
        parser.error("--schedule-step is outside the denoising schedule")
    return args


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _opaque_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if source.mode != "RGBA":
            return source.convert("RGB")
        background = Image.new("RGBA", source.size, (0, 0, 0, 255))
        background.alpha_composite(source)
        return background.convert("RGB")


def _prompt(azimuth: float, elevation: float) -> str:
    rounded = int(round(azimuth)) % 360
    descriptor = AZIMUTH_DESCRIPTORS.get(
        rounded, f"azimuth {azimuth:g} degree view"
    )
    elevation_text = PRODUCTION_ELEVATION_DESCRIPTORS.get(
        int(round(elevation)),
        (
            f"elevated {elevation:g} degree shot"
            if elevation > 0.0
            else f"low-angle {abs(elevation):g} degree shot"
        ),
    )
    return f"<sks> {descriptor} {elevation_text} medium shot"


def _prompt_azimuth(azimuth: float, offset: float) -> float:
    """Return the diagnostic prompt angle without changing target geometry."""
    return (float(azimuth) + float(offset)) % 360.0


def _view_prompt_azimuth(view: dict, offset: float = 0.0) -> float:
    """Semantic requested angle, which may differ from the rendered camera."""
    nominal = view.get("prompt_azimuth_degrees", view["azimuth_degrees"])
    return _prompt_azimuth(float(nominal), offset)


def _signed_angle_delta(actual: float, nominal: float) -> float:
    return (float(actual) - float(nominal) + 180.0) % 360.0 - 180.0


def _anchor_view(views: list[dict], azimuth: float) -> dict:
    def distance(view: dict) -> tuple[float, float]:
        view_azimuth = float(view["azimuth_degrees"]) % 360.0
        angular = abs((view_azimuth - azimuth + 180.0) % 360.0 - 180.0)
        return angular, abs(float(view["elevation_degrees"]))

    return min(views, key=distance)


def _encode_packed_vae(pipe, image, *, width, height, generator, torch):
    tensor = pipe.image_processor.preprocess(image, height, width).unsqueeze(2)
    tensor = tensor.to(
        device=pipe._execution_device,
        dtype=pipe.transformer.dtype,
    )
    latents = pipe._encode_vae_image(image=tensor, generator=generator)
    latent_height, latent_width = latents.shape[3:]
    packed = pipe._pack_latents(
        latents,
        batch_size=1,
        num_channels_latents=latents.shape[1],
        height=latent_height,
        width=latent_width,
    )
    return packed, (1, latent_height // 2, latent_width // 2)


def main() -> int:
    args = _arguments()
    dataset_root = args.dataset.resolve()
    dataset_manifest_path = dataset_root / "manifest.json"
    dataset_manifest = _load_json(dataset_manifest_path)
    views = list(dataset_manifest["views"])
    if args.production_ring or args.production_elevations:
        if args.views:
            raise SystemExit(
                "production ring selection and --view are mutually exclusive"
            )
        selected_elevations = set(args.production_elevations or [0])
        views = [
            view
            for view in views
            if int(round(float(view["elevation_degrees"]))) in selected_elevations
            and _view_prompt_azimuth(view) in PRODUCTION_RING_AZIMUTHS
        ]
        selected_pairs = {
            (
                int(_view_prompt_azimuth(view)) % 360,
                int(round(float(view["elevation_degrees"]))),
            )
            for view in views
        }
        expected_pairs = {
            (azimuth, elevation)
            for elevation in selected_elevations
            for azimuth in PRODUCTION_RING_AZIMUTHS
        }
        missing_pairs = sorted(expected_pairs - selected_pairs)
        if missing_pairs:
            raise SystemExit(
                "dataset is missing production views (azimuth, elevation): "
                f"{missing_pairs}"
            )
    if args.views:
        selected = set(args.views)
        views = [view for view in views if view["id"] in selected]
        missing = sorted(selected - {view["id"] for view in views})
        if missing:
            raise SystemExit(f"unknown dataset views: {missing}")
    if not views:
        raise SystemExit("no dataset views selected")
    blocks = [int(value.strip()) for value in args.blocks.split(",")]
    if len(blocks) != len(set(blocks)):
        raise SystemExit("feature block list contains duplicates")
    output_manifest_path = args.output_dir / "manifest.json"
    if output_manifest_path.exists() and not args.force:
        raise SystemExit(
            f"output already contains {output_manifest_path}; pass --force"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        CONDITION_IMAGE_SIZE,
        VAE_IMAGE_SIZE,
        calculate_shift,
        calculate_dimensions,
        retrieve_timesteps,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen canonical feature extraction requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters = profile.defaults()
    adapters = [
        adapter.to_dict()
        for adapter in (
            *profile.default_lora_adapters,
            qwen_multiple_angles_lora_adapter(),
        )
    ]
    backend = RealMlBackend()
    started = time.monotonic()
    print("Loading frozen Qwen Multiple Angles...", flush=True)
    loaded = backend.load_image_edit(
        {
            "profile_id": profile.stable_id,
            "parameters": parameters,
            "lora_adapters": adapters,
        }
    )
    pipe = backend._instruct_pipe
    if pipe is None:
        raise RuntimeError("Qwen pipeline did not load")
    transformer = pipe.transformer
    invalid_blocks = [
        index for index in blocks
        if not 0 <= index < len(transformer.transformer_blocks)
    ]
    if invalid_blocks:
        raise SystemExit(f"invalid transformer blocks: {invalid_blocks}")

    conditioning_root = (
        args.conditioning_dataset.resolve()
        if args.conditioning_dataset is not None
        else dataset_root
    )
    conditioning_manifest = _load_json(conditioning_root / "manifest.json")
    front_view = _anchor_view(conditioning_manifest["views"], 0.0)
    back_view = _anchor_view(conditioning_manifest["views"], 180.0)
    front = _opaque_rgb(conditioning_root / front_view["rgb"])
    back = _opaque_rgb(conditioning_root / back_view["rgb"])
    condition_width, condition_height = calculate_dimensions(
        CONDITION_IMAGE_SIZE, front.width / front.height
    )
    condition_images = [
        pipe.image_processor.resize(front, condition_height, condition_width),
        pipe.image_processor.resize(back, condition_height, condition_width),
    ]
    prompt_cache = {}
    pipe.enable_sequential_cpu_offload()
    with torch.inference_mode():
        for view in views:
            prompt = _prompt(
                _view_prompt_azimuth(view, args.prompt_azimuth_offset),
                float(view["elevation_degrees"]),
            )
            print(f"Encoding conditioning for {view['id']}...", flush=True)
            embeds, mask = pipe.encode_prompt(
                image=condition_images,
                prompt=prompt,
                device=pipe._execution_device,
                num_images_per_prompt=1,
                max_sequence_length=int(parameters["max_sequence_length"]),
            )
            prompt_cache[view["id"]] = (
                prompt,
                embeds.detach().cpu(),
                mask.detach().cpu() if mask is not None else None,
            )
            pipe.maybe_free_model_hooks()

    # Qwen-VL is not needed after the conditioning tensors are materialized.
    pipe.maybe_free_model_hooks()
    pipe.remove_all_hooks()
    text_encoder = pipe.text_encoder
    pipe.text_encoder = None
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    pipe.enable_sequential_cpu_offload()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    vae_width, vae_height = calculate_dimensions(
        VAE_IMAGE_SIZE, front.width / front.height
    )
    with torch.inference_mode():
        front_latents, front_shape = _encode_packed_vae(
            pipe,
            front,
            width=vae_width,
            height=vae_height,
            generator=generator,
            torch=torch,
        )
        back_latents, back_shape = _encode_packed_vae(
            pipe,
            back,
            width=vae_width,
            height=vae_height,
            generator=generator,
            torch=torch,
        )
    front_latents = front_latents.detach().cpu()
    back_latents = back_latents.detach().cpu()
    pipe.maybe_free_model_hooks()

    if args.denoising_steps:
        grid_size = args.output_resolution // pipe.vae_scale_factor // 2
        image_seq_len = grid_size * grid_size
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        requested_sigmas = np.linspace(
            1.0,
            1.0 / args.denoising_steps,
            args.denoising_steps,
        )
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            args.denoising_steps,
            pipe._execution_device,
            sigmas=requested_sigmas,
            mu=mu,
        )
        schedule = [
            {
                "step": step,
                "timestep": float(timestep.detach().cpu()),
                "normalized_timestep": float(timestep.detach().cpu()) / 1000.0,
                "sigma": float(pipe.scheduler.sigmas[step]),
            }
            for step, timestep in enumerate(timesteps)
        ]
        if args.schedule_steps:
            selected_steps = set(args.schedule_steps)
            schedule = [item for item in schedule if item["step"] in selected_steps]
    else:
        schedule = [
            {
                "step": 0,
                "timestep": args.readout_sigma * 1000.0,
                "normalized_timestep": args.readout_sigma,
                "sigma": args.readout_sigma,
            }
        ]

    captures = {}
    target_tokens = None

    def make_hook(block_index: int):
        def hook(_module, _inputs, output):
            if target_tokens is None:
                return
            captures[block_index] = (
                output[0, :target_tokens].detach().to(
                    device="cpu", dtype=torch.float16
                )
            )

        return hook

    handles = [
        transformer.transformer_blocks[index].img_norm2.register_forward_hook(
            make_hook(index)
        )
        for index in blocks
    ]
    output_views = []
    try:
        for view_index, view in enumerate(views, start=1):
            identifier = view["id"]
            prompt, prompt_embeds, prompt_mask = prompt_cache[identifier]
            target = _opaque_rgb(dataset_root / view["rgb"])
            generator.manual_seed(args.seed)
            print(
                f"[{view_index}/{len(views)}] {identifier}: VAE + "
                f"{args.noise_replicas * len(schedule)} readouts",
                flush=True,
            )
            with torch.inference_mode():
                target_latents, target_shape = _encode_packed_vae(
                    pipe,
                    target,
                    width=args.output_resolution,
                    height=args.output_resolution,
                    generator=generator,
                    torch=torch,
                )
            target_tokens = int(target_latents.shape[1])
            feature_samples = []
            for replica in range(args.noise_replicas):
                if any(item["sigma"] for item in schedule):
                    generator.manual_seed(args.seed + replica)
                    noise = torch.randn(
                        target_latents.shape,
                        generator=generator,
                        device="cpu",
                        dtype=target_latents.dtype,
                    ).to(target_latents.device)
                else:
                    noise = None
                for schedule_item in schedule:
                    captures.clear()
                    sigma = schedule_item["sigma"]
                    noisy_target = (
                        target_latents
                        if not sigma
                        else (1.0 - sigma) * target_latents + sigma * noise
                    )
                    with torch.inference_mode():
                        hidden_states = torch.cat(
                            (
                                noisy_target,
                                front_latents.to(
                                    device=pipe._execution_device,
                                    dtype=target_latents.dtype,
                                ),
                                back_latents.to(
                                    device=pipe._execution_device,
                                    dtype=target_latents.dtype,
                                ),
                            ),
                            dim=1,
                        )
                        embeddings = prompt_embeds.to(
                            device=pipe._execution_device,
                            dtype=target_latents.dtype,
                        )
                        embedding_mask = (
                            prompt_mask.to(device=pipe._execution_device)
                            if prompt_mask is not None
                            else None
                        )
                        timestep = torch.full(
                            (1,),
                            schedule_item["normalized_timestep"],
                            device=pipe._execution_device,
                            dtype=target_latents.dtype,
                        )
                        with transformer.cache_context("cond"):
                            transformer(
                                hidden_states=hidden_states,
                                timestep=timestep,
                                guidance=None,
                                encoder_hidden_states_mask=embedding_mask,
                                encoder_hidden_states=embeddings,
                                img_shapes=[
                                    [(target_shape), (front_shape), (back_shape)]
                                ],
                                attention_kwargs={},
                                return_dict=False,
                            )
                    missing = sorted(set(blocks) - set(captures))
                    if missing:
                        raise RuntimeError(f"feature hooks missed blocks: {missing}")
                    grid_height, grid_width = target_shape[1:]
                    stacked = np.stack(
                        [
                            captures[index]
                            .reshape(grid_height, grid_width, -1)
                            .numpy()
                            for index in blocks
                        ]
                    )
                    if args.denoising_steps or args.noise_replicas > 1:
                        feature_name = (
                            f"{identifier}-r{replica:02d}-"
                            f"s{schedule_item['step']:02d}.npy"
                        )
                    else:
                        feature_name = f"{identifier}.npy"
                    feature_path = args.output_dir / feature_name
                    np.save(feature_path, stacked, allow_pickle=False)
                    feature_samples.append(
                        {
                            "replica": replica,
                            **schedule_item,
                            "features": feature_name,
                            "shape": list(stacked.shape),
                        }
                    )
                    print(
                        f"[{view_index}/{len(views)}] {identifier} "
                        f"r{replica} s{schedule_item['step']}: "
                        f"{stacked.shape} {stacked.nbytes / 2**20:.1f} MiB",
                        flush=True,
                    )
                    del hidden_states, embeddings, stacked
                    captures.clear()
            output_views.append(
                {
                    "id": identifier,
                    "azimuth_degrees": float(view["azimuth_degrees"]),
                    "elevation_degrees": float(view["elevation_degrees"]),
                    "prompt_azimuth_degrees": _view_prompt_azimuth(
                        view, args.prompt_azimuth_offset
                    ),
                    "azimuth_error_degrees": _signed_angle_delta(
                        float(view["azimuth_degrees"]),
                        _view_prompt_azimuth(view, args.prompt_azimuth_offset),
                    ),
                    "prompt": prompt,
                    "features": feature_samples[0]["features"],
                    "shape": feature_samples[0]["shape"],
                    "samples": feature_samples,
                }
            )
            del target_latents
            captures.clear()
            target_tokens = None
            pipe.maybe_free_model_hooks()
            torch.cuda.empty_cache()
    finally:
        target_tokens = None
        for handle in handles:
            handle.remove()
        backend.unload_image_edit()

    manifest = {
        "schema": "diffusion-editor.qwen-canonical-features",
        "schema_version": 2,
        "dataset": str(dataset_root),
        "conditioning_dataset": str(conditioning_root),
        "dataset_manifest_sha256": _file_sha256(dataset_manifest_path),
        "profile": profile.stable_id,
        "loaded": loaded,
        "lora_adapters": adapters,
        "blocks": blocks,
        "feature_point": "img_norm2 output after attention, before AdaLN2 modulation",
        "readout_sigma": args.readout_sigma,
        "denoising_steps": args.denoising_steps,
        "noise_replicas": args.noise_replicas,
        "schedule": schedule,
        "output_resolution": args.output_resolution,
        "feature_grid": [
            args.output_resolution // pipe.vae_scale_factor // 2,
            args.output_resolution // pipe.vae_scale_factor // 2,
        ],
        "feature_channels": int(transformer.inner_dim),
        "dtype": "float16",
        "front_view": front_view["id"],
        "back_view": back_view["id"],
        "conditioning_views": [front_view["id"], back_view["id"]],
        "production_ring": bool(args.production_ring or args.production_elevations),
        "production_elevations": sorted(
            set(args.production_elevations or ([0] if args.production_ring else []))
        ),
        "seed": args.seed,
        "prompt_azimuth_offset": args.prompt_azimuth_offset,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "views": output_views,
    }
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Complete: {output_manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
