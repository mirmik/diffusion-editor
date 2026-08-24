#!/usr/bin/env python3
"""Run resumable stages of a canonical-head identity scaling experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--training-recipe",
        type=Path,
        help="Overlay config.training with a versioned training recipe.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=("render", "validate", "features", "train", "evaluate"),
        required=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--identity",
        action="append",
        dest="identities",
        help="Limit stages to selected identity ids; repeat as needed.",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPOSITORY_ROOT / value).resolve()


def _run(command: list[str], log_path: Path | None = None) -> None:
    print("RUN " + " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nRUN " + " ".join(command) + "\n")
        log.flush()
        process_start = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while True:
            try:
                returncode = process.wait(timeout=15.0)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - process_start
                print(
                    f"WAIT pid={process.pid} elapsed={elapsed:.0f}s log={log_path}",
                    flush=True,
                )
    if returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(
            f"command failed with exit {returncode}; {log_path}\n{tail}"
        )
    print(f"LOG {log_path}", flush=True)


def _dataset_path(identity: dict, output_root: Path) -> Path:
    if "dataset" in identity:
        return _resolve(identity["dataset"])
    return output_root / "datasets" / identity["id"]


def _feature_path(identity: dict, output_root: Path) -> Path:
    if "features" in identity:
        return _resolve(identity["features"])
    return output_root / "features" / identity["id"]


def _blender_command(source: Path, kind: str) -> list[str]:
    command = [
        "blender",
        "--background",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
    ]
    if kind == "blend":
        command.append(str(source))
    else:
        command.append("--factory-startup")
    return command


def _camera_jitter_pairs(config: dict, identity_id: str) -> list[tuple[float, float]]:
    """Return deterministic (prompt, actual camera) azimuth pairs."""
    jitter = config["views"].get("camera_azimuth_jitter")
    azimuths = [float(value) for value in config["views"]["azimuths"]]
    if not jitter:
        return [(azimuth, azimuth) for azimuth in azimuths]
    maximum = float(jitter["maximum_degrees"])
    minimum = float(jitter.get("minimum_absolute_degrees", 0.0))
    replicas = int(jitter.get("replicas_per_azimuth", 1))
    seed = int(jitter.get("seed", 0))
    if not 0.0 <= minimum < maximum < 45.0:
        raise ValueError(
            "camera jitter requires 0 <= minimum_absolute_degrees < "
            "maximum_degrees < 45"
        )
    if replicas <= 0:
        raise ValueError("camera jitter replicas_per_azimuth must be positive")
    pairs = []
    for nominal in azimuths:
        for replica in range(replicas):
            digest = hashlib.sha256(
                f"{seed}:{identity_id}:{nominal:g}:{replica}".encode("utf-8")
            ).digest()
            unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
            magnitude = minimum + (maximum - minimum) * unit
            sign = -1.0 if digest[8] & 1 else 1.0
            pairs.append((nominal, (nominal + sign * magnitude) % 360.0))
    return pairs


def _render(config: dict, identity: dict, output_root: Path, force: bool) -> None:
    if "dataset" in identity:
        print(f"SKIP {identity['id']}: existing dataset", flush=True)
        return
    output = _dataset_path(identity, output_root)
    manifest_path = output / "manifest.json"
    azimuth_pairs = _camera_jitter_pairs(config, identity["id"])
    expected_views = len(azimuth_pairs) * len(config["views"]["elevations"])
    incomplete = False
    if manifest_path.exists() and not force:
        rendered_views = len(
            json.loads(manifest_path.read_text(encoding="utf-8")).get("views", [])
        )
        if rendered_views == expected_views:
            print(
                f"SKIP {identity['id']}: complete {rendered_views}-view dataset",
                flush=True,
            )
            return
        incomplete = True
        print(
            f"RESTART {identity['id']}: incomplete dataset "
            f"({rendered_views}/{expected_views} views)",
            flush=True,
        )
    source = _resolve(identity["source"])
    command = _blender_command(source, identity["kind"])
    command += [
        "--python", "scripts/render-rain-canonical-dataset.py", "--",
        "--output-dir", str(output),
        "--asset-name", identity["name"],
        "--identity-id", identity["id"],
        "--pose-id", identity.get("pose_id", "authored-rest"),
        "--resolution", str(config["views"]["resolution"]),
        "--samples", str(config["views"]["samples"]),
    ]
    if config["views"].get("distance") is not None:
        command += ["--distance", str(config["views"]["distance"])]
    if config["views"].get("distance_unit") is not None:
        command += ["--distance-unit", config["views"]["distance_unit"]]
    if identity["kind"] == "blend":
        command.append("--opened-blend-character")
        command += [
            "--render-engine",
            config["views"].get("blend_render_engine", "eevee"),
        ]
        subdivision_level = config["views"].get(
            "blend_subdivision_render_level"
        )
        if subdivision_level is not None:
            command += ["--subdivision-render-level", str(subdivision_level)]
        for mesh_name in identity.get("include_meshes", []):
            command += ["--include-mesh", mesh_name]
        for mesh_name in identity.get("only_meshes", []):
            command += ["--only-mesh", mesh_name]
        for option, key in (
            ("--creator", "creator"),
            ("--source-page", "source_page"),
            ("--license", "license"),
            ("--rig-name", "rig_name"),
            ("--pelvis-bone", "pelvis_bone"),
        ):
            if identity.get(key):
                command += [option, identity[key]]
    else:
        command += ["--asset", str(source)]
    if config["views"].get("camera_azimuth_jitter"):
        for prompt_azimuth, camera_azimuth in azimuth_pairs:
            command += [
                "--azimuth-pair",
                f"{prompt_azimuth:g}:{camera_azimuth:.9g}",
            ]
    else:
        for azimuth in config["views"]["azimuths"]:
            command += ["--azimuth", str(azimuth)]
    for elevation in config["views"]["elevations"]:
        command += ["--elevation", str(elevation)]
    if force or incomplete:
        command.append("--force")
    _run(command, output / "render.log")


def _validate(identity: dict, output_root: Path) -> None:
    dataset = _dataset_path(identity, output_root)
    _run(
        [
            str(REPOSITORY_ROOT / "venv/bin/python"),
            "scripts/validate-canonical-pointmap-dataset.py",
            str(dataset),
            "--minimum-alpha-iou", "0.97",
        ],
        dataset / "validation.log",
    )


def _features(config: dict, identity: dict, output_root: Path, force: bool) -> None:
    if "features" in identity:
        print(f"SKIP {identity['id']}: existing features", flush=True)
        return
    output = _feature_path(identity, output_root)
    if (output / "manifest.json").exists() and not force:
        print(f"SKIP {identity['id']}: {output / 'manifest.json'} exists", flush=True)
        return
    feature = config["feature_extraction"]
    command = [
        str(REPOSITORY_ROOT / ".venv-workers/bin/python"),
        "scripts/extract-qwen-canonical-features.py",
        str(_dataset_path(identity, output_root)),
        "--output-dir", str(output),
        "--blocks", ",".join(map(str, feature["blocks"])),
        "--denoising-steps", str(feature["denoising_steps"]),
        "--noise-replicas", str(feature["noise_replicas"]),
        "--output-resolution", str(feature["output_resolution"]),
        "--seed", str(feature["seed"]),
    ]
    conditioning_root = feature.get("conditioning_dataset_root")
    if conditioning_root:
        command += [
            "--conditioning-dataset",
            str(_resolve(conditioning_root) / identity["id"]),
        ]
    for schedule_step in feature.get("schedule_steps", []):
        command += ["--schedule-step", str(schedule_step)]
    for elevation in config["views"]["elevations"]:
        command += ["--production-elevation", str(elevation)]
    if force:
        command.append("--force")
    _run(command, output / "extract.log")


def _train(config: dict, identities: list[dict], output_root: Path, force: bool) -> None:
    training = config["training"]
    run_id = training.get("run_id", "scale10-scratch")
    output = output_root / "training" / run_id
    command = [
        str(REPOSITORY_ROOT / ".venv-workers/bin/python"),
        "scripts/train-canonical-pointmap-head.py",
        *[
            str(_feature_path(identity, output_root))
            for identity in identities
            if identity["split"] == "train"
        ],
        "--output-dir", str(output),
        "--steps", str(training["steps"]),
        "--batch-size", str(training["batch_size"]),
        "--target-resolution", str(training["target_resolution"]),
        "--projection-channels", str(training["projection_channels"]),
        "--hidden-channels", str(training["hidden_channels"]),
        "--architecture", training.get("architecture", "local-v1"),
        "--learning-rate", str(training["learning_rate"]),
        "--weight-decay", str(training["weight_decay"]),
        "--host-cache-samples", str(training["host_cache_samples"]),
        "--seed", str(training["seed"]),
    ]
    if training.get("timestep_conditioning"):
        command.append("--timestep-conditioning")
    if training.get("checkpoint"):
        command += ["--checkpoint", str(_resolve(training["checkpoint"]))]
    if training.get("checkpoint_every_epochs"):
        command += [
            "--checkpoint-every-epochs",
            str(training["checkpoint_every_epochs"]),
        ]
    if force:
        command.append("--force")
    _run(command, output / "training.log")


def _evaluate(config: dict, identities: list[dict], output_root: Path, force: bool) -> None:
    run_id = config["training"].get("run_id", "scale10-scratch")
    checkpoint = output_root / "training" / run_id / "head.pt"
    for identity in identities:
        if identity["split"] not in {"validation", "test"}:
            continue
        if run_id == "scale10-scratch":
            output = output_root / "evaluation" / identity["id"]
        else:
            output = output_root / "evaluation" / run_id / identity["id"]
        command = [
            str(REPOSITORY_ROOT / ".venv-workers/bin/python"),
            "scripts/train-canonical-pointmap-head.py",
            str(_feature_path(identity, output_root)),
            "--output-dir", str(output),
            "--steps", "0",
            "--checkpoint", str(checkpoint),
            "--host-cache-samples", "1",
        ]
        if force:
            command.append("--force")
        _run(command, output / "evaluation.log")


def main() -> int:
    args = _arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.training_recipe is not None:
        recipe = json.loads(args.training_recipe.read_text(encoding="utf-8"))
        if recipe.get("schema") != "diffusion-editor.canonical-training-recipe":
            raise SystemExit(f"unsupported training recipe: {args.training_recipe}")
        config["training"] = {**config["training"], **recipe["training"]}
    identities = config["identities"]
    if args.identities:
        requested = set(args.identities)
        known = {identity["id"] for identity in identities}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"unknown identity ids: {missing}")
        identities = [
            identity for identity in identities if identity["id"] in requested
        ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for stage in args.stage:
        if stage == "render":
            for identity in identities:
                _render(config, identity, args.output_root, args.force)
        elif stage == "validate":
            for identity in identities:
                _validate(identity, args.output_root)
        elif stage == "features":
            for identity in identities:
                _features(config, identity, args.output_root, args.force)
        elif stage == "train":
            _train(config, identities, args.output_root, args.force)
        elif stage == "evaluate":
            _evaluate(config, identities, args.output_root, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
