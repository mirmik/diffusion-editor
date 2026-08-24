#!/usr/bin/env python3
"""Copy a canonical identity experiment out of disposable storage.

The copy is self-contained: every feature manifest is rebound to the copied
dataset and a realized experiment config records the permanent paths.  Source
files are never removed, and rsync makes interrupted copies resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--identity",
        action="append",
        dest="identities",
        help="Copy only selected identity ids; repeat as needed.",
    )
    return parser.parse_args()


def _resolve_repository_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _source_path(identity: dict, source_root: Path, kind: str) -> Path:
    configured = identity.get(kind)
    if configured:
        return _resolve_repository_path(configured)
    directory = "datasets" if kind == "dataset" else "features"
    return source_root / directory / identity["id"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"COPY {source} -> {destination}", flush=True)
    subprocess.run(
        ["rsync", "-a", "--info=stats1", f"{source}/", f"{destination}/"],
        check=True,
    )


def _rebind_feature_manifest(feature_root: Path, dataset_root: Path) -> None:
    feature_manifest_path = feature_root / "manifest.json"
    dataset_manifest_path = dataset_root / "manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    expected_hash = feature_manifest.get("dataset_manifest_sha256")
    actual_hash = _sha256(dataset_manifest_path)
    if expected_hash and expected_hash != actual_hash:
        raise RuntimeError(
            f"dataset manifest hash mismatch for {feature_root}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    feature_manifest["dataset"] = str(dataset_root.resolve())
    feature_manifest["dataset_manifest_sha256"] = actual_hash
    _atomic_json(feature_manifest_path, feature_manifest)


def main() -> int:
    args = _arguments()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise SystemExit("source and output roots must differ")

    identities = config["identities"]
    if args.identities:
        requested = set(args.identities)
        known = {identity["id"] for identity in identities}
        missing = sorted(requested - known)
        if missing:
            raise SystemExit(f"unknown identity ids: {missing}")
        identities = [identity for identity in identities if identity["id"] in requested]

    output_root.mkdir(parents=True, exist_ok=True)
    realized_path = output_root / "experiment.json"
    if realized_path.exists():
        realized = json.loads(realized_path.read_text(encoding="utf-8"))
        if (
            realized.get("schema") != config.get("schema")
            or realized.get("name") != config.get("name")
        ):
            raise RuntimeError(
                f"existing realized config describes another experiment: {realized_path}"
            )
    else:
        realized = json.loads(json.dumps(config))
    realized_by_id = {identity["id"]: identity for identity in realized["identities"]}
    for identity in identities:
        identity_id = identity["id"]
        dataset_target = output_root / "datasets" / identity_id
        feature_target = output_root / "features" / identity_id
        _copy_tree(_source_path(identity, source_root, "dataset"), dataset_target)
        _copy_tree(_source_path(identity, source_root, "features"), feature_target)
        _rebind_feature_manifest(feature_target, dataset_target)
        realized_by_id[identity_id]["dataset"] = str(dataset_target)
        realized_by_id[identity_id]["features"] = str(feature_target)

    realized["persisted_from"] = str(source_root)
    realized["persisted_config"] = str(config_path)
    _atomic_json(realized_path, realized)
    print(f"COMPLETE {realized_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
