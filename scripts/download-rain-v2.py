#!/usr/bin/env python3
"""Download and verify the Blender Studio Rain v2 training asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile


SOURCE_PAGE = "https://studio.blender.org/characters/rain/v2/"
ARCHIVE_URL = (
    "https://studio.blender.org/download-source/files/10/"
    "1033fad76f67bba3e362065d68c85791/"
    "1033fad76f67bba3e362065d68c85791.zip"
)
ARCHIVE_SHA256 = (
    "91d9ecf53b62c5ac533d86d711697fbe"
    "177b9ca6968ef46471661e2ffd8586eb"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Dedicated cache directory; the archive is not added to Git.",
    )
    parser.add_argument("--archive", type=Path, help="Use an existing ZIP.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(destination: Path) -> None:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={
            "User-Agent": "diffusion-editor-canonical-dataset/1",
            "Referer": SOURCE_PAGE,
        },
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    os.replace(temporary, destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination)
            except ValueError as error:
                raise RuntimeError(
                    f"archive member escapes output directory: {member.filename}"
                ) from error
        source.extractall(destination)


def main() -> int:
    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "rain-v2.zip"
    if args.archive:
        if args.archive.resolve() != archive.resolve():
            shutil.copyfile(args.archive, archive)
    elif not archive.is_file():
        print(f"Downloading {ARCHIVE_URL}", flush=True)
        _download(archive)
    actual = _sha256(archive)
    if actual != ARCHIVE_SHA256:
        raise SystemExit(
            f"Rain v2 archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {actual}"
        )
    asset_dir = args.output_dir / "rain-v2"
    asset_dir.mkdir(parents=True, exist_ok=True)
    blend = asset_dir / "rain_rig.blend"
    if not blend.is_file():
        _safe_extract(archive, asset_dir)
    provenance = {
        "asset": "Rain v2",
        "creator": "Blender Studio",
        "source_page": SOURCE_PAGE,
        "archive_url": ARCHIVE_URL,
        "archive_sha256": ARCHIVE_SHA256,
        "license": "CC-BY",
        "blend_file": str(blend.resolve()),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(blend.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
