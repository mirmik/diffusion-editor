"""Termin SDK provenance checks used by install, run, and CI.

The Termin wheels contain native bindings but borrow shared libraries from one
SDK installation.  A wheel from one SDK build can import against another SDK's
libraries far enough to produce obscure undefined-symbol errors.  This module
turns the SDK runtime manifest into an explicit contract and checks it before
native modules are imported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.parser import Parser
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_FILE = PROJECT_ROOT / ".termin-sdk"
RUNTIME_MANIFEST = "python-runtime-manifest.json"
DIRECT_TERMIN_DISTRIBUTIONS = (
    "tcbase",
    "tcgui",
    "tgfx",
    "termin-display",
)


class SdkContractError(RuntimeError):
    """The selected SDK, wheelhouse, or Python environment is inconsistent."""


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class WheelMetadata:
    name: str
    version: str
    requires: tuple[str, ...]
    native_members: tuple[str, ...]
    path: Path

    @property
    def normalized_name(self) -> str:
        return normalize_distribution(self.name)


@dataclass(frozen=True)
class SdkContract:
    root: Path
    python_abi: str
    site_packages: Path
    versions: dict[str, str]
    termin_distributions: frozenset[str]

    def version(self, name: str) -> str:
        normalized = normalize_distribution(name)
        try:
            return self.versions[normalized]
        except KeyError as exc:
            raise SdkContractError(
                f"{RUNTIME_MANIFEST} does not describe required distribution {name!r}"
            ) from exc


def _validate_sdk_layout(root: Path) -> Path:
    root = root.expanduser().resolve()
    missing = [
        name
        for name in ("lib", "wheels", RUNTIME_MANIFEST)
        if not (root / name).exists()
    ]
    if missing:
        rendered = ", ".join(missing)
        raise SdkContractError(f"Termin SDK {root} is incomplete; missing: {rendered}")
    if not (root / "lib").is_dir() or not (root / "wheels").is_dir():
        raise SdkContractError(f"Termin SDK {root} must contain lib/ and wheels/ directories")
    return root


def resolve_sdk(
    explicit: str | os.PathLike[str] | None = None,
    *,
    state_file: Path = DEFAULT_STATE_FILE,
) -> Path:
    """Resolve one SDK root without silently switching after installation."""

    candidate: Path | None = None
    source = ""
    if explicit:
        candidate = Path(explicit)
        source = "--sdk"
    elif os.environ.get("TERMIN_SDK"):
        candidate = Path(os.environ["TERMIN_SDK"])
        source = "TERMIN_SDK"
    elif state_file.is_file():
        raw = state_file.read_text(encoding="utf-8").strip()
        if not raw or "\n" in raw or "\r" in raw:
            raise SdkContractError(f"Invalid saved SDK path in {state_file}")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = state_file.parent / candidate
        source = str(state_file)
    elif (Path("/opt/termin") / "lib").is_dir():
        candidate = Path("/opt/termin")
        source = "/opt/termin"

    if candidate is None:
        raise SdkContractError(
            "Termin SDK not found. Set TERMIN_SDK or run ./install-deps.sh with "
            "TERMIN_SDK pointing at a complete SDK."
        )
    try:
        return _validate_sdk_layout(candidate)
    except SdkContractError as exc:
        raise SdkContractError(f"SDK selected by {source} is invalid: {exc}") from exc


def load_contract(
    root: Path,
    *,
    interpreter_abi: str | None = None,
) -> SdkContract:
    root = _validate_sdk_layout(root)
    manifest_path = root / RUNTIME_MANIFEST
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SdkContractError(f"Cannot read {manifest_path}: {exc}") from exc

    python_abi = str(payload.get("python_abi", ""))
    expected_abi = interpreter_abi or f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_abi != expected_abi:
        raise SdkContractError(
            f"SDK Python ABI is {python_abi or '<missing>'}, interpreter ABI is {expected_abi}"
        )

    site_packages_value = str(payload.get("site_packages", ""))
    site_packages = root / site_packages_value
    if not site_packages_value or not site_packages.is_dir():
        raise SdkContractError(
            f"{RUNTIME_MANIFEST} points to missing site-packages: "
            f"{site_packages_value or '<missing>'}"
        )

    versions: dict[str, str] = {}
    termin_distributions: set[str] = set()
    for item in payload.get("distributions", []):
        if not isinstance(item, dict) or not item.get("name") or not item.get("version"):
            continue
        name = normalize_distribution(str(item["name"]))
        versions[name] = str(item["version"])
        if item.get("kind") == "termin":
            termin_distributions.add(name)

    contract = SdkContract(
        root=root,
        python_abi=python_abi,
        site_packages=site_packages,
        versions=versions,
        termin_distributions=frozenset(termin_distributions),
    )
    for name in DIRECT_TERMIN_DISTRIBUTIONS:
        contract.version(name)

    native_build_ids = {
        local
        for name in ("tcbase", "tgfx", "termin-display")
        if (local := _native_sdk_build_id(contract.version(name))) is not None
    }
    if len(native_build_ids) != 1:
        raise SdkContractError(
            "SDK manifest mixes native binding build IDs: "
            + (", ".join(sorted(native_build_ids)) or "none found")
        )
    return contract


def _native_sdk_build_id(version: str) -> str | None:
    _base, separator, local = version.partition("+")
    if separator and local.startswith("sdk"):
        return local
    return None


def read_wheel_metadata(path: Path) -> WheelMetadata:
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
            if len(candidates) != 1:
                raise SdkContractError(
                    f"Wheel {path.name} has {len(candidates)} METADATA records"
                )
            parsed = Parser().parsestr(archive.read(candidates[0]).decode("utf-8"))
            native_members = tuple(
                name
                for name in names
                if name.lower().endswith((".so", ".pyd", ".dylib"))
            )
    except (OSError, BadZipFile, UnicodeDecodeError) as exc:
        raise SdkContractError(f"Cannot inspect wheel {path}: {exc}") from exc

    name = parsed.get("Name")
    version = parsed.get("Version")
    if not name or not version:
        raise SdkContractError(f"Wheel {path.name} has incomplete package metadata")
    return WheelMetadata(
        name=name,
        version=version,
        requires=tuple(parsed.get_all("Requires-Dist") or ()),
        native_members=native_members,
        path=path,
    )


def wheelhouse_metadata(contract: SdkContract) -> dict[tuple[str, str], WheelMetadata]:
    result: dict[tuple[str, str], WheelMetadata] = {}
    for path in sorted((contract.root / "wheels").glob("*.whl")):
        wheel = read_wheel_metadata(path)
        result[(wheel.normalized_name, wheel.version)] = wheel
    return result


def _requirement_name(requirement: str) -> str:
    # Termin's wheel metadata uses ordinary PEP 508 requirements.  Only the
    # distribution name is needed to follow the SDK-owned dependency closure.
    name = re.split(r"[\s\[\](;<>!=~]", requirement, maxsplit=1)[0]
    return normalize_distribution(name)


def _base_version(version: str) -> str:
    return version.partition("+")[0]


def _native_payload_matches_sdk(contract: SdkContract, wheel: WheelMetadata) -> bool:
    if not wheel.native_members:
        return False
    try:
        with ZipFile(wheel.path) as archive:
            for member in wheel.native_members:
                installed = contract.site_packages / member
                if not installed.is_file():
                    return False
                wheel_hash = hashlib.sha256(archive.read(member)).digest()
                installed_hash = hashlib.sha256(installed.read_bytes()).digest()
                if wheel_hash != installed_hash:
                    return False
    except (OSError, BadZipFile, KeyError):
        return False
    return True


def _select_sdk_wheel(
    contract: SdkContract,
    wheels: dict[tuple[str, str], WheelMetadata],
    name: str,
) -> WheelMetadata:
    manifest_version = contract.version(name)
    candidates = [
        wheel for (candidate_name, _version), wheel in wheels.items() if candidate_name == name
    ]
    exact = [wheel for wheel in candidates if wheel.version == manifest_version]
    if len(exact) == 1:
        return exact[0]

    # The current Termin build tags wheels from the latest artifact mtime. The
    # SDK runtime installation can retag an otherwise byte-identical binding
    # before the final wheelhouse stage. Accept that case only after comparing
    # every native payload byte against the selected SDK installation.
    compatible = [
        wheel
        for wheel in candidates
        if _base_version(wheel.version) == _base_version(manifest_version)
        and _native_sdk_build_id(wheel.version) is not None
        and _native_payload_matches_sdk(contract, wheel)
    ]
    if len(compatible) == 1:
        return compatible[0]

    available = ", ".join(sorted(wheel.version for wheel in candidates)) or "<none>"
    raise SdkContractError(
        f"SDK wheelhouse has no payload-compatible {name} for manifest version "
        f"{manifest_version}; available: {available}"
    )


def termin_requirement_versions(contract: SdkContract) -> dict[str, str]:
    """Return exact SDK-owned requirements needed by this application."""

    wheels = wheelhouse_metadata(contract)
    pending = [normalize_distribution(name) for name in DIRECT_TERMIN_DISTRIBUTIONS]
    selected: dict[str, WheelMetadata] = {}

    while pending:
        name = pending.pop()
        if name in selected:
            continue
        wheel = _select_sdk_wheel(contract, wheels, name)
        selected[name] = wheel
        for requirement in wheel.requires:
            dependency = _requirement_name(requirement)
            if dependency in contract.termin_distributions and dependency not in selected:
                pending.append(dependency)

    native_build_ids = {
        build_id
        for wheel in selected.values()
        if (build_id := _native_sdk_build_id(wheel.version)) is not None
    }
    if len(native_build_ids) != 1:
        raise SdkContractError(
            "SDK wheel dependency closure mixes native build IDs: "
            + (", ".join(sorted(native_build_ids)) or "none found")
        )
    return {name: selected[name].version for name in sorted(selected)}


def termin_requirement_closure(contract: SdkContract) -> tuple[str, ...]:
    versions = termin_requirement_versions(contract)
    return tuple(f"{name}=={version}" for name, version in versions.items())


def installed_version_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            result[normalize_distribution(name)] = distribution.version
    return result


def verify_installed(
    contract: SdkContract,
    installed: dict[str, str] | None = None,
) -> None:
    if installed is None:
        installed = installed_version_map()
    errors: list[str] = []

    for name, expected in termin_requirement_versions(contract).items():
        actual = installed.get(name)
        if actual != expected:
            errors.append(f"{name}: installed {actual or '<missing>'}, SDK requires {expected}")

    if errors:
        raise SdkContractError("Termin Python environment does not match the selected SDK:\n- " + "\n- ".join(errors))


def verify_imports() -> None:
    import tcbase  # noqa: F401
    import tcgui
    from termin.display import SDLBackendWindow
    from tgfx import Tgfx2Context, configure_default_shader_runtime

    required = (tcgui, SDLBackendWindow, Tgfx2Context, configure_default_shader_runtime)
    if any(value is None for value in required):
        raise SdkContractError("One or more required Termin runtime exports are unavailable")


def write_state(root: Path, state_file: Path = DEFAULT_STATE_FILE) -> None:
    root = _validate_sdk_layout(root)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(f"{state_file.name}.tmp")
    temporary.write_text(f"{root}\n", encoding="utf-8")
    temporary.replace(state_file)


def _sdk_from_args(args: argparse.Namespace) -> Path:
    return resolve_sdk(args.sdk, state_file=Path(args.state_file))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_sdk_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--sdk", help="explicit Termin SDK root")
        command.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))

    resolve = subparsers.add_parser("resolve", help="print the selected SDK root")
    add_sdk_options(resolve)

    requirements = subparsers.add_parser(
        "requirements", help="print the exact Termin wheel dependency closure"
    )
    add_sdk_options(requirements)

    verify = subparsers.add_parser(
        "verify-installed", help="verify installed Termin packages against the SDK"
    )
    add_sdk_options(verify)
    verify.add_argument("--imports", action="store_true", help="also import native runtime modules")

    save = subparsers.add_parser("write-state", help="persist the selected SDK for run.sh")
    add_sdk_options(save)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        root = _sdk_from_args(args)
        # Native package loaders read TERMIN_SDK during import. Keep the loader
        # on the same root that this command just validated, including when an
        # explicit --sdk overrides a previously saved project path.
        os.environ["TERMIN_SDK"] = str(root)
        if args.command == "resolve":
            print(root)
            return 0

        contract = load_contract(root)
        if args.command == "requirements":
            for requirement in termin_requirement_closure(contract):
                print(requirement)
            return 0
        if args.command == "verify-installed":
            # verify_installed also checks the wheelhouse closure, ensuring the
            # saved SDK can reproduce this environment.
            verify_installed(contract)
            if args.imports:
                verify_imports()
            print(f"Termin SDK runtime verified: {root} (Python {contract.python_abi})")
            return 0
        if args.command == "write-state":
            termin_requirement_closure(contract)
            write_state(root, Path(args.state_file))
            print(f"Saved Termin SDK: {root}")
            return 0
    except SdkContractError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
