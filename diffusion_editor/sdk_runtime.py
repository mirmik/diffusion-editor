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
import subprocess
import sys
import sysconfig
from typing import Iterable, Mapping
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_FILE = PROJECT_ROOT / ".termin-sdk"
RUNTIME_MANIFEST = "python-runtime-manifest.json"
DIRECT_TERMIN_DISTRIBUTIONS = (
    "tcbase",
    "tcgui",
    "termin-dispatch",
    "termin-gui-native",
    "tgfx",
    "termin-display",
)


class SdkContractError(RuntimeError):
    """The selected SDK, wheelhouse, or Python environment is inconsistent."""


@dataclass(frozen=True)
class PythonAbiIdentity:
    version: str
    soabi: str
    free_threaded: bool
    py_gil_disabled: bool

    @classmethod
    def from_mapping(cls, value: object, *, source: str) -> "PythonAbiIdentity":
        if not isinstance(value, Mapping):
            raise SdkContractError(
                f"{source} must be a structured Python ABI object; "
                "legacy string manifests are unsupported"
            )
        version = value.get("version")
        soabi = value.get("soabi")
        free_threaded = value.get("free_threaded")
        py_gil_disabled = value.get("py_gil_disabled")
        if (
            not isinstance(version, str)
            or not version
            or not isinstance(soabi, str)
            or not soabi
            or not isinstance(free_threaded, bool)
            or not isinstance(py_gil_disabled, bool)
        ):
            raise SdkContractError(
                f"{source} requires version, soabi, free_threaded, and "
                "py_gil_disabled with their canonical types"
            )

        match = re.match(r"^(?:cpython-|cp)(\d+)(t?)(?:-|$)", soabi)
        if match is None:
            raise SdkContractError(f"{source} has unsupported SOABI {soabi!r}")
        digits, suffix = match.groups()
        expected_digits = version.replace(".", "")
        if digits != expected_digits:
            raise SdkContractError(
                f"{source} version {version!r} disagrees with SOABI {soabi!r}"
            )
        if free_threaded != py_gil_disabled:
            raise SdkContractError(
                f"{source} free_threaded and py_gil_disabled markers disagree"
            )
        if bool(suffix) != free_threaded:
            raise SdkContractError(
                f"{source} SOABI {soabi!r} disagrees with free-threading markers"
            )
        return cls(version, soabi, free_threaded, py_gil_disabled)

    @classmethod
    def current(cls) -> "PythonAbiIdentity":
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        soabi = str(sysconfig.get_config_var("SOABI") or "")
        py_gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
        return cls.from_mapping(
            {
                "version": version,
                "soabi": soabi,
                "free_threaded": py_gil_disabled,
                "py_gil_disabled": py_gil_disabled,
            },
            source="current interpreter",
        )

    @property
    def wheel_abi_tag(self) -> str:
        return f"cp{self.version.replace('.', '')}{'t' if self.free_threaded else ''}"

    def __str__(self) -> str:
        return (
            f"{self.version} ({self.soabi}, "
            f"{'free-threaded' if self.free_threaded else 'GIL build'})"
        )


def normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class WheelMetadata:
    name: str
    version: str
    requires: tuple[str, ...]
    native_members: tuple[str, ...]
    abi_tags: tuple[str, ...]
    path: Path

    @property
    def normalized_name(self) -> str:
        return normalize_distribution(self.name)


@dataclass(frozen=True)
class SdkContract:
    root: Path
    python_abi: PythonAbiIdentity
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


def _read_contract(
    root: Path,
    *,
    interpreter_abi: PythonAbiIdentity | None,
) -> SdkContract:
    root = _validate_sdk_layout(root)
    manifest_path = root / RUNTIME_MANIFEST
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SdkContractError(f"Cannot read {manifest_path}: {exc}") from exc

    if payload.get("schema") != 3:
        raise SdkContractError(
            f"{RUNTIME_MANIFEST} schema must be 3; legacy manifests are unsupported"
        )
    python_abi = PythonAbiIdentity.from_mapping(
        payload.get("python_abi"),
        source=f"{RUNTIME_MANIFEST} python_abi",
    )
    if (
        python_abi.version != "3.14"
        or not python_abi.free_threaded
        or not python_abi.py_gil_disabled
        or python_abi.wheel_abi_tag != "cp314t"
    ):
        raise SdkContractError(
            f"Termin SDK must target CPython 3.14t free-threading; found {python_abi}"
        )
    if interpreter_abi is not None and python_abi != interpreter_abi:
        raise SdkContractError(
            f"SDK Python ABI is {python_abi}, interpreter ABI is {interpreter_abi}"
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


def load_contract(
    root: Path,
    *,
    interpreter_abi: PythonAbiIdentity | None = None,
) -> SdkContract:
    using_current_interpreter = interpreter_abi is None
    identity = PythonAbiIdentity.current() if using_current_interpreter else interpreter_abi
    contract = _read_contract(root, interpreter_abi=identity)
    if using_current_interpreter and _current_gil_enabled():
        raise SdkContractError(
            "Current CPython 3.14t runtime has the GIL enabled; "
            "unset PYTHON_GIL and restart with the SDK free-threaded interpreter"
        )
    return contract


def _current_gil_enabled() -> bool:
    probe = getattr(sys, "_is_gil_enabled", None)
    return bool(probe()) if probe is not None else True


def _probe_python_executable(executable: Path) -> tuple[PythonAbiIdentity, bool]:
    if not executable.is_file():
        raise SdkContractError(f"Python interpreter does not exist: {executable}")
    script = (
        "import json,sys,sysconfig;"
        "print(json.dumps({'version':f'{sys.version_info.major}.{sys.version_info.minor}',"
        "'soabi':sysconfig.get_config_var('SOABI') or '',"
        "'free_threaded':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),"
        "'py_gil_disabled':bool(sysconfig.get_config_var('Py_GIL_DISABLED')),"
        "'gil_enabled':bool(getattr(sys,'_is_gil_enabled',lambda:True)())}))"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SdkContractError(
            f"Cannot probe Python interpreter {executable}: {exc}"
        ) from exc
    identity = PythonAbiIdentity.from_mapping(payload, source=str(executable))
    return identity, payload.get("gil_enabled") is not False


def verify_python_executable(root: Path, executable: Path) -> PythonAbiIdentity:
    """Check an arbitrary interpreter against the SDK without importing the project in it."""

    contract = _read_contract(root, interpreter_abi=None)
    identity, gil_enabled = _probe_python_executable(executable)
    if identity != contract.python_abi:
        raise SdkContractError(
            f"Python interpreter ABI is {identity}, SDK ABI is {contract.python_abi}"
        )
    if gil_enabled:
        raise SdkContractError(
            f"Python interpreter {executable} started with the GIL enabled"
        )
    return identity


def sdk_python_executable(root: Path) -> Path:
    """Return the SDK-owned interpreter after checking its ABI and runtime mode."""

    root = _validate_sdk_layout(root)
    executable = root / "bin" / (
        "termin_python.exe" if os.name == "nt" else "termin_python"
    )
    if not executable.is_file():
        raise SdkContractError(
            f"Termin SDK is missing its canonical Python launcher: {executable}"
        )
    verify_python_executable(root, executable)
    return executable


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
            wheel_records = [name for name in names if name.endswith(".dist-info/WHEEL")]
            if len(candidates) != 1:
                raise SdkContractError(
                    f"Wheel {path.name} has {len(candidates)} METADATA records"
                )
            if len(wheel_records) != 1:
                raise SdkContractError(
                    f"Wheel {path.name} has {len(wheel_records)} WHEEL records"
                )
            parsed = Parser().parsestr(archive.read(candidates[0]).decode("utf-8"))
            wheel_record = Parser().parsestr(
                archive.read(wheel_records[0]).decode("utf-8")
            )
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
        abi_tags=tuple(
            tag.rsplit("-", 2)[1]
            for tag in (wheel_record.get_all("Tag") or ())
            if tag.count("-") == 2
        ),
        path=path,
    )


def _validate_wheel_abi(contract: SdkContract, wheel: WheelMetadata) -> None:
    if wheel.native_members and contract.python_abi.wheel_abi_tag not in wheel.abi_tags:
        rendered = ", ".join(wheel.abi_tags) or "<missing>"
        raise SdkContractError(
            f"Native wheel {wheel.path.name} has ABI tag(s) {rendered}; "
            f"SDK requires {contract.python_abi.wheel_abi_tag}"
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
        _validate_wheel_abi(contract, exact[0])
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
        _validate_wheel_abi(contract, compatible[0])
        return compatible[0]

    available = ", ".join(sorted(wheel.version for wheel in candidates)) or "<none>"
    raise SdkContractError(
        f"SDK wheelhouse has no payload-compatible {name} for manifest version "
        f"{manifest_version}; available: {available}"
    )


def termin_requirement_versions(contract: SdkContract) -> dict[str, str]:
    """Return exact SDK-owned requirements needed by this application."""

    selected = _selected_termin_wheels(contract)
    return {name: selected[name].version for name in sorted(selected)}


def _selected_termin_wheels(contract: SdkContract) -> dict[str, WheelMetadata]:
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
    return selected


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


def _installed_distribution_roots() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            result[normalize_distribution(name)] = Path(
                distribution.locate_file("")
            ).resolve()
    return result


def verify_installed_payloads(
    contract: SdkContract,
    distribution_roots: dict[str, Path] | None = None,
) -> None:
    """Reject same-version wheels whose installed files differ from the SDK.

    Termin rebuilds have historically reused a distribution version. Version
    equality alone can therefore combine current SDK libraries with stale
    pure-Python modules from a venv. Compare every wheel payload byte, while
    ignoring installer-generated dist-info files.
    """

    roots = (
        distribution_roots
        if distribution_roots is not None
        else _installed_distribution_roots()
    )
    errors: list[str] = []
    for name, wheel in sorted(_selected_termin_wheels(contract).items()):
        try:
            with ZipFile(wheel.path) as archive:
                payload_members = [
                    member
                    for member in archive.namelist()
                    if not member.endswith("/") and ".dist-info/" not in member
                ]
                if not payload_members:
                    continue
                root = roots.get(name)
                if root is None:
                    errors.append(f"{name}: installed distribution root not found")
                    continue
                for member in payload_members:
                    installed_path = root / member
                    if not installed_path.is_file():
                        errors.append(f"{name}: missing installed file {member}")
                        break
                    if hashlib.sha256(archive.read(member)).digest() != hashlib.sha256(
                        installed_path.read_bytes()
                    ).digest():
                        errors.append(f"{name}: installed file differs from SDK: {member}")
                        break
        except (OSError, BadZipFile, KeyError) as exc:
            errors.append(f"{name}: cannot compare installed payload: {exc}")

    if errors:
        raise SdkContractError(
            "Termin Python payloads do not match the selected SDK; "
            "run ./install-deps.sh to refresh the venv:\n- "
            + "\n- ".join(errors)
        )


def _require_module_from_runtime_environment(module) -> None:
    origin = getattr(module, "__file__", None)
    if not origin:
        raise SdkContractError(f"Cannot determine origin of {module.__name__}")
    resolved = Path(origin).resolve()
    # Installed wheels normally live in the venv, not SDK/site-packages. Their
    # bytes were checked above; module origin must at least be inside the same
    # interpreter environment rather than an unrelated user/global install.
    interpreter_root = Path(sys.prefix).resolve()
    if (
        PROJECT_ROOT not in resolved.parents
        and interpreter_root not in resolved.parents
    ):
        raise SdkContractError(
            f"{module.__name__} imported from unrelated location: {resolved}"
        )


def verify_imports(contract: SdkContract) -> None:
    import tcbase  # noqa: F401
    import tcgui
    import tgfx
    import termin.dispatch
    import termin.display
    import termin.display.window
    import termin.gui_native
    import termin.gui_native.window
    from termin.dispatch import Dispatcher
    from termin.display.window import WindowedGraphicsSession
    from termin.gui_native import OffscreenGuiComposition
    from termin.gui_native.window import GuiWindowAdapter, dynamic_texture_lease
    from tgfx import Tgfx2Context, configure_default_shader_runtime

    required = (
        tcgui,
        Dispatcher,
        GuiWindowAdapter,
        dynamic_texture_lease,
        OffscreenGuiComposition,
        WindowedGraphicsSession,
        Tgfx2Context,
        configure_default_shader_runtime,
    )
    if any(value is None for value in required):
        raise SdkContractError("One or more required Termin runtime exports are unavailable")
    if not hasattr(Tgfx2Context, "from_runtime"):
        raise SdkContractError(
            "Tgfx2Context.from_runtime is required by the windowed graphics contract"
        )
    for composition in (GuiWindowAdapter, OffscreenGuiComposition):
        if not callable(getattr(composition, "set_unhandled_key_handler", None)):
            raise SdkContractError(
                f"{composition.__name__}.set_unhandled_key_handler is required "
                "by the native command routing contract"
            )
    for module in (
        tcbase,
        tcgui,
        tgfx,
        termin.dispatch,
        termin.display,
        termin.display.window,
        termin.gui_native,
        termin.gui_native.window,
    ):
        _require_module_from_runtime_environment(module)


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

    python_executable = subparsers.add_parser(
        "python-executable",
        help="print the validated canonical Python interpreter from the SDK",
    )
    add_sdk_options(python_executable)

    verify_python = subparsers.add_parser(
        "verify-python",
        help="verify that the current interpreter exactly matches the SDK ABI",
    )
    add_sdk_options(verify_python)

    verify_python_executable_parser = subparsers.add_parser(
        "verify-python-executable",
        help="verify another Python executable against the SDK ABI",
    )
    add_sdk_options(verify_python_executable_parser)
    verify_python_executable_parser.add_argument(
        "--python", required=True, help="Python executable to probe"
    )

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
        if args.command == "python-executable":
            print(sdk_python_executable(root))
            return 0
        if args.command == "verify-python-executable":
            identity = verify_python_executable(root, Path(args.python))
            print(f"Python runtime verified: {Path(args.python)} ({identity})")
            return 0
        if args.command == "requirements":
            # Resolving filenames does not execute wheel code. This command is
            # intentionally usable from an ambient bootstrap Python before the
            # SDK-owned virtual environment exists.
            contract = _read_contract(root, interpreter_abi=None)
            for requirement in termin_requirement_closure(contract):
                print(requirement)
            return 0

        contract = load_contract(root)
        if args.command == "verify-python":
            print(f"Python runtime verified: {sys.executable} ({contract.python_abi})")
            return 0
        if args.command == "verify-installed":
            # verify_installed also checks the wheelhouse closure, ensuring the
            # saved SDK can reproduce this environment.
            verify_installed(contract)
            verify_installed_payloads(contract)
            if args.imports:
                verify_imports(contract)
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
