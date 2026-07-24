from __future__ import annotations

import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

import pytest

from diffusion_editor.sdk_runtime import (
    PythonAbiIdentity,
    SdkContractError,
    load_contract,
    resolve_sdk,
    sdk_python_executable,
    termin_requirement_closure,
    verify_installed,
    verify_installed_payloads,
)


NATIVE_VERSION = "0.1.0+sdk123"
TARGET_ABI = PythonAbiIdentity(
    version="3.14",
    soabi="cpython-314t-x86_64-linux-gnu",
    free_threaded=True,
    py_gil_disabled=True,
)


def _write_wheel(
    sdk: Path,
    name: str,
    version: str,
    *requires: str,
    native_member: str | None = None,
    native_payload: bytes = b"native-binding",
    payload_members: dict[str, bytes] | None = None,
    abi_tag: str | None = None,
) -> None:
    wheel_name = name.replace("-", "_")
    tag = abi_tag or ("cp314-cp314t-linux_x86_64" if native_member else "py3-none-any")
    path = sdk / "wheels" / f"{wheel_name}-{version}-{tag}.whl"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            f"{wheel_name}-{version}.dist-info/METADATA",
            "\n".join(metadata) + "\n",
        )
        archive.writestr(
            f"{wheel_name}-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: diffusion-editor-tests\n"
            f"Root-Is-Purelib: {'false' if native_member else 'true'}\n"
            f"Tag: {tag}\n",
        )
        if native_member is not None:
            archive.writestr(native_member, native_payload)
        for member, payload in (payload_members or {}).items():
            archive.writestr(member, payload)


def _make_sdk(tmp_path: Path, *, stale_tgfx: bool = False) -> Path:
    sdk = tmp_path / "sdk"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "wheels").mkdir()
    (sdk / "lib/python3.14t/site-packages").mkdir(parents=True)
    versions = {
        "tcbase": NATIVE_VERSION,
        "tcgui": "0.1.0",
        "tgfx": NATIVE_VERSION,
        "termin-display": NATIVE_VERSION,
        "termin-scene": NATIVE_VERSION,
    }
    payload = {
        "schema": 3,
        "python_abi": {
            "version": TARGET_ABI.version,
            "soabi": TARGET_ABI.soabi,
            "free_threaded": TARGET_ABI.free_threaded,
            "py_gil_disabled": TARGET_ABI.py_gil_disabled,
        },
        "site_packages": "lib/python3.14t/site-packages",
        "distributions": [
            {"name": name, "version": version, "kind": "termin"}
            for name, version in versions.items()
        ],
    }
    (sdk / "python-runtime-manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    _write_wheel(sdk, "tcbase", NATIVE_VERSION)
    _write_wheel(sdk, "tcgui", "0.1.0", "tcbase", "tgfx")
    _write_wheel(
        sdk,
        "tgfx",
        "0.1.0+sdk-old" if stale_tgfx else NATIVE_VERSION,
        "tcbase",
    )
    _write_wheel(sdk, "termin-display", NATIVE_VERSION, "termin-scene", "tgfx")
    _write_wheel(sdk, "termin-scene", NATIVE_VERSION, "tcbase")
    return sdk


def _load(sdk: Path):
    return load_contract(sdk, interpreter_abi=TARGET_ABI)


def test_requirement_closure_is_exact_and_includes_sdk_transitives(tmp_path: Path):
    contract = _load(_make_sdk(tmp_path))

    assert termin_requirement_closure(contract) == (
        f"tcbase=={NATIVE_VERSION}",
        "tcgui==0.1.0",
        f"termin-display=={NATIVE_VERSION}",
        f"termin-scene=={NATIVE_VERSION}",
        f"tgfx=={NATIVE_VERSION}",
    )


def test_stale_wheelhouse_fails_before_install(tmp_path: Path):
    contract = _load(_make_sdk(tmp_path, stale_tgfx=True))

    with pytest.raises(SdkContractError, match="no payload-compatible tgfx"):
        termin_requirement_closure(contract)


def test_installed_native_build_must_match_manifest(tmp_path: Path):
    contract = _load(_make_sdk(tmp_path))
    installed = {
        "tcbase": NATIVE_VERSION,
        "tcgui": "0.1.0",
        "tgfx": "0.1.0+sdk-other",
        "termin-display": NATIVE_VERSION,
        "termin-scene": NATIVE_VERSION,
    }

    with pytest.raises(SdkContractError, match="tgfx: installed"):
        verify_installed(contract, installed)


def test_saved_sdk_path_is_resolved_without_ambient_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sdk = _make_sdk(tmp_path)
    state = tmp_path / ".termin-sdk"
    state.write_text("sdk\n", encoding="utf-8")
    monkeypatch.delenv("TERMIN_SDK", raising=False)

    assert resolve_sdk(state_file=state) == sdk.resolve()


def test_manifest_rejects_wrong_interpreter_abi(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    gil_abi = PythonAbiIdentity(
        version="3.14",
        soabi="cpython-314-x86_64-linux-gnu",
        free_threaded=False,
        py_gil_disabled=False,
    )

    with pytest.raises(SdkContractError, match="interpreter ABI"):
        load_contract(sdk, interpreter_abi=gil_abi)


def test_manifest_rejects_soabi_without_free_threading_suffix(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    manifest = sdk / "python-runtime-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["python_abi"]["soabi"] = "cpython-314-x86_64-linux-gnu"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SdkContractError, match="SOABI.*disagrees"):
        _load(sdk)


def test_manifest_rejects_missing_free_threading_markers(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    manifest = sdk / "python-runtime-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    del payload["python_abi"]["py_gil_disabled"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SdkContractError, match="requires version, soabi"):
        _load(sdk)


def test_legacy_manifest_is_rejected(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    manifest = sdk / "python-runtime-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema"] = 1
    payload["python_abi"] = "3.14"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SdkContractError, match="schema must be 3"):
        _load(sdk)


def test_native_cp314_wheel_is_rejected_for_cp314t_sdk(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    next((sdk / "wheels").glob("tcbase-*.whl")).unlink()
    _write_wheel(
        sdk,
        "tcbase",
        NATIVE_VERSION,
        native_member="tcbase/_native.so",
        abi_tag="cp314-cp314-linux_x86_64",
    )

    with pytest.raises(SdkContractError, match=r"ABI tag.*cp314; SDK requires cp314t"):
        termin_requirement_closure(_load(sdk))


def test_sdk_python_launcher_must_match_manifest_and_start_without_gil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sdk = _make_sdk(tmp_path)
    launcher = sdk / "bin/termin_python"
    launcher.parent.mkdir()
    launcher.touch(mode=0o755)
    output = {
        "version": TARGET_ABI.version,
        "soabi": TARGET_ABI.soabi,
        "free_threaded": True,
        "py_gil_disabled": True,
        "gil_enabled": False,
    }
    monkeypatch.setattr(
        "diffusion_editor.sdk_runtime.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(output), stderr=""
        ),
    )

    assert sdk_python_executable(sdk) == launcher.resolve()


def test_sdk_python_launcher_rejects_runtime_gil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sdk = _make_sdk(tmp_path)
    launcher = sdk / "bin/termin_python"
    launcher.parent.mkdir()
    launcher.touch(mode=0o755)
    output = {
        "version": TARGET_ABI.version,
        "soabi": TARGET_ABI.soabi,
        "free_threaded": True,
        "py_gil_disabled": True,
        "gil_enabled": True,
    }
    monkeypatch.setattr(
        "diffusion_editor.sdk_runtime.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=json.dumps(output), stderr=""
        ),
    )

    with pytest.raises(SdkContractError, match="started with the GIL enabled"):
        sdk_python_executable(sdk)


def test_retagged_wheel_is_accepted_only_when_native_payload_matches(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    dependencies = {
        "tcbase": (),
        "tgfx": ("tcbase",),
        "termin-display": ("termin-scene", "tgfx"),
        "termin-scene": ("tcbase",),
    }
    for name, requires in dependencies.items():
        next((sdk / "wheels").glob(f"{name.replace('-', '_')}-*.whl")).unlink()
        native_member = f"{name.replace('-', '_')}/_native.so"
        installed = sdk / "lib/python3.14t/site-packages" / native_member
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(f"matching-{name}".encode())
        _write_wheel(
            sdk,
            name,
            "0.1.0+sdk456",
            *requires,
            native_member=native_member,
            native_payload=f"matching-{name}".encode(),
        )

    contract = _load(sdk)

    requirements = termin_requirement_closure(contract)
    assert "tgfx==0.1.0+sdk456" in requirements
    assert "termin-display==0.1.0+sdk456" in requirements


def test_installed_payload_must_match_selected_sdk_wheel(tmp_path: Path):
    sdk = _make_sdk(tmp_path)
    tcgui_wheel = next((sdk / "wheels").glob("tcgui-*.whl"))
    tcgui_wheel.unlink()
    _write_wheel(
        sdk,
        "tcgui",
        "0.1.0",
        "tcbase",
        "tgfx",
        payload_members={"tcgui/widgets/renderer.py": b"current-sdk-renderer\n"},
    )
    installed = tmp_path / "installed"
    (installed / "tcgui/widgets").mkdir(parents=True)
    (installed / "tcgui/widgets/renderer.py").write_bytes(b"stale-venv-renderer\n")

    with pytest.raises(SdkContractError, match="installed file differs from SDK"):
        verify_installed_payloads(
            _load(sdk),
            {"tcgui": installed},
        )

    (installed / "tcgui/widgets/renderer.py").write_bytes(
        b"current-sdk-renderer\n"
    )
    verify_installed_payloads(_load(sdk), {"tcgui": installed})
