from __future__ import annotations

import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

from diffusion_editor.sdk_runtime import (
    SdkContractError,
    load_contract,
    resolve_sdk,
    termin_requirement_closure,
    verify_installed,
    verify_installed_payloads,
)


NATIVE_VERSION = "0.1.0+sdk123"


def _write_wheel(
    sdk: Path,
    name: str,
    version: str,
    *requires: str,
    native_member: str | None = None,
    native_payload: bytes = b"native-binding",
    payload_members: dict[str, bytes] | None = None,
) -> None:
    wheel_name = name.replace("-", "_")
    path = sdk / "wheels" / f"{wheel_name}-{version}-py3-none-any.whl"
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
        if native_member is not None:
            archive.writestr(native_member, native_payload)
        for member, payload in (payload_members or {}).items():
            archive.writestr(member, payload)


def _make_sdk(tmp_path: Path, *, stale_tgfx: bool = False) -> Path:
    sdk = tmp_path / "sdk"
    (sdk / "lib").mkdir(parents=True)
    (sdk / "wheels").mkdir()
    (sdk / "lib/python/site-packages").mkdir(parents=True)
    versions = {
        "tcbase": NATIVE_VERSION,
        "tcgui": "0.1.0",
        "tgfx": NATIVE_VERSION,
        "termin-display": NATIVE_VERSION,
        "termin-scene": NATIVE_VERSION,
    }
    payload = {
        "schema": 1,
        "python_abi": f"{sys.version_info.major}.{sys.version_info.minor}",
        "site_packages": "lib/python/site-packages",
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


def test_requirement_closure_is_exact_and_includes_sdk_transitives(tmp_path: Path):
    contract = load_contract(_make_sdk(tmp_path))

    assert termin_requirement_closure(contract) == (
        f"tcbase=={NATIVE_VERSION}",
        "tcgui==0.1.0",
        f"termin-display=={NATIVE_VERSION}",
        f"termin-scene=={NATIVE_VERSION}",
        f"tgfx=={NATIVE_VERSION}",
    )


def test_stale_wheelhouse_fails_before_install(tmp_path: Path):
    contract = load_contract(_make_sdk(tmp_path, stale_tgfx=True))

    with pytest.raises(SdkContractError, match="no payload-compatible tgfx"):
        termin_requirement_closure(contract)


def test_installed_native_build_must_match_manifest(tmp_path: Path):
    contract = load_contract(_make_sdk(tmp_path))
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

    with pytest.raises(SdkContractError, match="interpreter ABI"):
        load_contract(sdk, interpreter_abi="9.9")


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
        installed = sdk / "lib/python/site-packages" / native_member
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

    contract = load_contract(sdk)

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
            load_contract(sdk),
            {"tcgui": installed},
        )

    (installed / "tcgui/widgets/renderer.py").write_bytes(
        b"current-sdk-renderer\n"
    )
    verify_installed_payloads(load_contract(sdk), {"tcgui": installed})
