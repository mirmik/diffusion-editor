from pathlib import Path
import runpy


RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/run-canonical-identity-experiment.py")
)


def test_blender_command_propagates_python_failures():
    source = Path("/assets/character.blend")

    assert RUNNER["_blender_command"](source, "blend") == [
        "blender",
        "--background",
        "--disable-autoexec",
        "--python-exit-code",
        "1",
        str(source),
    ]


def test_fbx_blender_command_starts_from_factory_scene():
    assert RUNNER["_blender_command"](Path("/assets/character.fbx"), "fbx")[-1] == (
        "--factory-startup"
    )


def test_camera_jitter_is_deterministic_and_identity_specific():
    config = {
        "views": {
            "azimuths": [0, 45],
            "camera_azimuth_jitter": {
                "minimum_absolute_degrees": 5,
                "maximum_degrees": 20,
                "replicas_per_azimuth": 2,
                "seed": 17,
            },
        }
    }
    rain = RUNNER["_camera_jitter_pairs"](config, "rain")
    assert rain == RUNNER["_camera_jitter_pairs"](config, "rain")
    assert rain != RUNNER["_camera_jitter_pairs"](config, "jay")
    assert len(rain) == 4
    for nominal, actual in rain:
        delta = (actual - nominal + 180.0) % 360.0 - 180.0
        assert 5.0 <= abs(delta) <= 20.0
