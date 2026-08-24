from __future__ import annotations

import json
from pathlib import Path
import runpy


MODULE = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "experiment-multiple-angles-grid.py"
    )
)


def test_candidate_seeds_replace_the_whole_batch_deterministically() -> None:
    assert MODULE["candidate_seeds"](2399806712, 4) == [
        2399806712,
        2399806713,
        2399806714,
        2399806715,
    ]


def test_rejected_batch_is_archived_without_leaving_mixable_views(
    tmp_path: Path,
) -> None:
    outputs = [tmp_path / "mv-eye-000.png", tmp_path / "mv-eye-045.png"]
    for output in outputs:
        output.write_bytes(b"image")
    (tmp_path / "contact-eye.png").write_bytes(b"contact")
    jobs = [
        {"azimuth_degrees": angle, "output": output}
        for angle, output in zip((0, 45), outputs)
    ]
    manifest = {
        "seed": 100,
        "jobs": [
            {"azimuth_degrees": angle, "output": str(output.resolve())}
            for angle, output in zip((0, 45), outputs)
        ],
        "orientation_verification": {"accepted": False},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    archive = MODULE["_archive_batch"](
        tmp_path,
        jobs,
        seed=100,
        reason="orientation verification failed",
    )

    assert not any(output.exists() for output in outputs)
    assert not (tmp_path / "manifest.json").exists()
    assert (archive / "contact-eye.png").is_file()
    archived = json.loads((archive / "manifest.json").read_text())
    assert archived["batch_status"] == "rejected"
    assert all(Path(record["output"]).parent == archive for record in archived["jobs"])
