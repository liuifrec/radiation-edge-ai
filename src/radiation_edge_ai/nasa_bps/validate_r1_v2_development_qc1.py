"""Validate the repaired NASA BPS R1 v2 DEVELOPMENT manifest (QC1).

This thin wrapper preserves the original pre-repair validator and its audit
outputs. It pins the repaired 7,200-nucleus manifest SHA256, points validation
to the QC1 manifest by default, and writes QC1 results to a separate directory.
The underlying hard checks remain exactly those in
``validate_r1_v2_development.py``.

No final-holdout manifest, image, or phenotype data are read.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPAIRED_DEV_MANIFEST_SHA256 = (
    "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9"
)


def load_validator():
    path = Path(__file__).with_name("validate_r1_v2_development.py")
    spec = importlib.util.spec_from_file_location("r1_v2_development_validator_qc1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_flag(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def main() -> int:
    validator = load_validator()
    validator.FROZEN_DEV_MANIFEST_SHA256 = REPAIRED_DEV_MANIFEST_SHA256

    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"

    if not has_flag("--manifest"):
        sys.argv.extend(
            [
                "--manifest",
                str(
                    metadata_root
                    / "r1_v2_freeze"
                    / "r1_v2_development_manifest_100_qc1.csv"
                ),
            ]
        )

    if not has_flag("--output-dir"):
        sys.argv.extend(
            [
                "--output-dir",
                str(data_root / "nasa_bps_microscopy" / "r1_v2_development" / "qc_qc1"),
            ]
        )

    print("Radiation Edge AI - NASA BPS R1 v2 DEVELOPMENT QC1 wrapper")
    print(f"Pinned repaired manifest SHA256: {REPAIRED_DEV_MANIFEST_SHA256}")
    print("Underlying hard checks: validate_r1_v2_development.py")
    print("FINAL HOLDOUT STATUS: UNTOUCHED")
    print("")

    return int(validator.main())


if __name__ == "__main__":
    raise SystemExit(main())
