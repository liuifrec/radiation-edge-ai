"""Freeze the successful NASA BPS R1 v2 KL720 INT8 BIE holdout result.

This stage performs no inference and no quantization. It pins the exact BIE,
pre-holdout BIE freeze, and completed BIE deployment-equivalence summary before
NEF compilation. The accepted biological result is therefore immutable before
creating the physical-device artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FROZEN_BIE_SHA256 = (
    "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
)
FROZEN_PRE_HOLDOUT_BIE_FREEZE_SHA256 = (
    "63192365e77aa269d5ec792f2b184f8ada6f1603c451da47b1d1718ceb3dcc57"
)
FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256 = (
    "c15776c6820a4aa28fff8e18c16b576164ce3ed2cad61ba2057efdef79800ca4"
)
EXPECTED_DIRECTION_COUNTS = {
    "overall": "10/11",
    "4h": "4/4",
    "24+48h": "6/7",
    "by_source": {
        "BALBCF2": "4/5",
        "C57BLF2": "3/3",
        "C57BLF3": "3/3",
    },
}
EXPECTED_PEAK = "3/3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return obj


def check_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    deployment_root = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    int8_root = deployment_root / "kneron_int8"
    equivalence_root = int8_root / "bie_holdout_equivalence"
    parser.add_argument(
        "--bie",
        default=str(int8_root / "r1_countnet_v2_final_kl720_int8.bie"),
    )
    parser.add_argument(
        "--pre-holdout-bie-freeze",
        default=str(int8_root / "bie_deployment_candidate_freeze.json"),
    )
    parser.add_argument(
        "--equivalence-summary",
        default=str(equivalence_root / "bie_holdout_equivalence_summary.json"),
    )
    args = parser.parse_args()

    bie_path = Path(args.bie).resolve()
    prefreeze_path = Path(args.pre_holdout_bie_freeze).resolve()
    summary_path = Path(args.equivalence_summary).resolve()
    freeze_path = equivalence_root / "bie_holdout_equivalence_freeze.json"

    bie_sha = check_sha(bie_path, FROZEN_BIE_SHA256, "BIE")
    prefreeze_sha = check_sha(
        prefreeze_path,
        FROZEN_PRE_HOLDOUT_BIE_FREEZE_SHA256,
        "Pre-holdout BIE freeze",
    )
    summary_sha = check_sha(
        summary_path,
        FROZEN_BIE_EQUIVALENCE_SUMMARY_SHA256,
        "BIE equivalence summary",
    )

    # The exact summary hash is the primary outcome identity. Parse it as JSON
    # only to ensure it remains a readable structured artifact.
    read_json(summary_path)

    freeze = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_KL720_INT8_BIE_BIOLOGICAL_EQUIVALENCE_PASS_BEFORE_NEF",
        "bie_sha256": bie_sha,
        "pre_holdout_bie_freeze_sha256": prefreeze_sha,
        "bie_holdout_equivalence_summary_sha256": summary_sha,
        "observed_bag_metrics": {
            "mae": 0.7922,
            "rmse": 0.8893,
            "pearson": 0.9501,
            "spearman": 0.5729,
        },
        "observed_delta_metrics": {
            "mae": 0.9405,
            "pearson": 0.9634,
            "spearman": 0.8182,
        },
        "direction_counts": EXPECTED_DIRECTION_COUNTS,
        "peak_time_recovery": EXPECTED_PEAK,
        "all_seven_original_biological_gates_pass": True,
        "exact_fp32_direction_peak_signature_reproduced": True,
        "raw_phenotype_table_read_by_bie_equivalence_stage": False,
        "ptq_tuning_after_holdout": False,
        "ptq_retuning_after_this_freeze_authorized": False,
        "authorized_next_stage": "COMPILE_THIS_EXACT_ACCEPTED_BIE_TO_NEF_WITHOUT_REQUANTIZATION",
    }

    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    if freeze_path.exists():
        existing = read_json(freeze_path)
        old = dict(existing)
        new = dict(freeze)
        old.pop("created_utc", None)
        new.pop("created_utc", None)
        if old != new:
            raise RuntimeError(
                f"Existing BIE equivalence freeze differs from accepted result: {freeze_path}"
            )
        print("Radiation Edge AI - NASA BPS R1 v2 BIE biological-equivalence freeze")
        print("existing equivalence freeze verified without overwrite: YES")
    else:
        freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
        print("Radiation Edge AI - NASA BPS R1 v2 BIE biological-equivalence freeze")
        print("new equivalence freeze written: YES")

    print(f"BIE SHA256: {bie_sha}")
    print(f"pre-holdout BIE freeze SHA256: {prefreeze_sha}")
    print(f"BIE equivalence summary SHA256: {summary_sha}")
    print("all seven original biological gates: PASS")
    print("exact FP32 direction/peak signature reproduced: YES")
    print("raw phenotype table read by this freeze: NO")
    print("post-holdout PTQ retuning authorized: NO")
    print(f"freeze: {freeze_path}")
    print(f"freeze SHA256: {sha256_file(freeze_path)}")
    print("NASA BPS R1 V2 KL720 INT8 BIE BIOLOGICAL EQUIVALENCE FROZEN: YES")
    print("NEXT GATE: compile this exact accepted BIE to NEF without re-quantization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
