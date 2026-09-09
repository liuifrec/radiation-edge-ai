"""Compile the accepted NASA BPS R1 v2 KL720 BIE into NEF without re-quantization.

This stage is downstream of the successful BIE biological-equivalence holdout.
It verifies the exact accepted BIE and its post-equivalence freeze, then creates
KL720 ModelConfig directly from that BIE and calls ktc.compile. No ONNX
optimization, calibration, PTQ, BIE regeneration, or holdout inference occurs.

Python 3.7 compatible for the pinned Kneron Toolchain v0.33.1 image.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import ktc

FROZEN_BIE_SHA256 = (
    "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
)
FROZEN_EQUIVALENCE_FREEZE_SHA256 = (
    "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
)
FROZEN_EQUIVALENCE_SUMMARY_SHA256 = (
    "c15776c6820a4aa28fff8e18c16b576164ce3ed2cad61ba2057efdef79800ca4"
)
FROZEN_PRE_HOLDOUT_BIE_FREEZE_SHA256 = (
    "63192365e77aa269d5ec792f2b184f8ada6f1603c451da47b1d1718ceb3dcc57"
)
PINNED_TOOLCHAIN_IMAGE = (
    "kneron/toolchain@sha256:2207d99c9f78deef90647a3da3047ec3f7942ec689385a586fee476f77e89041"
)
EXPECTED_TOOLCHAIN_VERSION = "kneron/toolchain:v0.33.1"
EXPECTED_PLATFORM = "720"
EXPECTED_MODEL_ID = 32770
EXPECTED_MODEL_VERSION = "8b29"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object: {}".format(path))
    return obj


def check_sha(path, expected, label):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            "{} SHA256 mismatch: expected {}, got {}".format(label, expected, actual)
        )
    return actual


def toolchain_version():
    for candidate in (Path("/workspace/version.txt"), Path("/workspace/VERSION")):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bie", required=True)
    parser.add_argument("--equivalence-freeze", required=True)
    parser.add_argument("--equivalence-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-name",
        default="r1_countnet_v2_final_kl720.nef",
    )
    args = parser.parse_args()

    bie_path = Path(args.bie).resolve()
    freeze_path = Path(args.equivalence_freeze).resolve()
    equivalence_summary_path = Path(args.equivalence_summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.output_name
    compile_summary_path = output_dir / "nef_compile_summary.json"

    if compile_summary_path.exists():
        raise RuntimeError(
            "NEF compile result already exists; refusing overwrite: {}".format(
                compile_summary_path
            )
        )

    bie_sha = check_sha(bie_path, FROZEN_BIE_SHA256, "Accepted BIE")
    freeze_sha = check_sha(
        freeze_path, FROZEN_EQUIVALENCE_FREEZE_SHA256, "BIE equivalence freeze"
    )
    equivalence_summary_sha = check_sha(
        equivalence_summary_path,
        FROZEN_EQUIVALENCE_SUMMARY_SHA256,
        "BIE equivalence summary",
    )

    freeze = read_json(freeze_path)
    checks = {
        "status": "FROZEN_KL720_INT8_BIE_BIOLOGICAL_EQUIVALENCE_PASS_BEFORE_NEF",
        "bie_sha256": FROZEN_BIE_SHA256,
        "pre_holdout_bie_freeze_sha256": FROZEN_PRE_HOLDOUT_BIE_FREEZE_SHA256,
        "bie_holdout_equivalence_summary_sha256": FROZEN_EQUIVALENCE_SUMMARY_SHA256,
        "all_seven_original_biological_gates_pass": True,
        "exact_fp32_direction_peak_signature_reproduced": True,
        "raw_phenotype_table_read_by_bie_equivalence_stage": False,
        "ptq_tuning_after_holdout": False,
        "ptq_retuning_after_this_freeze_authorized": False,
        "authorized_next_stage": "COMPILE_THIS_EXACT_ACCEPTED_BIE_TO_NEF_WITHOUT_REQUANTIZATION",
    }
    for key, expected in checks.items():
        if freeze.get(key) != expected:
            raise RuntimeError(
                "Equivalence freeze invariant mismatch for {}: expected {!r}, got {!r}".format(
                    key, expected, freeze.get(key)
                )
            )

    # Parse the exact result artifact only to ensure it remains readable; the
    # immutable SHA above is the primary identity and no outcome is used to tune
    # compilation.
    read_json(equivalence_summary_path)

    version = toolchain_version()
    if version != EXPECTED_TOOLCHAIN_VERSION:
        raise RuntimeError(
            "Unexpected Kneron toolchain version: expected {}, got {}".format(
                EXPECTED_TOOLCHAIN_VERSION, version or "UNKNOWN"
            )
        )

    print("Radiation Edge AI - NASA BPS R1 v2 frozen BIE -> KL720 NEF")
    print("accepted BIE SHA256: {}".format(bie_sha))
    print("BIE equivalence freeze SHA256: {}".format(freeze_sha))
    print("BIE equivalence summary SHA256: {}".format(equivalence_summary_sha))
    print("toolchain image: {}".format(PINNED_TOOLCHAIN_IMAGE))
    print("toolchain version: {}".format(version))
    print("platform/model: {}/{}:{}".format(EXPECTED_PLATFORM, EXPECTED_MODEL_ID, EXPECTED_MODEL_VERSION))
    print("all seven biological gates already frozen PASS: YES")
    print("re-quantization performed: NO")
    print("holdout inference performed by this stage: NO")

    print("")
    print("[Create ModelConfig directly from accepted BIE]")
    km = ktc.ModelConfig(
        EXPECTED_MODEL_ID,
        EXPECTED_MODEL_VERSION,
        EXPECTED_PLATFORM,
        bie_path=str(bie_path),
    )
    print("ModelConfig: PASS")

    print("")
    print("[Compile NEF]")
    started = time.time()
    returned = Path(ktc.compile([km])).resolve()
    elapsed = time.time() - started
    if not returned.is_file():
        raise RuntimeError("Kneron compile returned missing NEF path: {}".format(returned))

    if returned.resolve() != destination.resolve():
        shutil.copy2(str(returned), str(destination))
    if not destination.is_file():
        raise RuntimeError("Persistent NEF was not created: {}".format(destination))

    nef_sha = sha256_file(destination)
    summary = {
        "status": "COMPLETE_KL720_NEF_COMPILE_FROM_ACCEPTED_BIE",
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_bie": str(bie_path),
        "source_bie_sha256": bie_sha,
        "bie_equivalence_freeze": str(freeze_path),
        "bie_equivalence_freeze_sha256": freeze_sha,
        "bie_equivalence_summary": str(equivalence_summary_path),
        "bie_equivalence_summary_sha256": equivalence_summary_sha,
        "pre_holdout_bie_freeze_sha256": FROZEN_PRE_HOLDOUT_BIE_FREEZE_SHA256,
        "all_seven_original_biological_gates_pass": True,
        "exact_fp32_direction_peak_signature_reproduced": True,
        "pinned_toolchain_image": PINNED_TOOLCHAIN_IMAGE,
        "toolchain_version": version,
        "platform": EXPECTED_PLATFORM,
        "model_id": EXPECTED_MODEL_ID,
        "model_version": EXPECTED_MODEL_VERSION,
        "nef_returned": str(returned),
        "nef_persistent": str(destination),
        "nef_sha256": nef_sha,
        "nef_size_bytes": destination.stat().st_size,
        "compile_elapsed_seconds": elapsed,
        "onnx_optimization_repeated": False,
        "ptq_repeated": False,
        "bie_regenerated": False,
        "holdout_inference_performed": False,
        "authorized_next_stage": "ONE_SAMPLE_NEF_VS_BIE_COMPILE_INTEGRITY_GATE",
    }
    compile_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("NASA BPS R1 V2 KL720 NEF COMPILE COMPLETE: YES")
    print("ACCEPTED BIE PRESERVED: YES")
    print("NEF: {}".format(destination))
    print("NEF SHA256: {}".format(nef_sha))
    print("NEF size: {:.2f} MiB".format(destination.stat().st_size / float(1024 ** 2)))
    print("Elapsed: {:.1f} s".format(elapsed))
    print("Summary: {}".format(compile_summary_path))
    print("Summary SHA256: {}".format(sha256_file(compile_summary_path)))
    print("re-quantization performed: NO")
    print("holdout inference performed by this stage: NO")
    print("NEXT GATE: one-sample NEF vs frozen BIE compile-integrity verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
