"""Read-only post-release evidence audit for NASA BPS R1 v2.

This module is additive and must not alter the frozen RC1 model, deployment
artifacts, holdout predictions, gates, or result directories.  It verifies the
authoritative frozen inputs it reads, compares actual per-contrast identities,
and emits a separate post-release audit record.

It performs no model inference and has no Kneron/hardware imports.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python >=3.11 is the intended audit runtime
    tomllib = None

import numpy as np

from radiation_edge_ai.audit_utils import (
    enrich_contrast_sample_names,
    finite_float,
    finite_int,
    join_contrasts,
    peak_series_audit,
    read_csv_rows,
    read_json_object,
    refuse_nonempty_output_dir,
    reject_nonfinite_json,
    sha256_file,
    write_csv_rows,
)

FP32_REFERENCE_FREEZE_SHA256 = "abb58026d4f798554ef7e44346a989499242fcbd89d8c6629a1c4209d6c16ce3"
BIE_EQUIVALENCE_SUMMARY_SHA256 = "c15776c6820a4aa28fff8e18c16b576164ce3ed2cad61ba2057efdef79800ca4"
BIE_EQUIVALENCE_FREEZE_SHA256 = "e901d4be044c5ea5507b3b60643043eb8aa0af5643a0b7fba2f9bdd7c588bb0a"
PHYSICAL_PREDICTIONS_SHA256 = "91ea895a16cd97067af63b5c5c4434e972629bfda4c9582ad456a79e007b6d46"
PHYSICAL_INFERENCE_SUMMARY_SHA256 = (
    "7147ba4de4ba44fa64ed6b414e990970cd08ac83ea5e4b993ed38e9ad0a93aba"
)
PHYSICAL_PREDICTIONS_FREEZE_SHA256 = (
    "e4e7d9150e2e33479e2ec8894e583f0d09a3d890a6a97fbd356a1a0fb78d2b30"
)
FINAL_PHYSICAL_SUMMARY_SHA256 = "8d236ac3cee264975d0bb3de2397f15063cb213070956f45ce174baa71d69786"
FINAL_PHYSICAL_FREEZE_SHA256 = "0259ad745215dfb567a2edbd1593fb7dc789de6a4366dbcdfeb7234901b844bb"
RC_MANIFEST_SHA256 = "89604d902cb1e38c23926c9f731bb47778babce1d0c9c6ba9d49c571ed00dc67"
NEF_SHA256 = "d04f02855a8a82ae7cb5ff48af91eee69c341142d1097469890385bd23ee097b"
BIE_SHA256 = "c52b8a78c595347667bad7a34950af92adbe734179b913646d06513a88fc1dd8"
EXPECTED_NUCLEI = 2058
EXPECTED_CONTRASTS = 11
EXPOSED_DOSE = {"Fe": "0.82", "X-ray": "1.0"}

PACKAGER_REHASHED_KEYS = {
    "final_physical_equivalence_freeze",
    "final_physical_equivalence_summary",
    "physical_predictions_freeze",
    "physical_predictions",
    "physical_inference_summary",
    "nef",
    "nef_deployment_freeze",
    "bie",
    "bie_equivalence_freeze",
    "fp32_reference_freeze",
    "qc1_manifest",
}


def check_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def verify_transitive_files(parent: Mapping[str, Any], key: str, directory: Path) -> dict[str, str]:
    mapping = parent.get(key)
    if not isinstance(mapping, Mapping) or not mapping:
        raise RuntimeError(f"Missing transitive hash mapping {key!r}")
    verified: dict[str, str] = {}
    for name, meta in mapping.items():
        if not isinstance(meta, Mapping) or not isinstance(meta.get("sha256"), str):
            raise TypeError(f"Invalid hash metadata for {key}.{name}")
        path = directory / str(name)
        expected = str(meta["sha256"])
        verified[str(name)] = check_sha(path, expected, f"{key}.{name}")
    return verified


def find_exact_output_quantization_metadata(paths: list[Path]) -> dict[str, Any] | None:
    """Return exact output radix/scale only if persisted metadata is explicit and unambiguous."""
    candidates: list[dict[str, Any]] = []

    def walk(obj: Any, location: str) -> None:
        if isinstance(obj, Mapping):
            keys = {str(k).lower() for k in obj}
            if (
                "radix" in keys
                and "scale" in keys
                and any(token in location.lower() for token in ("output", "quant"))
            ):
                try:
                    radix = finite_int(obj.get("radix"), label=f"{location}.radix")
                    scale = finite_float(obj.get("scale"), label=f"{location}.scale")
                except RuntimeError:
                    pass
                else:
                    candidates.append({"radix": radix, "scale": scale, "location": location})
            for key, child in obj.items():
                walk(child, f"{location}.{key}")
        elif isinstance(obj, list):
            for index, child in enumerate(obj):
                walk(child, f"{location}[{index}]")

    for path in paths:
        if path.is_file() and path.suffix.lower() == ".json":
            try:
                obj = read_json_object(path)
            except (OSError, RuntimeError, TypeError, ValueError):
                obj = None
            if obj is not None:
                walk(obj, str(path))

    unique = {(item["radix"], item["scale"]) for item in candidates}
    if len(unique) != 1:
        return None
    radix, scale = next(iter(unique))
    return {
        "radix": radix,
        "scale": scale,
        "step": 1.0 / ((2.0**radix) * scale),
        "locations": [item["location"] for item in candidates],
    }


def quantization_diagnostic(
    physical_rows: list[dict[str, str]],
    bie_rows: list[dict[str, str]],
    descriptor: dict[str, Any] | None,
) -> dict[str, Any]:
    p_by_id = {row["sample_id"]: row for row in physical_rows}
    b_by_id = {row["sample_id"]: row for row in bie_rows}
    if len(p_by_id) != len(physical_rows) or len(b_by_id) != len(bie_rows):
        raise RuntimeError("Duplicate sample_id in physical/BIE nucleus rows")
    if set(p_by_id) != set(b_by_id):
        raise RuntimeError("Physical/BIE nucleus key sets differ")

    records: list[dict[str, Any]] = []
    for sample_id in sorted(p_by_id):
        p = p_by_id[sample_id]
        b = b_by_id[sample_id]
        pv = finite_float(p["physical_kl720_burden"], label="physical_kl720_burden")
        bv = finite_float(b["bie_burden"], label="bie_burden")
        row = {
            "sample_id": sample_id,
            "sample_name": p.get("sample_name"),
            "source_name": p.get("source_name"),
            "particle_type": p.get("particle_type"),
            "dose_Gy": p.get("dose_Gy"),
            "hr_post_exposure": p.get("hr_post_exposure"),
            "physical": pv,
            "bie": bv,
            "physical_minus_bie": pv - bv,
        }
        records.append(row)

    result: dict[str, Any] = {
        "n": len(records),
        "exact_output_quantization_metadata_available": descriptor is not None,
        "max_abs_physical_minus_bie": max(abs(row["physical_minus_bie"]) for row in records),
    }
    if descriptor is None:
        result["status"] = (
            "Exact persisted output radix/scale not found unambiguously; "
            "output-code occupancy and step histogram intentionally not inferred from rounded console text."
        )
        return result

    step = finite_float(descriptor["step"], label="output quantization step")
    if step <= 0:
        raise RuntimeError("Output quantization step must be positive")
    for row in records:
        row["physical_code_nearest"] = int(np.rint(row["physical"] / step))
        row["physical_code_grid_residual"] = row["physical"] - row["physical_code_nearest"] * step
        steps = row["physical_minus_bie"] / step
        row["physical_minus_bie_steps"] = steps
        row["nearest_integer_step_difference"] = int(np.rint(steps))
        row["step_difference_residual"] = steps - row["nearest_integer_step_difference"]

    occupancy = Counter(int(row["physical_code_nearest"]) for row in records)
    step_hist = Counter(int(row["nearest_integer_step_difference"]) for row in records)
    result.update(
        {
            "status": "Exact persisted output radix/scale used for diagnostic.",
            "descriptor": descriptor,
            "code_occupancy": {str(code): count for code, count in sorted(occupancy.items())},
            "difference_step_histogram": {
                str(code): count for code, count in sorted(step_hist.items())
            },
            "max_abs_code_grid_residual": max(
                abs(row["physical_code_grid_residual"]) for row in records
            ),
            "max_abs_step_difference_residual": max(
                abs(row["step_difference_residual"]) for row in records
            ),
            "boundary_frequency": (
                "NOT_COMPUTED: exact signed-code bounds were not persisted in the audited metadata"
            ),
        }
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["sample_name"])].append(row)
    result["by_bag"] = {
        bag: {
            "n": len(rows),
            "mean_physical_minus_bie": float(np.mean([r["physical_minus_bie"] for r in rows])),
            "max_abs_physical_minus_bie": max(abs(r["physical_minus_bie"]) for r in rows),
            "step_histogram": dict(
                Counter(int(r["nearest_integer_step_difference"]) for r in rows)
            ),
        }
        for bag, rows in sorted(grouped.items())
    }
    return result


def rc_packaging_coverage(rc_manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifact_hashes = rc_manifest.get("artifact_sha256") or {}
    additional = rc_manifest.get("additional_frozen_identities") or {}
    if not isinstance(artifact_hashes, Mapping) or not isinstance(additional, Mapping):
        raise TypeError("Unexpected RC manifest packaging schema")
    observed_keys = set(artifact_hashes)
    return {
        "rehash_map_keys": sorted(observed_keys),
        "all_expected_packager_rehash_keys_present": PACKAGER_REHASHED_KEYS <= observed_keys,
        "missing_expected_packager_rehash_keys": sorted(PACKAGER_REHASHED_KEYS - observed_keys),
        "listed_additional_identities_not_rehashed_by_artifact_map": dict(additional),
        "interpretation": (
            "artifact_sha256 entries were the packager's direct local rehash set; "
            "additional_frozen_identities are provenance identities recorded separately."
        ),
    }


def repository_state(repo_root: Path) -> dict[str, Any]:
    def git(*args: str, allow_failure: bool = False) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 and not allow_failure:
            raise RuntimeError(
                f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return proc.stdout.strip()

    pyproject = repo_root / "pyproject.toml"
    package: dict[str, Any] = {}
    if pyproject.is_file() and tomllib is not None:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = parsed.get("project") or {}
        package = {
            "name": project.get("name"),
            "version": project.get("version"),
            "requires_python": project.get("requires-python"),
            "dependencies": project.get("dependencies"),
            "scripts": project.get("scripts"),
        }

    tests_root = repo_root / "tests"
    workflow_root = repo_root / ".github" / "workflows"
    return {
        "repo_root": str(repo_root),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "dirty_porcelain": git("status", "--porcelain", "--untracked-files=all"),
        "local_tags": [line for line in git("tag", "--list").splitlines() if line.strip()],
        "package_metadata": package,
        "tracked_test_files": sorted(
            str(path.relative_to(repo_root)) for path in tests_root.rglob("*") if path.is_file()
        )
        if tests_root.is_dir()
        else [],
        "ci_workflows": sorted(
            str(path.relative_to(repo_root)) for path in workflow_root.rglob("*") if path.is_file()
        )
        if workflow_root.is_dir()
        else [],
        "python_runtime": sys.version,
    }


def write_missing_inventory(
    output_dir: Path,
    repo: dict[str, Any],
    required: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "BLOCKED_MISSING_LOCAL_ARTIFACTS",
        "repository_state": repo,
        "required_artifacts": required,
        "instruction": (
            "Restore or point the audit at the listed frozen artifacts. No biological or numerical "
            "result was fabricated from missing inputs."
        ),
    }
    (output_dir / "missing_artifacts.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    deployment = model_root / "nasa_bps_53bp1" / "r1_v2_deployment"
    confirmation = model_root / "nasa_bps_53bp1" / "r1_v2_final_holdout_confirmation"
    bie_dir = deployment / "kneron_int8" / "bie_holdout_equivalence"
    physical_root = deployment / "kneron_nef" / "physical_holdout_hardware"
    biological = physical_root / "biological_equivalence"
    parser.add_argument("--confirmation-dir", default=str(confirmation))
    parser.add_argument("--bie-dir", default=str(bie_dir))
    parser.add_argument("--physical-root", default=str(physical_root))
    parser.add_argument("--biological-dir", default=str(biological))
    parser.add_argument(
        "--qc1-manifest",
        default=str(
            data_root
            / "nasa_bps_microscopy"
            / "metadata"
            / "r1_v2_freeze"
            / "r1_v2_final_holdout_manifest_blinded_qc1.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    confirmation_dir = Path(args.confirmation_dir).resolve()
    bie_dir = Path(args.bie_dir).resolve()
    physical_root = Path(args.physical_root).resolve()
    biological_dir = Path(args.biological_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    refuse_nonempty_output_dir(output_dir)
    repo_root = Path(__file__).resolve().parents[3]
    repo = repository_state(repo_root)

    fp32_freeze_path = confirmation_dir / "fp32_deployment_reference_freeze.json"
    bie_summary_path = bie_dir / "bie_holdout_equivalence_summary.json"
    bie_freeze_path = bie_dir / "bie_holdout_equivalence_freeze.json"
    physical_predictions_path = physical_root / "physical_holdout_nucleus_predictions.csv"
    physical_inference_summary_path = physical_root / "physical_holdout_inference_summary.json"
    physical_predictions_freeze_path = physical_root / "physical_holdout_predictions_freeze.json"
    final_summary_path = biological_dir / "physical_holdout_biological_equivalence_summary.json"
    final_freeze_path = biological_dir / "physical_holdout_biological_equivalence_freeze.json"
    rc_manifest_path = (
        biological_dir / "release_candidate" / "nasa_bps_r1_v2_kl720_rc1_manifest.json"
    )

    authoritative = {
        fp32_freeze_path: (FP32_REFERENCE_FREEZE_SHA256, "FP32 reference freeze"),
        bie_summary_path: (BIE_EQUIVALENCE_SUMMARY_SHA256, "BIE equivalence summary"),
        bie_freeze_path: (BIE_EQUIVALENCE_FREEZE_SHA256, "BIE equivalence freeze"),
        physical_predictions_path: (PHYSICAL_PREDICTIONS_SHA256, "physical predictions"),
        physical_inference_summary_path: (
            PHYSICAL_INFERENCE_SUMMARY_SHA256,
            "physical inference summary",
        ),
        physical_predictions_freeze_path: (
            PHYSICAL_PREDICTIONS_FREEZE_SHA256,
            "physical prediction freeze",
        ),
        final_summary_path: (FINAL_PHYSICAL_SUMMARY_SHA256, "final physical summary"),
        final_freeze_path: (FINAL_PHYSICAL_FREEZE_SHA256, "final physical freeze"),
        rc_manifest_path: (RC_MANIFEST_SHA256, "RC manifest"),
    }
    required_inventory = [
        {
            "label": label,
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": expected,
        }
        for path, (expected, label) in authoritative.items()
    ]
    missing_authoritative = [row for row in required_inventory if not row["exists"]]
    if missing_authoritative:
        write_missing_inventory(output_dir, repo, required_inventory)
        print("NASA R1 v2 post-release evidence audit: BLOCKED - missing local artifacts")
        for row in missing_authoritative:
            print(f"MISSING: {row['label']}: {row['path']}")
        print(f"actionable inventory: {output_dir / 'missing_artifacts.json'}")
        return 2
    checked_hashes = {
        label: check_sha(path, expected, label) for path, (expected, label) in authoritative.items()
    }

    fp32_freeze = read_json_object(fp32_freeze_path)
    bie_summary = read_json_object(bie_summary_path)
    final_summary = read_json_object(final_summary_path)
    rc_manifest = read_json_object(rc_manifest_path)

    fp32_reference_files = verify_transitive_files(fp32_freeze, "reference_files", confirmation_dir)
    bie_outputs = verify_transitive_files(bie_summary, "outputs", bie_dir)
    physical_outputs = verify_transitive_files(final_summary, "outputs", biological_dir)

    fp32_delta_path = confirmation_dir / "final_holdout_matched_radiation_deltas.csv"
    bie_delta_path = bie_dir / "bie_holdout_matched_radiation_deltas.csv"
    physical_delta_path = biological_dir / "physical_holdout_matched_radiation_deltas.csv"
    for path in (fp32_delta_path, bie_delta_path, physical_delta_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Historical schema note: the frozen FP32 delta CSV predates the BIE/physical
    # deployment tables and does not contain sham/exposed sample-name columns.
    # Reconstruct those identities from each stage's already-hash-verified bag
    # table, while checking any names present in the newer delta tables.
    fp32_bag_path = confirmation_dir / "final_holdout_bag_predictions.csv"
    bie_bag_path = bie_dir / "bie_holdout_bag_predictions.csv"
    physical_bag_path = biological_dir / "physical_holdout_bag_predictions.csv"
    expected_bag_hashes = (
        (fp32_bag_path, fp32_reference_files.get(fp32_bag_path.name), "FP32 bag predictions"),
        (bie_bag_path, bie_outputs.get(bie_bag_path.name), "BIE bag predictions"),
        (
            physical_bag_path,
            physical_outputs.get(physical_bag_path.name),
            "physical bag predictions",
        ),
    )
    for bag_path, expected_hash, bag_label in expected_bag_hashes:
        if not isinstance(expected_hash, str):
            raise TypeError(f"{bag_label} not covered by verified transitive hash mapping")
        check_sha(bag_path, expected_hash, bag_label)

    fp32_rows = enrich_contrast_sample_names(
        read_csv_rows(fp32_delta_path),
        read_csv_rows(fp32_bag_path),
        exposed_dose_by_branch=EXPOSED_DOSE,
        label="FP32",
    )
    bie_rows_for_join = enrich_contrast_sample_names(
        read_csv_rows(bie_delta_path),
        read_csv_rows(bie_bag_path),
        exposed_dose_by_branch=EXPOSED_DOSE,
        label="BIE",
    )
    physical_rows_for_join = enrich_contrast_sample_names(
        read_csv_rows(physical_delta_path),
        read_csv_rows(physical_bag_path),
        exposed_dose_by_branch=EXPOSED_DOSE,
        label="physical",
    )
    joined, sign_audit = join_contrasts(fp32_rows, bie_rows_for_join, physical_rows_for_join)
    if len(joined) != EXPECTED_CONTRASTS:
        raise RuntimeError(f"Expected {EXPECTED_CONTRASTS} matched contrasts, got {len(joined)}")
    peaks = peak_series_audit(joined)

    physical_rows = read_csv_rows(physical_predictions_path)
    bie_nucleus_path = bie_dir / "bie_holdout_nucleus_predictions.csv"
    if sha256_file(bie_nucleus_path) != bie_outputs["bie_holdout_nucleus_predictions.csv"]:
        raise RuntimeError("BIE nucleus predictions no longer match BIE summary")
    bie_rows = read_csv_rows(bie_nucleus_path)
    if len(physical_rows) != EXPECTED_NUCLEI or len(bie_rows) != EXPECTED_NUCLEI:
        raise RuntimeError("Expected 2058 frozen physical/BIE nucleus rows")

    descriptor_search_paths = [
        physical_inference_summary_path,
        physical_predictions_freeze_path,
        final_summary_path,
        final_freeze_path,
    ]
    descriptor = find_exact_output_quantization_metadata(descriptor_search_paths)
    quant = quantization_diagnostic(physical_rows, bie_rows, descriptor)

    coverage = rc_packaging_coverage(rc_manifest)
    portable_inventory = {
        "required_directly_verified_artifacts": sorted(coverage["rehash_map_keys"]),
        "additional_recorded_identities": sorted(
            (rc_manifest.get("additional_frozen_identities") or {}).keys()
        ),
        "environment_or_runtime_metadata_gaps": [
            "durable physical-device serial/identifier is not established by the RC artifact map",
            "physical hardware runner console may print firmware but the audited summary schema does not persist it",
            "host Python/NumPy/OpenCV/kp package versions are not all pinned in the RC artifact map",
            "the RC manifest is a provenance index, not a self-contained portable artifact archive",
        ],
    }

    legacy_flag = bool(final_summary.get("exact_fp32_direction_peak_signature_reproduced"))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_READ_ONLY_POSTRELEASE_EVIDENCE_AUDIT",
        "legacy_seven_gate_decision_changed": False,
        "repository_state": repo,
        "authoritative_hashes_checked": checked_hashes,
        "transitive_fp32_reference_files_checked": fp32_reference_files,
        "transitive_bie_outputs_checked": bie_outputs,
        "transitive_physical_biological_outputs_checked": physical_outputs,
        "contrast_audit": sign_audit,
        "legacy_exact_direction_peak_signature_flag": legacy_flag,
        "actual_per_contrast_sign_identity_fp32_vs_physical": sign_audit[
            "actual_fp32_vs_physical_per_contrast_sign_identity"
        ],
        "complete_peak_series": peaks,
        "quantization_diagnostic": quant,
        "rc_packaging_coverage": coverage,
        "portable_evidence_inventory": portable_inventory,
        "claim_boundary": (
            "This audit does not modify or re-evaluate the original seven-gate decision. "
            "Per-contrast sign identity is reported separately from legacy agreement-count signatures."
        ),
    }
    reject_nonfinite_json(report, label="postrelease audit report")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "nasa_r1_v2_per_contrast_audit.csv", joined)
    if peaks:
        write_csv_rows(output_dir / "nasa_r1_v2_peak_series_audit.csv", peaks)
    (output_dir / "nasa_r1_v2_postrelease_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_dir / "portable_evidence_inventory.json").write_text(
        json.dumps(portable_inventory, indent=2, allow_nan=False), encoding="utf-8"
    )

    print("NASA R1 v2 post-release evidence audit: COMPLETE")
    print(f"repository branch/head: {repo['branch']} / {repo['head']}")
    print(f"tracked/untracked dirty entries present: {bool(repo['dirty_porcelain'])}")
    print(f"local tags: {repo['local_tags']}")
    print(f"contrasts: {len(joined)}")
    print(f"legacy exact-direction/peak aggregate flag: {legacy_flag}")
    print(
        "actual FP32 vs physical per-contrast sign identity: "
        f"{sign_audit['actual_fp32_vs_physical_per_contrast_sign_identity']}"
    )
    print(
        "changed FP32-vs-physical contrast identities: "
        f"{len(sign_audit['fp32_vs_physical_changed_contrasts'])}"
    )
    print(f"complete 4/24/48 peak series audited: {len(peaks)}")
    print(f"exact persisted output quantization metadata found: {descriptor is not None}")
    print("original seven-gate decision changed by audit: NO")
    print(f"audit output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
