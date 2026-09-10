"""Deterministic phenotype-blind geometry repair for NASA BPS R1 v2 final holdout.

This gate is used only after validate_r1_v2_final_holdout_blinded.py reports
geometry-only hard-QC failures and zero non-geometry failures. It never reads
NASA phenotype/reference outcomes.

For each failed frozen slot, the script reconstructs the original eligible
Source Name x condition candidate pool using the already frozen holdout ranking
seed, then selects the first previously unselected candidate that passes the
identical development hard QC. Network/materialization failure is not treated as
biological/QC failure and aborts the run rather than causing a candidate skip.

If and only if the repaired 2,058-nucleus manifest passes the full identical hard
QC, the repaired QC1 manifest is serialized and its SHA256 becomes the final
holdout image identity to pin before the one-time phenotype evaluator is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

FROZEN_REPAIRED_CHECKPOINT_SHA256 = (
    "2b07c67d6e50e7a36e26e0052feebc826b518e6efae863ba312ba9761a4eb3d5"
)
FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256 = (
    "40f015f7f8a8acb71289c1fe91218253e0b518c7e98f920511a5792dd5b6862d"
)
EXPECTED_NUCLEI = 2058
FORBIDDEN_COLUMN_TOKENS = (
    "nfoci",
    "phenotype",
    "reference",
    "ground_truth",
    "prediction",
    "predicted",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reject_outcome_columns(rows: list[dict[str, str]], label: str) -> None:
    bad = sorted(
        column
        for column in rows[0]
        if any(token in column.lower() for token in FORBIDDEN_COLUMN_TOKENS)
    )
    if bad:
        raise RuntimeError(f"{label} contains forbidden outcome/reference columns: {bad}")


def is_geometry_only_failure(row: dict[str, str]) -> bool:
    error = str(row.get("error", ""))
    return (
        str(row.get("ok", "0")).strip() not in {"1", "1.0", "True", "true"}
        and "native geometry" in error
        and ";" not in error
    )


def manifest_condition(row: dict[str, str], freeze: Any) -> tuple[str, str, str]:
    return (
        str(row["particle_type"]).strip(),
        freeze.normalize_number(row["dose_Gy"]),
        freeze.normalize_hour(row["hr_post_exposure"]),
    )


def build_replacement_row(
    frozen_row: dict[str, str],
    candidate: dict[str, str],
    rank_sha256: str,
    freeze: Any,
) -> dict[str, str]:
    replacement = dict(frozen_row)
    parsed = freeze.parse_filename(candidate["filename"])

    if candidate["osd_source_name"] != frozen_row["source_name"]:
        raise RuntimeError("Replacement candidate changed Source Name")
    if candidate["osd_sample_name"] != frozen_row["sample_name"]:
        raise RuntimeError("Replacement candidate changed Sample Name")
    if manifest_condition(frozen_row, freeze) != freeze.condition_key(candidate):
        raise RuntimeError("Replacement candidate changed radiation condition")

    replacement.update(
        {
            "nucleus_key": candidate["nucleus_key"],
            "plate": parsed["plate"],
            "acquisition": parsed["acquisition"],
            "well": parsed["well"],
            "fitc_filename": candidate["filename"],
            "dapi_filename": candidate["dapi_filename"],
            "mask_filename": candidate["mask_filename"],
            "selection_rank_sha256": rank_sha256,
            "v1_member": "0",
            "v1_sample_id": "",
        }
    )
    return replacement


def materialize_candidate(
    row: dict[str, str],
    image_root: Path,
    materialize: Any,
    *,
    retries: int,
    timeout: int,
) -> list[dict[str, Any]]:
    specs = (
        ("fitc", row["fitc_filename"], materialize.FITC_BASE_URL),
        ("dapi", row["dapi_filename"], materialize.DAPI_MASK_BASE_URL),
        ("mask", row["mask_filename"], materialize.DAPI_MASK_BASE_URL),
    )
    results: list[dict[str, Any]] = []
    for channel, filename, base in specs:
        task = materialize.DownloadTask(
            sample_id=row["sample_id"],
            nucleus_key=row["nucleus_key"],
            source_name=row["source_name"],
            sample_name=row["sample_name"],
            channel=channel,
            filename=filename,
            url=materialize.object_url(base, filename),
            target_path=image_root / channel / filename,
        )
        result = materialize.run_task(
            task,
            retries=max(1, retries),
            timeout=max(1, timeout),
            overwrite=False,
        )
        results.append(result)
        if result["status"] == "failed":
            raise RuntimeError(
                "Candidate materialization failed; aborting rather than skipping a ranked candidate: "
                f"{channel} {filename}: {result.get('error', '')}"
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    data_root = Path(os.environ.get("RADEDGE_DATA_ROOT", r"D:\radiation-edge-ai-data\data"))
    model_root = Path(os.environ.get("RADEDGE_MODEL_ROOT", r"D:\radiation-edge-ai-data\models"))
    metadata_root = data_root / "nasa_bps_microscopy" / "metadata"
    holdout_root = data_root / "nasa_bps_microscopy" / "r1_v2_final_holdout_blinded"
    script_dir = Path(__file__).resolve().parent

    parser.add_argument(
        "--manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_final_holdout_manifest_blinded.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            model_root
            / "nasa_bps_53bp1"
            / "r1_v2_final_candidate"
            / "r1_countnet_v2_final_bn_recalibrated.pt"
        ),
    )
    parser.add_argument(
        "--qc-table",
        default=str(holdout_root / "qc" / "r1_v2_final_holdout_triplet_qc.csv"),
    )
    parser.add_argument(
        "--qc-summary",
        default=str(holdout_root / "qc" / "r1_v2_final_holdout_qc_summary.json"),
    )
    parser.add_argument(
        "--crosswalk",
        default=str(metadata_root / "fitc_53bp1_osd366_crosswalk.csv"),
    )
    parser.add_argument(
        "--dapi-mask-meta",
        default=str(metadata_root / "meta_DAPI_MASK.csv"),
    )
    parser.add_argument("--image-root", default=str(holdout_root / "images"))
    parser.add_argument(
        "--output-manifest",
        default=str(metadata_root / "r1_v2_freeze" / "r1_v2_final_holdout_manifest_blinded_qc1.csv"),
    )
    parser.add_argument("--output-dir", default=str(holdout_root / "qc"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    qc_table_path = Path(args.qc_table).resolve()
    qc_summary_path = Path(args.qc_summary).resolve()
    crosswalk_path = Path(args.crosswalk).resolve()
    dapi_path = Path(args.dapi_mask_meta).resolve()
    image_root = Path(args.image_root).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        manifest_path,
        checkpoint_path,
        qc_table_path,
        qc_summary_path,
        crosswalk_path,
        dapi_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    if checkpoint_sha != FROZEN_REPAIRED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Repaired checkpoint SHA256 mismatch: {checkpoint_sha}")
    if manifest_sha != FROZEN_BLINDED_HOLDOUT_MANIFEST_SHA256:
        raise RuntimeError(f"Frozen blinded holdout manifest SHA256 mismatch: {manifest_sha}")

    frozen_rows = read_csv(manifest_path)
    manifest_fields = list(frozen_rows[0])
    reject_outcome_columns(frozen_rows, "Frozen holdout manifest")
    if len(frozen_rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"Holdout nuclei={len(frozen_rows)}; expected {EXPECTED_NUCLEI}")

    qc_rows = read_csv(qc_table_path)
    reject_outcome_columns(qc_rows, "Blinded QC table")
    if len(qc_rows) != EXPECTED_NUCLEI:
        raise RuntimeError(f"QC rows={len(qc_rows)}; expected {EXPECTED_NUCLEI}")

    with qc_summary_path.open("r", encoding="utf-8") as handle:
        prior_summary = json.load(handle)
    if int(prior_summary.get("non_geometry_failures", -1)) != 0:
        raise RuntimeError("Prior blinded QC reported a non-geometry failure; repair is forbidden")

    frozen_by_id = {row["sample_id"]: row for row in frozen_rows}
    qc_by_id = {row["sample_id"]: row for row in qc_rows}
    if len(frozen_by_id) != EXPECTED_NUCLEI or len(qc_by_id) != EXPECTED_NUCLEI:
        raise RuntimeError("Duplicate sample_id detected in manifest or QC table")
    if set(frozen_by_id) != set(qc_by_id):
        raise RuntimeError("QC table sample IDs do not match frozen holdout manifest")
    for sample_id, qc_row in qc_by_id.items():
        if qc_row["nucleus_key"] != frozen_by_id[sample_id]["nucleus_key"]:
            raise RuntimeError(f"QC/manifest nucleus mismatch for {sample_id}")

    failures = [row for row in qc_rows if str(row.get("ok", "0")) not in {"1", "1.0"}]
    geometry_failures = [row for row in failures if is_geometry_only_failure(row)]
    non_geometry = [row for row in failures if row not in geometry_failures]
    if non_geometry:
        raise RuntimeError(f"Non-geometry hard failures present; confirmatory repair forbidden: {non_geometry[:3]}")
    if not geometry_failures:
        raise RuntimeError("No geometry-only failure found; use the ordinary QC1 manifest path instead")

    freeze = load_module("nasa_bps_freeze_for_holdout_repair", script_dir / "freeze_r1_v2_manifests.py")
    dev_qc = load_module("nasa_bps_dev_qc_for_holdout_repair", script_dir / "validate_r1_v2_development.py")
    final_qc = load_module(
        "nasa_bps_final_qc_for_holdout_repair", script_dir / "validate_r1_v2_final_holdout_blinded.py"
    )
    materialize = load_module(
        "nasa_bps_materialize_for_holdout_repair", script_dir / "materialize_r1_v2_final_holdout_blinded.py"
    )

    final_qc.validate_blinded_structure(frozen_rows)
    crosswalk = read_csv(crosswalk_path)
    dapi_meta = read_csv(dapi_path)
    reject_outcome_columns(crosswalk, "FITC/OSD crosswalk")
    eligible, missing_pairs = freeze.enrich_candidates(crosswalk, dapi_meta)
    freeze.validate_single_sample_per_cell(eligible)

    repaired_rows = [dict(row) for row in frozen_rows]
    row_index = {row["sample_id"]: i for i, row in enumerate(repaired_rows)}
    selected_keys = {row["nucleus_key"] for row in repaired_rows}
    candidate_audit: list[dict[str, Any]] = []
    replacement_audit: list[dict[str, Any]] = []

    print("Radiation Edge AI - NASA BPS R1 v2 FINAL HOLDOUT geometry repair")
    print("PHENOTYPE-BLIND DETERMINISTIC REPLACEMENT ONLY")
    print(f"repaired checkpoint SHA256 pinned: {checkpoint_sha}")
    print(f"frozen blinded holdout manifest SHA256: {manifest_sha}")
    print(f"geometry-only failed slots: {len(geometry_failures)}")
    print(f"non-geometry failures: {len(non_geometry)}")
    print(f"eligible metadata missing DAPI/MASK pairs: {missing_pairs}")
    print("NASA phenotype/reference tables read: NO")
    print("")

    cv2.setNumThreads(0)
    for failed in sorted(geometry_failures, key=lambda row: row["sample_id"]):
        frozen = frozen_by_id[failed["sample_id"]]
        source = frozen["source_name"]
        cond = manifest_condition(frozen, freeze)
        pool = eligible.get((source, cond), [])
        ranked = sorted(
            pool,
            key=lambda row: freeze.deterministic_rank(freeze.R1_V2_HOLDOUT_SEED, row["nucleus_key"]),
        )
        if not ranked:
            raise RuntimeError(f"No eligible candidates for {source} {cond}")

        replacement: dict[str, str] | None = None
        attempts = 0
        for candidate in ranked:
            key = candidate["nucleus_key"]
            if key in selected_keys:
                continue
            attempts += 1
            rank_sha = freeze.deterministic_rank(freeze.R1_V2_HOLDOUT_SEED, key)
            proposal = build_replacement_row(frozen, candidate, rank_sha, freeze)
            transfer = materialize_candidate(
                proposal,
                image_root,
                materialize,
                retries=args.retries,
                timeout=args.timeout,
            )
            inspection = dev_qc.inspect_one(proposal, image_root)
            candidate_audit.append(
                {
                    "failed_sample_id": frozen["sample_id"],
                    "failed_nucleus_key": frozen["nucleus_key"],
                    "source_name": source,
                    "particle_type": frozen["particle_type"],
                    "dose_Gy": frozen["dose_Gy"],
                    "hr_post_exposure": frozen["hr_post_exposure"],
                    "candidate_attempt": attempts,
                    "candidate_nucleus_key": key,
                    "candidate_rank_sha256": rank_sha,
                    "candidate_qc_ok": inspection.get("ok", 0),
                    "candidate_qc_error": inspection.get("error", ""),
                    "fitc_transfer": transfer[0]["status"],
                    "dapi_transfer": transfer[1]["status"],
                    "mask_transfer": transfer[2]["status"],
                }
            )
            if int(inspection.get("ok", 0)) == 1:
                replacement = proposal
                selected_keys.add(key)
                replacement_audit.append(
                    {
                        "sample_id": frozen["sample_id"],
                        "old_nucleus_key": frozen["nucleus_key"],
                        "old_fitc_filename": frozen["fitc_filename"],
                        "old_height": failed.get("height", ""),
                        "old_width": failed.get("width", ""),
                        "new_nucleus_key": proposal["nucleus_key"],
                        "new_fitc_filename": proposal["fitc_filename"],
                        "new_dapi_filename": proposal["dapi_filename"],
                        "new_mask_filename": proposal["mask_filename"],
                        "new_selection_rank_sha256": rank_sha,
                        "candidate_attempt": attempts,
                    }
                )
                break

        if replacement is None:
            raise RuntimeError(
                f"No phenotype-blind QC-passing unused candidate found for {frozen['sample_id']} {source} {cond}"
            )
        repaired_rows[row_index[frozen["sample_id"]]] = replacement
        print(
            f"REPLACED {frozen['sample_id']} {frozen['nucleus_key']} -> "
            f"{replacement['nucleus_key']} after {attempts} unused candidate(s)"
        )

    # Structural invariants must be unchanged after nucleus-level replacement.
    final_qc.validate_blinded_structure(repaired_rows)
    if Counter(row["sample_id"] for row in repaired_rows) != Counter(row["sample_id"] for row in frozen_rows):
        raise RuntimeError("sample_id structure changed during geometry repair")

    # Full hard-QC rerun over the entire repaired holdout before freezing a new identity.
    full_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(dev_qc.inspect_one, row, image_root) for row in repaired_rows]
        for i, future in enumerate(as_completed(futures), start=1):
            full_results.append(future.result())
            if i == 1 or i % 250 == 0 or i == len(futures):
                nfail = sum(int(row.get("ok", 0)) != 1 for row in full_results)
                print(f"[{i:>4}/{len(futures)}] repaired-manifest QC failures={nfail}")
    full_results.sort(key=lambda row: str(row["sample_id"]))
    final_failures = [row for row in full_results if int(row.get("ok", 0)) != 1]

    candidate_audit_path = output_dir / "r1_v2_final_holdout_geometry_candidate_audit.csv"
    replacement_audit_path = output_dir / "r1_v2_final_holdout_geometry_replacements.csv"
    final_qc_path = output_dir / "r1_v2_final_holdout_triplet_qc_after_geometry_repair.csv"
    write_csv(candidate_audit_path, candidate_audit)
    write_csv(replacement_audit_path, replacement_audit)
    write_csv(final_qc_path, full_results)

    if final_failures:
        if output_manifest.exists():
            output_manifest.unlink(missing_ok=True)
        raise RuntimeError(
            f"Repaired holdout still has {len(final_failures)} hard-QC failures; no QC1 manifest frozen"
        )

    write_csv(output_manifest, repaired_rows, fieldnames=manifest_fields)
    repaired_manifest_sha = sha256_file(output_manifest)
    if repaired_manifest_sha == manifest_sha:
        raise RuntimeError("Geometry replacement occurred but repaired manifest SHA256 did not change")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phenotype_status": "FINAL_BLINDED_DO_NOT_JOIN_PHENOTYPES",
        "nasa_phenotype_reference_tables_read": False,
        "repaired_checkpoint_sha256": checkpoint_sha,
        "parent_blinded_holdout_manifest_sha256": manifest_sha,
        "qc1_repaired_holdout_manifest": str(output_manifest),
        "qc1_repaired_holdout_manifest_sha256": repaired_manifest_sha,
        "nuclei": len(repaired_rows),
        "geometry_replacements": len(replacement_audit),
        "non_geometry_failures_before_repair": 0,
        "hard_failures_after_repair": 0,
        "candidate_attempts": len(candidate_audit),
        "candidate_audit": str(candidate_audit_path),
        "replacement_audit": str(replacement_audit_path),
        "full_repaired_qc_table": str(final_qc_path),
        "hard_qc_pass": True,
    }
    summary_path = output_dir / "r1_v2_final_holdout_geometry_repair_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("")
    print("[Phenotype-blind geometry-repair result]")
    print(f"geometry replacements: {len(replacement_audit)}")
    print(f"candidate attempts: {len(candidate_audit)}")
    print(f"repaired holdout nuclei passing hard QC: {len(repaired_rows)}/{len(repaired_rows)}")
    print(f"QC1 repaired holdout manifest: {output_manifest}")
    print(f"QC1 repaired holdout manifest SHA256: {repaired_manifest_sha}")
    print(f"replacement audit: {replacement_audit_path}")
    print(f"summary: {summary_path}")
    print("NASA phenotype/reference tables read: NO")
    print("FINAL HOLDOUT PHENOTYPE STATUS: BLINDED")
    print("NASA BPS R1 V2 FINAL HOLDOUT GEOMETRY REPAIR COMPLETE: YES")
    print("NEXT GATE: pin repaired manifest SHA256 and freeze the one-time phenotype evaluator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
