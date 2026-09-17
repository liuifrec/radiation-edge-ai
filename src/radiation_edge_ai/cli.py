"""Command-line interface for the Radiation Edge AI offline control plane."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from radiation_edge_ai.control import (
    ControlPlaneError,
    collect_doctor_report,
    create_run_plan,
    get_assay,
    list_assays,
    verify_run_plan,
)


def _dump_json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def _print_doctor(report: dict[str, object]) -> None:
    runtime = report["runtime"]
    storage = report["storage"]
    optional = report["optional_runtime"]
    assays = report["assays"]
    assert isinstance(runtime, dict)
    assert isinstance(storage, dict)
    assert isinstance(optional, dict)
    assert isinstance(assays, dict)

    print("Radiation Edge AI - offline doctor")
    print(f"Python: {runtime['python_executable']}")
    print(f"Version: {runtime['python_version']}")
    for name in ("data", "models", "cache"):
        item = storage[name]
        assert isinstance(item, dict)
        state = "OK" if item["exists"] and item["is_dir"] else "MISSING"
        print(f"{name:>6}: {state}  {item['path']}")

    print(f"ONNX Runtime importable: {optional['onnxruntime_importable']}")
    kl720 = optional["kl720_python"]
    assert isinstance(kl720, dict)
    print(f"KL720 Python present:    {kl720['exists']}  {kl720['path']}")
    print("KL720 hardware scan:     NOT PERFORMED")
    print("Assay asset probes:")
    for assay_id, item in assays.items():
        assert isinstance(item, dict)
        data = item["data"]
        models = item["models"]
        assert isinstance(data, dict)
        assert isinstance(models, dict)
        print(
            f"  {assay_id}: data={'FOUND' if data['found'] else 'not found'}, "
            f"models={'FOUND' if models['found'] else 'not found'}"
        )


def _print_assays(assay_id: Optional[str]) -> None:  # noqa: UP045
    specs = (get_assay(assay_id),) if assay_id else list_assays()
    for index, spec in enumerate(specs):
        if index:
            print()
        print(spec.assay_id)
        print(f"  title: {spec.title}")
        print(f"  endpoint: {spec.endpoint}")
        print(f"  input: {spec.input_kind}")
        print(f"  backends: {', '.join(spec.approved_backends)}")
        print(f"  required metadata: {', '.join(spec.required_metadata)}")
        print(f"  guard: {spec.interpretation_guard}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="radedge",
        description="Deterministic offline control plane for Radiation Edge AI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser(
        "doctor",
        help="inspect local runtime and asset availability",
    )
    doctor.add_argument("--json", action="store_true")

    assays = subparsers.add_parser(
        "assays",
        help="list frozen assay specifications",
    )
    assays.add_argument("assay_id", nargs="?")
    assays.add_argument("--json", action="store_true")

    plan = subparsers.add_parser(
        "plan",
        help="create an immutable content-addressed run plan",
    )
    plan.add_argument("--assay", required=True)
    plan.add_argument("--backend", required=True)
    plan.add_argument("--input", required=True, type=Path)
    plan.add_argument("--model", required=True, type=Path)
    plan.add_argument("--metadata", required=True, type=Path)
    plan.add_argument("--config", type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)

    verify = subparsers.add_parser(
        "verify",
        help="verify a run-plan fingerprint and artifacts",
    )
    verify.add_argument("plan_path", type=Path)
    verify.add_argument("--no-artifacts", action="store_true")
    verify.add_argument("--json", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = collect_doctor_report()
            _dump_json(report) if args.json else _print_doctor(report)
            return 0

        if args.command == "assays":
            specs = (get_assay(args.assay_id),) if args.assay_id else list_assays()
            if args.json:
                _dump_json([spec.to_dict() for spec in specs])
            else:
                _print_assays(args.assay_id)
            return 0

        if args.command == "plan":
            plan_path = create_run_plan(
                assay_id=args.assay,
                backend=args.backend,
                input_path=args.input,
                model_path=args.model,
                metadata_path=args.metadata,
                config_path=args.config,
                output_dir=args.output_dir,
            )
            print(plan_path)
            return 0

        if args.command == "verify":
            report = verify_run_plan(
                args.plan_path,
                check_artifacts=not args.no_artifacts,
            )
            if args.json:
                _dump_json(report)
            else:
                print(f"run_id: {report['run_id']}")
                status = "PASS" if report["fingerprint_ok"] else "FAIL"
                print(f"fingerprint: {status}")
                if report["artifact_check_performed"]:
                    status = "PASS" if report["artifacts_ok"] else "FAIL"
                    print(f"artifacts: {status}")
                print(f"verification: {'PASS' if report['ok'] else 'FAIL'}")
            return 0 if report["ok"] else 3

    except ControlPlaneError as exc:
        print(f"radedge: ERROR: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
