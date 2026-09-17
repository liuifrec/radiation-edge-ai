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
)
from radiation_edge_ai.execution import execute_run_plan, verify_target
from radiation_edge_ai.measurement import (
    ASSAY_ID as MEASUREMENT_ASSAY_ID,
)
from radiation_edge_ai.measurement import (
    create_measurement_plan,
    execute_measurement_plan,
)
from radiation_edge_ai.nasa_endpoint import (
    ASSAY_ID as NASA_ENDPOINT_ASSAY_ID,
)
from radiation_edge_ai.nasa_endpoint import (
    create_nasa_endpoint_report,
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

    run = subparsers.add_parser(
        "run",
        help="execute a frozen run plan with its approved backend",
    )
    run.add_argument("plan_path", type=Path)
    run.add_argument("--onnx-python", type=Path)
    run.add_argument("--kl720-python", type=Path)
    run.add_argument("--kl720-port", type=int)
    run.add_argument("--kl720-timeout-ms", type=int, default=10000)

    measure_plan = subparsers.add_parser(
        "measure-plan",
        help="create a content-addressed assay measurement plan",
    )
    measure_plan.add_argument("--assay", required=True)
    measure_plan.add_argument("--predictions", required=True, type=Path)
    measure_plan.add_argument("--burden-column", required=True)
    measure_plan.add_argument("--output-dir", required=True, type=Path)

    measure = subparsers.add_parser(
        "measure",
        help="execute a verified assay measurement plan",
    )
    measure.add_argument("plan_path", type=Path)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="reconstruct an approved assay endpoint from frozen predictions",
    )
    aggregate.add_argument("--assay", required=True)
    aggregate.add_argument("--predictions", required=True, type=Path)
    aggregate.add_argument("--burden-column", required=True)
    aggregate.add_argument("--output-dir", required=True, type=Path)

    verify = subparsers.add_parser(
        "verify",
        help="verify a run plan, run record, measurement plan, or endpoint report",
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

        if args.command == "run":
            record_path = execute_run_plan(
                args.plan_path,
                onnx_python=args.onnx_python,
                kl720_python=args.kl720_python,
                kl720_port=args.kl720_port,
                kl720_timeout_ms=args.kl720_timeout_ms,
            )
            print(record_path)
            return 0

        if args.command == "measure-plan":
            if args.assay != MEASUREMENT_ASSAY_ID:
                raise ControlPlaneError(
                    "Measurement planning is not implemented for assay "
                    f"{args.assay!r}"
                )
            plan_path = create_measurement_plan(
                predictions_path=args.predictions,
                burden_column=args.burden_column,
                output_dir=args.output_dir,
            )
            print(plan_path)
            return 0

        if args.command == "measure":
            record_path = execute_measurement_plan(
                args.plan_path,
            )
            print(record_path)
            return 0

        if args.command == "aggregate":
            if args.assay != NASA_ENDPOINT_ASSAY_ID:
                raise ControlPlaneError(
                    "Endpoint aggregation is not implemented for assay "
                    f"{args.assay!r}"
                )
            report_path = create_nasa_endpoint_report(
                predictions_path=args.predictions,
                burden_column=args.burden_column,
                output_dir=args.output_dir,
            )
            print(report_path)
            return 0

        if args.command == "verify":
            report = verify_target(
                args.plan_path,
                check_artifacts=not args.no_artifacts,
            )
            if args.json:
                _dump_json(report)
            else:
                print(f"kind: {report['kind']}")
                if report["kind"] in {"run_plan", "run_record"}:
                    print(f"run_id: {report['run_id']}")
                elif report["kind"] in {
                    "measurement_plan",
                    "measurement_record",
                }:
                    print(f"measurement_id: {report['measurement_id']}")
                elif report["kind"] == "nasa_endpoint_report":
                    print(f"aggregation_id: {report['aggregation_id']}")
                if report["kind"] == "run_plan":
                    status = "PASS" if report["fingerprint_ok"] else "FAIL"
                    print(f"fingerprint: {status}")
                    if report["artifact_check_performed"]:
                        status = "PASS" if report["artifacts_ok"] else "FAIL"
                        print(f"artifacts: {status}")
                elif report["kind"] == "run_record":
                    print(f"plan: {'PASS' if report['plan_ok'] else 'FAIL'}")
                    print(f"raw output: {'PASS' if report['raw_output_ok'] else 'FAIL'}")
                    print(
                        "worker manifest: "
                        f"{'PASS' if report['worker_manifest_ok'] else 'FAIL'}"
                    )
                elif report["kind"] == "measurement_plan":
                    print(
                        "fingerprint: "
                        f"{'PASS' if report['fingerprint_ok'] else 'FAIL'}"
                    )
                    if report["source_artifact_check_performed"]:
                        print(
                            "source predictions: "
                            f"{'PASS' if report['source_predictions_ok'] else 'FAIL'}"
                        )
                elif report["kind"] == "measurement_record":
                    print(
                        "fingerprint: "
                        f"{'PASS' if report['fingerprint_ok'] else 'FAIL'}"
                    )
                    print(
                        "measurement plan: "
                        f"{'PASS' if report['plan_ok'] else 'FAIL'}"
                    )
                    print(
                        "endpoint report: "
                        f"{'PASS' if report['endpoint_report_ok'] else 'FAIL'}"
                    )
                    print(
                        "sample aggregates: "
                        f"{'PASS' if report['sample_aggregates_ok'] else 'FAIL'}"
                    )
                    print(
                        "cross-artifact bindings: "
                        f"{'PASS' if report['cross_artifact_bindings_ok'] else 'FAIL'}"
                    )
                    print(
                        f"measurement counts: {report['n_nuclei']} nuclei / "
                        f"{report['n_samples']} samples"
                    )
                elif report["kind"] == "nasa_endpoint_report":
                    print(
                        "fingerprint: "
                        f"{'PASS' if report['fingerprint_ok'] else 'FAIL'}"
                    )
                    if report["source_artifact_check_performed"]:
                        print(
                            "source predictions: "
                            f"{'PASS' if report['source_predictions_ok'] else 'FAIL'}"
                        )
                    print(
                        "sample aggregates: "
                        f"{'PASS' if report['sample_aggregates_ok'] else 'FAIL'}"
                    )
                    print(
                        f"endpoint counts: {report['n_nuclei']} nuclei / "
                        f"{report['n_samples']} samples"
                    )
                else:
                    raise ControlPlaneError(
                        f"Unknown verification report kind: {report['kind']!r}"
                    )
                print(f"verification: {'PASS' if report['ok'] else 'FAIL'}")
            return 0 if report["ok"] else 3

    except ControlPlaneError as exc:
        print(f"radedge: ERROR: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
