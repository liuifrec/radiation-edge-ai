"""Read-only, fail-closed verification across independently frozen artifacts.

The collection report is an ephemeral inspection result, not a frozen
scientific evidence record and not a biological-fidelity calculation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from radiation_edge_ai.control import ControlPlaneError, sha256_file
from radiation_edge_ai.execution import verify_target


def verify_collection(paths: Sequence[Path]) -> dict[str, Any]:
    """Verify all targets, preserving failures without running any inference.

    A duplicate normalized target is a command error, not another observation.
    Each source JSON is hashed before and after verification to detect any
    mid-audit change to the record under inspection.
    """
    if not paths:
        raise ControlPlaneError("At least one verification target is required")

    normalized: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        target = Path(raw_path).expanduser().resolve()
        if target in seen:
            raise ControlPlaneError(f"Duplicate verification target: {target}")
        seen.add(target)
        normalized.append(target)

    items: list[dict[str, Any]] = []
    for target in normalized:
        source_sha256: Optional[str] = None
        try:
            source_sha256 = sha256_file(target)
            verification = verify_target(target, check_artifacts=True)
            if not isinstance(verification, dict):
                raise ControlPlaneError("Verifier returned a non-object report")
            kind = verification.get("kind")
            verdict = verification.get("ok")
            if not isinstance(kind, str) or not kind.strip():
                raise ControlPlaneError("Verifier returned no valid artifact kind")
            if type(verdict) is not bool:
                raise ControlPlaneError("Verifier returned a non-boolean ok verdict")
            if sha256_file(target) != source_sha256:
                raise ControlPlaneError("Target bytes changed during verification")
            items.append(
                {
                    "path": str(target),
                    "source_sha256": source_sha256,
                    "kind": kind,
                    "ok": verdict,
                    "verification": verification,
                }
            )
        except (ControlPlaneError, OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            items.append(
                {
                    "path": str(target),
                    "source_sha256": source_sha256,
                    "kind": "unverified",
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    n_pass = sum(item["ok"] is True for item in items)
    n_total = len(items)
    return {
        "schema_version": 1,
        "report_type": "verification_collection_report",
        "artifact_check_performed": True,
        "n_targets": n_total,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
        "items": items,
        "ok": n_pass == n_total,
    }
