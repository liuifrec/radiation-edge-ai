"""Execute one frozen DNAi deployment window through KL720.

This module composes the generic Radiation Edge AI run-plan/execution layer.
It performs no field stitching, fiber reconstruction, annotation reading, or
biological-fidelity evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from radiation_edge_ai.control import (
    ControlPlaneError,
    create_run_plan,
    load_json_object,
    sha256_file,
)
from radiation_edge_ai.dna_fiber.field_record_v2 import (
    EXPECTED_NEF_SHA256,
)
from radiation_edge_ai.execution import (
    execute_run_plan,
    verify_run_record,
)

ASSAY_ID = "dnai-fiber-v3"
EXPECTED_TENSOR = (1, 3, 512, 512)


def _artifact_record(
    path: Path,
) -> dict[str, object]:
    resolved = (
        path
        .expanduser()
        .resolve()
    )

    return {
        "path": str(resolved),
        "sha256": sha256_file(
            resolved
        ),
        "size_bytes": (
            resolved.stat().st_size
        ),
    }


def _write_metadata(
    *,
    output_dir: Path,
    sample_id: str,
    pixel_size_um: float,
    channel_semantics: str,
    image_index: int,
    window_index: int,
) -> Path:
    metadata = {
        "sample_id": sample_id,
        "pixel_size_um": (
            pixel_size_um
        ),
        "channel_semantics": (
            channel_semantics
        ),
        "image_index": (
            image_index
        ),
        "window_index": (
            window_index
        ),
        "execution_role": (
            "DNAi frozen deployment window"
        ),
    }

    root = (
        output_dir
        .expanduser()
        .resolve()
        / "metadata"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        root
        / (
            f"image_{image_index:04d}_"
            f"window_{window_index:02d}.json"
        )
    )

    if path.exists():
        existing = load_json_object(
            path
        )

        if existing != metadata:
            raise ControlPlaneError(
                "Conflicting existing DNAi KL720 "
                f"window metadata: {path}"
            )

        return path

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")

    return path


def execute_verified_kl720_window(
    *,
    tensor_path: Path,
    model_path: Path,
    sample_id: str,
    pixel_size_um: float,
    channel_semantics: str,
    image_index: int,
    window_index: int,
    y: int,
    x: int,
    output_dir: Path,
    kl720_python: Optional[Path] = None,  # noqa: UP045
    kl720_port: Optional[int] = None,  # noqa: UP045
    kl720_timeout_ms: int = 10000,
) -> dict[str, object]:
    """Execute and verify one frozen physical DNAi window."""

    tensor = (
        tensor_path
        .expanduser()
        .resolve()
    )

    model = (
        model_path
        .expanduser()
        .resolve()
    )

    if not tensor.is_file():
        raise ControlPlaneError(
            f"DNAi window tensor missing: {tensor}"
        )

    if tensor.suffix.lower() != ".npy":
        raise ControlPlaneError(
            "DNAi KL720 window input must be .npy"
        )

    if not model.is_file():
        raise ControlPlaneError(
            f"DNAi KL720 model missing: {model}"
        )

    if model.suffix.lower() != ".nef":
        raise ControlPlaneError(
            "DNAi KL720 model must be .nef"
        )

    if (
        sha256_file(model)
        != EXPECTED_NEF_SHA256
    ):
        raise ControlPlaneError(
            "DNAi KL720 model is not the frozen NEF"
        )

    if (
        image_index < 0
        or window_index < 0
        or y < 0
        or x < 0
    ):
        raise ControlPlaneError(
            "DNAi window indices/coordinates "
            "must be non-negative"
        )

    if (
        not isinstance(sample_id, str)
        or not sample_id.strip()
    ):
        raise ControlPlaneError(
            "DNAi sample_id is empty"
        )

    if (
        not isinstance(
            channel_semantics,
            str,
        )
        or not channel_semantics.strip()
    ):
        raise ControlPlaneError(
            "DNAi channel semantics are empty"
        )

    if (
        not math.isfinite(
            pixel_size_um
        )
        or pixel_size_um <= 0.0
    ):
        raise ControlPlaneError(
            "DNAi pixel size must be positive"
        )

    if kl720_timeout_ms <= 0:
        raise ControlPlaneError(
            "KL720 timeout must be positive"
        )

    array = np.load(
        tensor,
        allow_pickle=False,
    )

    if (
        not isinstance(array, np.ndarray)
        or array.shape
        != EXPECTED_TENSOR
        or array.dtype
        != np.float32
        or not np.isfinite(
            array
        ).all()
    ):
        raise ControlPlaneError(
            "DNAi frozen window must be finite "
            "float32 [1,3,512,512]"
        )

    root = (
        output_dir
        .expanduser()
        .resolve()
    )

    metadata_path = (
        _write_metadata(
            output_dir=root,
            sample_id=sample_id,
            pixel_size_um=(
                pixel_size_um
            ),
            channel_semantics=(
                channel_semantics
            ),
            image_index=image_index,
            window_index=(
                window_index
            ),
        )
    )

    plan_path = create_run_plan(
        assay_id=ASSAY_ID,
        backend="kl720",
        input_path=tensor,
        model_path=model,
        metadata_path=(
            metadata_path
        ),
        output_dir=(
            root / "runs"
        ),
    )

    record_path = execute_run_plan(
        plan_path,
        kl720_python=(
            kl720_python
        ),
        kl720_port=(
            kl720_port
        ),
        kl720_timeout_ms=(
            kl720_timeout_ms
        ),
    )

    report = verify_run_record(
        record_path,
        check_source_artifacts=True,
    )

    if not report.get("ok"):
        raise ControlPlaneError(
            "DNAi KL720 window run failed verification"
        )

    record = load_json_object(
        record_path
    )

    if (
        record.get("backend")
        != "kl720"
        or record.get("assay_id")
        != ASSAY_ID
    ):
        raise ControlPlaneError(
            "Verified DNAi KL720 run has "
            "unexpected backend/assay"
        )

    sources = record.get(
        "source_artifacts"
    )

    raw_output = record.get(
        "raw_output"
    )

    runtime = record.get(
        "runtime"
    )

    timing = record.get(
        "timing"
    )

    if not all(
        isinstance(value, dict)
        for value in (
            sources,
            raw_output,
            runtime,
            timing,
        )
    ):
        raise ControlPlaneError(
            "Verified KL720 run record is malformed"
        )

    assert isinstance(
        sources,
        dict,
    )
    assert isinstance(
        raw_output,
        dict,
    )
    assert isinstance(
        runtime,
        dict,
    )
    assert isinstance(
        timing,
        dict,
    )

    run_input = sources.get(
        "input"
    )

    run_model = sources.get(
        "model"
    )

    if (
        not isinstance(
            run_input,
            dict,
        )
        or not isinstance(
            run_model,
            dict,
        )
    ):
        raise ControlPlaneError(
            "KL720 run source artifacts are malformed"
        )

    if (
        run_input.get("sha256")
        != sha256_file(tensor)
        or run_model.get("sha256")
        != EXPECTED_NEF_SHA256
    ):
        raise ControlPlaneError(
            "KL720 run source binding mismatch"
        )

    if (
        raw_output.get("dtype")
        != "float32"
        or raw_output.get("shape")
        != [1, 3, 512, 512]
    ):
        raise ControlPlaneError(
            "KL720 DNAi output tensor contract changed"
        )

    port = runtime.get(
        "selected_usb_port"
    )

    product_id = runtime.get(
        "product_id"
    )

    if (
        not isinstance(port, int)
        or port < 0
        or product_id != 0x720
    ):
        raise ControlPlaneError(
            "KL720 device identity is invalid"
        )

    inference_seconds = timing.get(
        "inference_send_receive_seconds"
    )

    if (
        isinstance(
            inference_seconds,
            bool,
        )
        or not isinstance(
            inference_seconds,
            (int, float),
        )
        or not math.isfinite(
            float(
                inference_seconds
            )
        )
        or float(
            inference_seconds
        )
        < 0.0
    ):
        raise ControlPlaneError(
            "KL720 inference timing is invalid"
        )

    output_path = Path(
        str(
            raw_output.get(
                "path",
                "",
            )
        )
    ).expanduser().resolve()

    if not output_path.is_file():
        raise ControlPlaneError(
            "Verified KL720 raw output is missing"
        )

    return {
        "window_index": (
            window_index
        ),
        "y": y,
        "x": x,
        "input": (
            _artifact_record(
                tensor
            )
        ),
        "run_record": (
            _artifact_record(
                record_path
            )
        ),
        "output": dict(
            raw_output
        ),
        "inference_ms": (
            float(
                inference_seconds
            )
            * 1000.0
        ),
    }
