from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from radiation_edge_ai.dna_fiber.matching_sensitivity import (
    greedy_iou_matching,
    matching_summary,
    maximum_cardinality_iou_matching,
    reference_defined_nonempty,
)
from radiation_edge_ai.dna_fiber.provenance_cache import (
    CacheIdentity,
    verify_cached_array,
    write_cache_metadata,
)


@dataclass
class Rect:
    x0: float
    x1: float
    y0: float = 0.0
    y1: float = 10.0

    def bbox_iou(self, other: Rect) -> float:
        ix0 = max(self.x0, other.x0)
        ix1 = min(self.x1, other.x1)
        iy0 = max(self.y0, other.y0)
        iy1 = min(self.y1, other.y1)
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area_a = max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)
        area_b = max(0.0, other.x1 - other.x0) * max(0.0, other.y1 - other.y0)
        union = area_a + area_b - intersection
        return 0.0 if union <= 0 else intersection / union


def test_greedy_counterexample_real_boxes_vs_maximum_cardinality():
    # X=[0,10], Y=[4,14], A=[1,11], B=[-2,8].  At IoU>=0.5:
    # A-X=.818 (greedy takes), A-Y=.538, B-X=.667, B-Y=.25.
    # Greedy therefore yields one match; maximum-cardinality yields two.
    reference = [Rect(1, 11), Rect(-2, 8)]
    candidate = [Rect(0, 10), Rect(4, 14)]
    greedy = greedy_iou_matching(reference, candidate, threshold=0.5)
    maximum = maximum_cardinality_iou_matching(reference, candidate, threshold=0.5)
    assert len(greedy) == 1
    assert len(maximum) == 2
    assert matching_summary(2, 2, greedy)["f1"] == 0.5
    assert matching_summary(2, 2, maximum)["f1"] == 1.0


def test_all_empty_and_zero_match_denominators():
    empty = matching_summary(0, 0, [])
    assert empty["precision"] == empty["recall"] == empty["f1"] == 1.0
    missing = matching_summary(2, 0, [])
    assert missing["precision"] == 0.0 and missing["recall"] == 0.0 and missing["f1"] == 0.0
    extra = matching_summary(0, 2, [])
    assert extra["precision"] == 0.0 and extra["recall"] == 0.0 and extra["f1"] == 0.0


def test_reference_defined_nonempty_excludes_physical_candidate_logic():
    assert reference_defined_nonempty(0, [0, 0, 0, 0]) is False
    assert reference_defined_nonempty(1, [0, 0, 0, 0]) is True
    assert reference_defined_nonempty(0, [0, 0, 1, 0]) is True
    with pytest.raises(RuntimeError):
        reference_defined_nonempty(-1, [0])


def test_cache_valid_shape_wrong_identity_is_rejected(tmp_path: Path):
    array_path = tmp_path / "output.npy"
    np.save(array_path, np.zeros((1, 3, 8, 8), dtype=np.float32), allow_pickle=False)
    metadata_path = tmp_path / "output.provenance.json"
    identity = CacheIdentity(
        input_sha256="a" * 64,
        model_sha256="b" * 64,
        preprocessing_sha256="c" * 64,
        runtime_provider="CPUExecutionProvider",
        runtime_version="1.0",
        expected_shape=(1, 3, 8, 8),
    )
    write_cache_metadata(array_path, metadata_path, identity)
    wrong = CacheIdentity(
        input_sha256="d" * 64,
        model_sha256=identity.model_sha256,
        preprocessing_sha256=identity.preprocessing_sha256,
        runtime_provider=identity.runtime_provider,
        runtime_version=identity.runtime_version,
        expected_shape=identity.expected_shape,
    )
    with pytest.raises(RuntimeError, match="identity"):
        verify_cached_array(array_path, metadata_path, wrong)


def test_legacy_cache_without_metadata_is_unverified(tmp_path: Path):
    array_path = tmp_path / "legacy.npy"
    np.save(array_path, np.zeros((1, 3, 8, 8), dtype=np.float32), allow_pickle=False)
    identity = CacheIdentity(
        input_sha256="a" * 64,
        model_sha256="b" * 64,
        preprocessing_sha256="c" * 64,
        runtime_provider="CPUExecutionProvider",
        runtime_version="1.0",
        expected_shape=(1, 3, 8, 8),
    )
    with pytest.raises(RuntimeError, match="Legacy/unverified cache"):
        verify_cached_array(array_path, tmp_path / "missing.json", identity)
