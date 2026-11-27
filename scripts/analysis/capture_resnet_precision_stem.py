"""Capture native boundaries around MaxPool on the frozen ResNet18 precision comparator."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mobilenet_feature_capture as capture


PANEL = "fecec721ddf8ad424ed9f5ac13594a0488d2fe6805aa0031aff11c18a602113a"
BIE = "3f654f267ae280f27d6aaf6715b32af4285c5de6f4eb13c844e02e70f2f69c9b"
CAPTURE = "82dbd9ae23a8a3b54a79625d849f6355c39612623990d48ee5c96407d31df52f"
TAPS = {"stem_relu": "/features/features.2/Relu_output_0",
        "stem_pool": "/features/features.3/MaxPool_output_0", "burden": "burden"}


def trace(args):
    capture.TAPS = TAPS
    capture.capture_one(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    capture.checked(capture.__file__, CAPTURE)
    model = capture.checked(Path("/model.bie"), BIE)
    rows = json.loads(capture.checked(Path("/panel.json"), PANEL).read_bytes())["rows"]
    capture.check(len(rows) == 48 and len({r["sample_name"] for r in rows}) == 48 and
                  all(r["fold_female_role"] == "train" and r["source_name"] in capture.SOURCES for r in rows),
                  "Unexpected ResNet training panel")
    args.out.mkdir(exist_ok=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(trace, [(r, args.out, model, Path("/export"), PANEL) for r in rows]))


if __name__ == "__main__":
    main()
