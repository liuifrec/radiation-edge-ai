"""Run the platform's registered Spotiflow adapter from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from radiation_edge_ai.hardware.pretrained_infer import main

if __name__ == "__main__":
    main()
