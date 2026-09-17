import importlib
import sys


def test_audit_imports_do_not_require_kneron_modules():
    for name in ("kp", "ktc"):
        sys.modules.pop(name, None)
    importlib.import_module("radiation_edge_ai.audit_utils")
    importlib.import_module("radiation_edge_ai.nasa_bps.audit_r1_v2_evidence")
    importlib.import_module("radiation_edge_ai.dna_fiber.audit_hardware_characterization_512")
    importlib.import_module("radiation_edge_ai.dna_fiber.provenance_cache")
    importlib.import_module("radiation_edge_ai.dna_fiber.matching_sensitivity")
