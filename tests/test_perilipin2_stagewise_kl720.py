"""Source-lock and dependency-isolation guards; no models, images, or hardware."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "windows"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import perilipin2_kl720_stagewise_proof as proof  # noqa: E402
import perilipin2_kl720_stagewise_vendor_gate as gate  # noqa: E402


def node(op, inputs, outputs, name="node", nested=()):
    return NS(op_type=op, input=inputs, output=outputs, name=name,
              attribute=[NS(type=t) for t in nested])


def graph(nodes, inputs=("rgb_chw_normalized",), weights=()):
    return NS(node=nodes, input=[NS(name=n) for n in inputs],
              initializer=[NS(name=w) for w in weights], sparse_initializer=[])


class ClosureTests(unittest.TestCase):
    def test_dependency_closure_keeps_skip_path_but_not_unrelated_nodes(self):
        g = graph([
            node("Conv", ["rgb_chw_normalized", "w"], ["encoder"]),
            node("Relu", ["encoder"], ["skip"]),
            node("Conv", ["encoder", "w"], ["deep"]),
            node("Constant", [], ["dead"]),
            node("ConvTranspose", ["deep", "up_w"], ["up"]),
            node("Concat", ["up", "skip"], ["merged"]),
            node("Conv", ["merged", "w"], ["head"]),
        ], weights=("w", "up_w"))
        indices, names = proof.dependency_closure(g, "merged")
        self.assertEqual(indices, {0, 1, 2, 4, 5})
        self.assertIn("up_w", names)
        self.assertNotIn("dead", names)
        self.assertNotIn("head", names)

    def test_duplicate_tensor_and_unknown_dependency_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
            proof.dependency_closure(graph([
                node("Relu", ["rgb_chw_normalized"], ["same"]),
                node("Relu", ["rgb_chw_normalized"], ["same"]),
            ]), "same")
        with self.assertRaisesRegex(ValueError, "Unresolved dependency"):
            proof.dependency_closure(graph([
                node("Relu", ["unknown"], ["out"]),
            ]), "out")

    def test_control_flow_and_multiple_graph_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "Nested graph"):
            proof.dependency_closure(graph([
                node("If", ["rgb_chw_normalized"], ["out"], nested=(5,))
            ]), "out")
        with self.assertRaisesRegex(ValueError, "one model input"):
            proof.dependency_closure(graph([
                node("Relu", ["a"], ["out"])
            ], inputs=("a", "b")), "out")

    def test_original_decoder_stage_geometry_pinned(self):
        rows, nodes = [], []
        for i, (c, h) in enumerate([(96, 32), (72, 64), (48, 128), (24, 256)]):
            name = "up{}".format(i)
            rows.append({"node_name": name, "input_shape": [1, c, h, h],
                         "target_shape": [1, c, 2*h, 2*h]})
            nodes.append(node("ConvTranspose", ["pre" + name, "kernel"],
                              ["post" + name], name))
        observed = proof.cutpoints({"resize_nodes": rows}, NS(graph=graph(nodes)))
        self.assertEqual(observed[0],
                         ("encoder_before_first_upsample", "preup0", [1, 96, 32, 32]))
        self.assertEqual(observed[-1], ("upsample_4", "postup3", [1, 24, 512, 512]))
        rows[0]["input_shape"][1] = 97
        with self.assertRaisesRegex(ValueError, "shape"):
            proof.cutpoints({"resize_nodes": rows}, NS(graph=graph(nodes)))

    def test_stages_must_have_strictly_increasing_convtranspose_count(self):
        rows = [{"stage": label, "tensor": label,
                 "node_count": 10 + i*4, "convtranspose_count": i}
                for i, label in enumerate(proof.STAGES)]
        proof.verify_monotone(rows, "logits")
        rows[2]["convtranspose_count"] = 4
        with self.assertRaisesRegex(ValueError, "Non-monotone"):
            proof.verify_monotone(rows, "logits")


class VendorTests(unittest.TestCase):
    def example(self):
        return {"schema": gate.PROOF_SCHEMA, "status": "STAGED_FP32_PARITY_EXACT",
                "source_frozen_sha256": gate.v.FROZEN_SOURCE_SHA,
                "nef_compiled": False, "physical_kl720_inference": False,
                "stages": [
                    {"stage": label, "filename": label + ".onnx",
                     "sha256": "a"*64, "node_count": i+20,
                     "convtranspose_count": i, "initializers": 3,
                     "tensor": label, "shape": [1, 96, 64, 64],
                     "parity": [{"max_abs_error": 0.0, "exact_values": True}]*2}
                    for i, label in enumerate(gate.STAGES)]}

    def test_only_proven_stage_accepted(self):
        value = self.example()
        self.assertEqual(gate.checked_stage(value, "upsample_2")["stage"], "upsample_2")
        value["stages"][2]["parity"][0]["exact_values"] = False
        with self.assertRaisesRegex(ValueError, "exact-parity"):
            gate.checked_stage(value, "upsample_2")

    def test_unexpected_model_identity_fails(self):
        value = self.example()
        value["source_frozen_sha256"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "source-frozen"):
            gate.checked_stage(value, "upsample_1")
        value = self.example()
        value["stages"][4]["convtranspose_count"] = 3
        with self.assertRaisesRegex(ValueError, "node-count"):
            gate.checked_stage(value, "upsample_4")

    def test_vendor_failure_detection_never_promotes_report_to_device_pass(self):
        self.assertIn("compiler invalid input",
                      gate.v.failure_markers("compiler invalid input"))
        self.assertTrue(gate.v.failure_markers('Failure for model "input/input"'))
        self.assertTrue(gate.v.failure_markers("Err: 134"))
        self.assertFalse(gate.v.failure_markers('Success for model "input/input"'))


if __name__ == "__main__":
    unittest.main()
