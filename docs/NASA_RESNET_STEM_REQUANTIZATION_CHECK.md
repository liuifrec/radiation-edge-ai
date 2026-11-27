# Follow-up to the standard INT8 split-stem training replay

The first replay found a mean preactivation increment of −0.069048 after
substituting the native stem MaxPool, following a mean increment of +0.024260
at Conv and −0.000225 at ReLU. This follow-up tests a predicted consequence
of the observed 16-bit-to-8-bit boundary. Freeze the check before examining
the native pool code distribution.

Use the same 48 training nuclei with the original pilot reported separately.
Extract the stem MaxPool operation from the frozen optimized ONNX. On each
nucleus require bit-exact FP32 ReLU-to-MaxPool replay. Apply that same
operation to the observed BIE ReLU activation. Compare its unrounded result
with the actual signed native pool words. Verify each channel's observed
signed-word spacing reproduces all native decoded float values. Report how
many FP32-on-native-ReLU values lie within the signed range, and how many
match clipped floor or nearest rounding. Count differences between these
predictions away from ambiguous integer and half-integer values (0.0001
code units), broken down into matches with each prediction or neither.
Use the existing code spacing; no per-case offset or model fitting.
Record FP32-suffix readouts on the exact pooled native ReLU for comparison
with the suffix on the actual BIE pool. These checks exclude assumptions
about any operations fused inside the vendor graph and cannot uniquely
establish the hardware rounding implementation.
