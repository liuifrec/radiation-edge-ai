# ResNet18 stem MaxPool precision probe

Freeze this follow-up before computing any boundary-specific corrective result.
Use the original 48 ResNet18 training inputs. Report the one historical pilot
separately. In the primary paired comparison, insert each fixed BIE's actual
stem ReLU and actual following MaxPool tensors into the same optimized FP32
suffix and measure the post-pool minus pre-pool **preactivation** in foci/nucleus.
First reproduce the full FP32 preactivation from each FP32 cut. Require a
byte-matched input and model identity for every replay. Report signed mean,
mean absolute contribution, source means, paired differences and zeros.
There is no new threshold for declaring equivalence to zero.

If the pinned KL720 interface cannot enforce a precision assignment to this
particular boundary, perform one host-side sensitivity experiment. Hold the
**standard** BIE's decoded stem ReLU tensor fixed. Apply the original ONNX
MaxPool and round its output once to the full-precision BIE's actual per-channel
MaxPool grid using round-to-nearest, ties-to-even and saturation at its signed
effective width. Substitute the resulting tensor into the same FP32 suffix.
Compare this increment with the measured standard-BIE increment on the same
nucleus. Save per-case values, clipping counts, the descriptor identity and
the population of exact half steps. This question isolates the effect of
a finer output grid given the original upstream activations. Its downstream
arithmetic is FP32. It supplies no native mixed-precision BIE, compiler
performance estimate or candidate eligibility.
