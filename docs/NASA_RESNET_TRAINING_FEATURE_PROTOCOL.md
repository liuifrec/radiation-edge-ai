# Frozen ResNet18 training feature replay

This diagnostic uses the standard female-fold INT8 BIE and its archived vendor
dump-level-2 outputs. The selection receipt SHA256 is
`fecec721ddf8ad424ed9f5ac13594a0488d2fe6805aa0031aff11c18a602113a`.
One nucleus per training bag covers four sources and 48 bags. Report the
original pilot `BPSR1V2D_01501` separately and summarize the remaining 47.
These are previously measured calibration inputs. Biological source, rather
than nucleus, defines an independent source.

1. Verify selection, artifact hashes, training membership and all native files
   against the archived analysis. Read the public BIE input and output
   descriptors. Reconstruct each final signed word with the actual output
   scale and radix. Use native decoded floats for intermediate computations.
2. For original and optimized ONNX, expose the stem Conv, ReLU, MaxPool,
   final residual map, pooled vector, affine head and final output. Compare
   the tapped and unchanged graphs on each input. Use the original affine
   weights on the native pooled vector. Confirm the optimized and original
   head weights and bias match. Bind all substitutions to graph output names
   and native size records.
3. Run the optimized FP32 suffix from each stem boundary. Require the FP32
   activation to reproduce the full graph pre-ReLU head within 0.00001
   foci/nucleus on every case before using the native activation. Compute
   successive head changes from the rounded-input FP32 execution, then from
   Conv, ReLU and MaxPool substitutions. Report the earliest measured negative
   source-mean increment consistent across all four sources. Report magnitudes
   even if sources disagree. No implication of a specific fused arithmetic
   defect follows from such a boundary.
4. Count stem signed storage endpoints and determine whether values at the
   lower endpoint remain positive after ReLU. Report final-map and pooled
   FP32, native and signed-error distributions, channelwise errors and signed
   weight contributions. Compare FP32 head-on-native-pool after ReLU with
   the original output and final BIE. Separate measured head, ReLU and
   terminal changes. Keep intermediate scaling evidence distinct from public
   model IO metadata.
5. Summarize mean signed/absolute effects for each source, radiation
   condition, original FP32 zero and positive-burden quartiles inherited
   from the training panel, and combined input RMS quartile. Retain all
   per-sample results and report simulator runtimes separately from the new
   host replay. This readout supports mechanism comparison and engineering
   candidate design only. It has no deployment advancement gate.
