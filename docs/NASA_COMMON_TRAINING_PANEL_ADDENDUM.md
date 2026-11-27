# Matched training-source comparison

The enrollment metadata show only four common, nonpilot sample IDs between
the two original diagnostic panels. All four come from the C57BL sources.
To compare architectures on identical input bytes across all four sources,
apply the **unchanged ResNet18 standard INT8 BIE** to every one of the 24
MobileNetV2 confirmation rows in the already sealed selection
`534fffa2f104bf395952fe25ecc8708ca4b91bf032dc521b19298cf5fe35ebfe`.
This addendum precedes the additional ResNet18 model executions. It adds
no precision configuration or selection criterion. The original 47-nucleus
ResNet18 analysis remains the primary within-architecture sample.

Use the same original rectified-head substitution and units as the primary
protocol. Verify the common subset against the original ResNet18 archive
by input hash, terminal value and pooled-head reconstruction. Report six
cases per source and all 24 pairs. The range strata were chosen by MobileNetV2
FP32. The selected cases do not represent a random sample of either model's
prediction distribution.
