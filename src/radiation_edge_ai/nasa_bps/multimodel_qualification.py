"""Pinned MobileNetV2 female-fold identities shared with the Python 3.7 vendor worker."""

ARCHITECTURE = "mobilenet_v2"
FOLD = "fold_female_role"
CAMPAIGN_SHA = "7d5c042e2d6f929a3454fc3890f2228c13670a3a0577b674bae9b170b6585687"
SCREEN_SHA = "9bab4fa38ef118a2da7cfbaa9f7fa710eb75f02c132c0e3d87be945c53dad983"
EXPORT_SHA = "40b89d4b349dfa4060e4fd62a31cf27647b6fb04c1bdef93b4f7b84165f32257"
MOBILE = {
    "source_onnx_sha256": "2f18b2330ca3289ab422e1ff47b526b4bfa6a46fc4bd7354346609715f575e45",
    "optimized_onnx_sha256": "75824e1ed4f8520d90a942c1351926b922db9b4284b8b6d7cd5663428e73c074",
    "training_run_sha256": "861e93ef0a128fdab8f81cb2d992a3fce8833aa92fdc09494c9008ebad4fe5b5",
    "fp32_nuclei_sha256": "13fe8bec71a7077f5c98e4548d1cf845550140837e55972b5e99dac8ea98379f",
    "fp32_bags_sha256": "374355b631164f1c8a0ae7e587dfb7fc41c099c924861d8f5611c5fe5b12ca7f",
    "development_manifest_sha256": "23e21fdc8bf00a2814b64c56223cfd3d8e96eda71bdbff5d8ec74dae218cf5a9",
    "nasa_reference_sha256": "fcbf1339805e02a7ca9cec36b476438b0512b222bd70ac2f90ed75c238a0b01f",
}
# Set only when a prospective candidate policy has been committed before fixed readout.
POLICY_SHA = None


def verify_policy(policy, actual_sha):
    if not POLICY_SHA or actual_sha != POLICY_SHA:
        raise ValueError("Unrecognized prospective MobileNetV2 policy")
    if (policy.get("architecture"), policy.get("fold")) != (ARCHITECTURE, FOLD):
        raise ValueError("Unexpected MobileNetV2 architecture or fold")
    if any(policy.get(k) != v for k, v in MOBILE.items()):
        raise ValueError("Policy does not bind the frozen MobileNetV2 inputs")


def verify_staged_manifest(staged, staged_digest, manifest_digest, policy):
    if (staged_digest != policy["fp32_preparation_sha256"] or
            staged.get("status") != "FP32_PREPARATION_COMPLETE" or
            staged.get("input_manifest_sha256") != manifest_digest):
        raise ValueError("Evaluation inputs differ from the sealed FP32 preparation")
