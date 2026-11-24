"""NASA burden-regression architectures with an import-safe public registry.

The frozen R1 v2 scripts remain unchanged. This factory reproduces their
R1CountNetV1 architecture and adds candidate backbones for a new common
benchmark. Creating a model does not validate its accuracy or KL720 execution.

Only ``create_model`` imports PyTorch, and only the torchvision candidates
import torchvision. Merely listing models or inspecting their contracts does
not require either package, a GPU, a network connection or the Kneron SDK.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    family: str
    description: str
    pretrained_weights: str | None
    input_channels: int = 3
    input_height: int = 256
    input_width: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SPECS = (
    ModelSpec(
        architecture="r1countnet_v1",
        family="compact_conv_bn_relu",
        description=(
            "Six-stage compact CNN matching the historical NASA R1CountNetV1 "
            "architecture and state-dictionary keys. New weights need evaluation."
        ),
        pretrained_weights=None,
    ),
    ModelSpec(
        architecture="mobilenet_v2",
        family="mobilenet_v2",
        description="Torchvision MobileNetV2 features with a scalar NASA burden head.",
        pretrained_weights="MobileNet_V2_Weights.IMAGENET1K_V1",
    ),
    ModelSpec(
        architecture="resnet18",
        family="resnet",
        description="Torchvision ResNet18 features with a scalar NASA burden head.",
        pretrained_weights="ResNet18_Weights.IMAGENET1K_V1",
    ),
)
_BY_NAME = {spec.architecture: spec for spec in _SPECS}
MODEL_NAMES = tuple(spec.architecture for spec in _SPECS)


def list_model_specs() -> tuple[ModelSpec, ...]:
    """Return the initial extensible architecture queue without loading models."""

    return _SPECS


def get_model_spec(architecture: str) -> ModelSpec:
    if not isinstance(architecture, str) or architecture not in _BY_NAME:
        raise ValueError(
            f"Unknown NASA architecture {architecture!r}; choose one of: "
            + ", ".join(MODEL_NAMES)
        )
    return _BY_NAME[architecture]


def _initialization(spec: ModelSpec, pretrained: bool) -> dict[str, Any]:
    if not isinstance(pretrained, bool):
        raise ValueError("pretrained must be an explicit boolean")
    if pretrained and spec.pretrained_weights is None:
        raise ValueError(
            f"{spec.architecture} has no pretrained weight selection; "
            "load a separately identified checkpoint when required"
        )
    return {
        "pretrained_backbone": pretrained,
        "pretrained_weights": spec.pretrained_weights if pretrained else None,
        "head": "normal_weight_std_0.01_bias_0.5_then_relu",
    }


def model_contract(architecture: str, *, pretrained: bool = False) -> dict[str, Any]:
    """Describe a newly constructed candidate and the shared scientific input.

    Pretrained weights are explicitly opt-in and may require a download when
    the caller constructs the model. They do not change NASA preprocessing.
    """

    spec = get_model_spec(architecture)
    return {
        "schema_version": 1,
        "architecture": spec.architecture,
        "family": spec.family,
        "task": "nasa_sample_53bp1_burden_regression",
        "input": {
            "dtype": "float32",
            "layout": "NCHW",
            "training_shape": [None, 3, 256, 256],
            "deployment_shape": [1, 3, 256, 256],
            "channels": ["FITC_53BP1", "DAPI", "zeros"],
            "native_pixel_scale": True,
            "resize": False,
            "padding": "center_to_256",
            "normalization": "per_native_channel_percentile_1_to_99.5_clip_0_to_1",
            "mask_is_model_input": False,
        },
        "output": {
            "shape": [None],
            "semantics": "latent_continuous_per_nucleus_53bp1_burden",
            "nonnegative": True,
            "sample_aggregation": "arithmetic_mean",
            "reference_label": "sample_plate_well_avg_nfoci",
        },
        "initialization": _initialization(spec, pretrained),
        "deployment_validation": "required_for_each_trained_model",
    }


def create_model(architecture: str, *, pretrained: bool = False) -> Any:
    """Build a model with nonnegative ``[N]`` outputs for NASA ``[N,3,256,256]``.

    The default is random initialization and does not download weights.
    Torchvision is an optional, version-compatible training dependency for
    MobileNetV2 and ResNet18; it is unnecessary for the compact CNN.
    """

    spec = get_model_spec(architecture)
    contract = model_contract(architecture, pretrained=pretrained)
    try:
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "Creating a NASA model requires PyTorch in the scientific/training "
            "environment; model listing does not."
        ) from exc

    if spec.architecture == "r1countnet_v1":
        channels = (3, 24, 32, 48, 64, 96, 128)
        features = nn.Sequential(
            *(
                nn.Sequential(
                    nn.Conv2d(a, b, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(b),
                    nn.ReLU(inplace=True),
                )
                for a, b in zip(channels[:-1], channels[1:])
            )
        )
        feature_channels = 128
    else:
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                f"Creating {spec.architecture} requires a torchvision build "
                "compatible with the scientific environment's PyTorch."
            ) from exc
        if spec.architecture == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.mobilenet_v2(weights=weights)
            features = backbone.features
            feature_channels = backbone.last_channel
        else:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            features = nn.Sequential(*list(backbone.children())[:-2])
            feature_channels = backbone.fc.in_features

    class BurdenRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = nn.Linear(feature_channels, 1)
            self.nonnegative = nn.ReLU()
            nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.head.bias, 0.5)

        def forward(self, x: Any) -> Any:
            x = self.features(x)
            x = self.pool(x).flatten(1)
            x = self.head(x)
            x = self.nonnegative(x)
            return x[:, 0]

    model = BurdenRegressor()
    model.radedge_contract = contract
    return model


def checkpoint_identity(path: Path | str) -> dict[str, Any]:
    """Fingerprint checkpoint bytes without importing torch or loading pickle.

    This is an artifact identity, not validation of the stored architecture or
    weights. The training/export adapter owns that validation and its metadata.
    """

    resolved = Path(path).expanduser().resolve()
    before = resolved.stat()
    if not resolved.is_file():
        raise ValueError(f"Checkpoint is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Checkpoint changed while its identity was being computed")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size}
