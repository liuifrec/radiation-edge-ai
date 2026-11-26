"""Shared neural/host boundary for the released 2D Spotiflow 0.6.5 checkpoints."""

import numpy as np

SIDE = 128
CHANNELS = ("heatmap_logit", "raw_flow_0", "raw_flow_1", "raw_flow_2")


def _check_geometry(shape):
    if len(shape) != 2 or any((s + 15) // 16 * 16 != SIDE for s in shape):
        raise ValueError("Only crops whose native 16-pixel grid padding is 128 x 128 are supported")


def prepare_input(image):
    """Apply the upstream single-tile 2D normalizer and its original boundary padding."""
    _check_geometry(image.shape)
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite Spotiflow input")
    from spotiflow.utils import center_pad, normalize

    image = normalize(image.astype(np.float32)[..., None])
    padded, _ = center_pad(image, (SIDE, SIDE, 1), mode="reflect")
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)


def neural_core(model):
    """Retain exactly the released feature extractor, finest heatmap head and raw flow head."""
    import torch
    from torch import nn

    if (model.config.is_3d or model.config.in_channels != 1 or model.config.out_channels != 1
            or model.config.sigma != 1 or not model.config.compute_flow or model._post.mix_last
            or not isinstance(model._bg_remover, nn.Identity) or model._downsampler is not None):
        raise ValueError("Unsupported Spotiflow model configuration")

    class SpotiflowCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = model._backbone
            self.finest_head = model._post.heads[0]
            self.finest_output = model._post.last_convs[0]
            self.flow = model._flow

        def forward(self, image):
            finest = self.backbone(image)[0]
            logit = self.finest_output(self.finest_head(finest))
            return torch.cat((logit, self.flow(finest)), dim=1)

    return SpotiflowCore().eval()


def join_heads(logits, raw_flow):
    """Restore the original channel axis after separate descriptor-aware output decoding."""
    for values, channels in ((logits, 1), (raw_flow, 3)):
        if values.shape != (1, channels, SIDE, SIDE) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError("Expected ordered float32 logit and raw-flow NCHW outputs")
    return np.concatenate((logits, raw_flow), axis=1)


def decode_heads(heads, shape, threshold):
    """Decode the released model's four raw heads on the host using its native peak routine."""
    import torch
    from torch.nn.functional import normalize
    from spotiflow.utils import center_crop, filter_shape, flow_to_vector, prob_to_points, subpixel_offset

    _check_geometry(shape)
    shape = tuple(shape)
    if (heads.shape != (1, 4, SIDE, SIDE) or heads.dtype != np.float32
            or not np.isfinite(heads).all() or not 0 < threshold < 1):
        raise ValueError("Expected finite float32 NCHW heatmap/flow heads and a probability threshold")
    raw = torch.from_numpy(np.ascontiguousarray(heads))
    heatmap = center_crop(torch.sigmoid(raw[0, 0]).numpy(), shape)
    unit_flow = normalize(raw[:, 1:], dim=1)[0].permute(1, 2, 0).numpy()
    flow = np.ascontiguousarray(center_crop(unit_flow, shape))
    peaks = prob_to_points(heatmap, prob_thresh=threshold, exclude_border=False, mode="fast", min_distance=1)
    confidence = heatmap[tuple(peaks.astype(int).T)]
    offsets = subpixel_offset(peaks, flow_to_vector(flow, sigma=1.0), heatmap, radius=0)
    refined = peaks + offsets
    points, valid = filter_shape(refined, shape, return_mask=True)
    return {"heatmap": heatmap, "flow": flow, "points": points, "confidence": confidence[valid]}


def target_assignment(mask, target_label, points, count):
    """Use nearest pixel centre with a tie towards the higher index; never clip coordinates."""
    import numpy as np
    if type(count) is not int or count < 0 or len(points) != count:
        raise ValueError("Detector count and coordinates disagree")
    if target_label <= 0 or not np.any(mask == target_label):
        raise ValueError("Unknown target nucleus label")
    xy = np.asarray(points, dtype=np.float64)
    if count == 0:
        return [], [], []
    if xy.shape != (count, 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid native YX coordinates")
    in_frame = np.all((xy >= -0.5) & (xy < np.array(mask.shape) - 0.5), axis=1)
    native_pixels = np.floor(xy[in_frame] + 0.5).astype(int)
    pixels, labels = [None] * count, [None] * count
    for i, (y, x) in zip(np.flatnonzero(in_frame), native_pixels):
        pixels[i] = [int(y), int(x)]
        labels[i] = int(mask[y, x])
    return pixels, [label == target_label for label in labels], labels
