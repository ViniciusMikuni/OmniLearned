"""Dense conversions are confined to operations that require them."""

import torch


def padded(values, offsets, width=None):
    lengths = offsets[1:] - offsets[:-1]
    width = max(1, int(lengths.max())) if width is None else width
    mask = torch.arange(width, device=values.device)[None] < lengths[:, None]
    result = values.new_zeros((len(lengths), width, *values.shape[1:]))
    result[mask] = values
    return result, mask.unsqueeze(-1)


def offsets_from_mask(mask):
    lengths = mask.squeeze(-1).sum(-1)
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))


def dense_batch(batch):
    out = dict(batch)
    if "offsets" in batch:
        for key in ("X", "pid", "add_info", "data_pid"):
            if batch.get(key) is not None:
                out[key], out["mask"] = padded(batch[key], batch["offsets"])
    return out
