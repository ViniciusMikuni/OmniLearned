"""Exact within-event neighbors, bounded temporary distance memory."""

import torch


@torch.no_grad()
def knn(points, mask, k, chunk_size=256):
    if k < 1:
        raise ValueError("k must be positive")
    b, n, _ = points.shape
    valid, work = mask.squeeze(-1).bool(), points.float()
    norms = work.square().sum(-1)
    indices_all, masks_all = [], []
    take = min(k, n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d = (
            norms[:, start:end, None]
            + norms[:, None, :]
            - 2 * work[:, start:end] @ work.transpose(1, 2)
        )
        d = d.clamp_min_(0).masked_fill(~valid[:, None, :], torch.inf)
        local = torch.arange(end - start, device=points.device)
        d[:, local, local + start] = torch.inf
        distances, indices = d.topk(take, dim=-1, largest=False, sorted=True)
        live = distances.isfinite() & valid[:, start:end, None]
        if take < k:
            indices = torch.nn.functional.pad(indices, (0, k - take))
            live = torch.nn.functional.pad(live, (0, k - take))
        centers = torch.arange(start, end, device=points.device)[None, :, None]
        indices_all.append(torch.where(live, indices, centers))
        masks_all.append(live)
    indices = (
        torch.cat(indices_all, 1)
        + torch.arange(b, device=points.device)[:, None, None] * n
    )
    return indices.reshape(-1), torch.cat(masks_all, 1).unsqueeze(-1)
