from sklearn import metrics
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from copy import deepcopy
import torch.distributed as dist
from torch.distributed import init_process_group, get_rank
import torch.nn.functional as F
import requests


def get_model_parameters(model_size):
    model_dict = {}
    if model_size == "small":
        model_dict["num_transformers"] = 8
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 8
        model_dict["base_dim"] = 128
        model_dict["mlp_ratio"] = 2

    elif model_size == "lite":
        model_dict["num_transformers"] = 8
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 8
        model_dict["base_dim"] = 128
        model_dict["mlp_ratio"] = 2
        model_dict["use_local"] = False

    elif model_size == "dense":
        model_dict["num_transformers"] = 8
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 8
        model_dict["base_dim"] = 128
        model_dict["mlp_ratio"] = 2
        model_dict["use_local"] = False
        model_dict["use_attn"] = False

    elif model_size == "medium":
        model_dict["num_transformers"] = 12
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 16
        model_dict["base_dim"] = 512
        model_dict["mlp_ratio"] = 2

    elif model_size == "large":
        model_dict["num_transformers"] = 28
        model_dict["num_transformers_head"] = 4
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 32
        model_dict["base_dim"] = 1024
        model_dict["mlp_ratio"] = 2
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    return model_dict


def print_metrics(y_preds_np, y_np, thresholds=[0.3, 0.5], background_class=0):
    # Compute multiclass AUC
    auc_ovo = metrics.roc_auc_score(
        y_np,
        y_preds_np if y_preds_np.shape[-1] > 2 else y_preds_np[:, -1],
        multi_class="ovo",
    )
    print(f"AUC: {auc_ovo:.4f}\n")

    accuracy = metrics.accuracy_score(y_np, np.argmax(y_preds_np, axis=1))

    print(f"ACC: {accuracy:.4f}\n")

    num_classes = y_preds_np.shape[1]

    for signal_class in range(num_classes):
        if signal_class == background_class:
            continue

        # Create binary labels: 1 for signal_class, 0 for background_class, ignore others
        mask = (y_np == signal_class) | (y_np == background_class)
        y_bin = (y_np[mask] == signal_class).astype(int)
        scores_bin = y_preds_np[mask, signal_class] / (
            y_preds_np[mask, signal_class] + y_preds_np[mask, background_class]
        )

        # Compute ROC
        fpr, tpr, _ = metrics.roc_curve(y_bin, scores_bin)

        print(f"Signal class {signal_class} vs Background class {background_class}:")

        for threshold in thresholds:
            bineff = np.argmax(tpr > threshold)
            print(
                "Class {} effS at {} 1.0/effB = {}".format(
                    signal_class, tpr[bineff], 1.0 / fpr[bineff]
                )
            )


class CLIPLoss(nn.Module):
    # From AstroCLIP: https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/models/astroclip.py#L117
    def get_logits(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        clean_features = F.normalize(clean_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        perturbed_features = F.normalize(perturbed_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features

        logits_per_clean = logit_scale * clean_features @ perturbed_features.T
        return logits_per_clean, logits_per_clean.T

    def forward(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        weight=None,
        logit_scale: float = 2.74,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the clean and perturbed features
        logits_per_clean, logits_per_perturbed = self.get_logits(
            clean_features, perturbed_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_clean.shape[0], device=clean_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_clean, labels, reduction="none")
            + F.cross_entropy(logits_per_perturbed, labels, reduction="none")
        ) / 2
        if weight is not None:
            total_loss = torch.mean(weight * total_loss)
        else:
            total_loss = total_loss.mean()
        return {"contrastive_loss": total_loss} if output_dict else total_loss


def sum_reduce(num, device):
    r"""Sum the tensor across the devices."""
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def pad_array(tensor_list, M: int = 150) -> torch.Tensor:
    """
    Given a list of torch tensors, each of shape (B, N_i, F),
    pads or truncates each along dimension N to length M,
    and returns a single tensor of shape (I, M, F), where
      H = sum over list of B,
      M = target length,
      F = feature dimension.
    """
    # Determine total number of samples and feature dim
    H = sum(t.shape[0] for t in tensor_list)
    _, _, F = tensor_list[0].shape

    # Use the dtype/device of the first tensor
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype

    # Allocate output buffer
    out = torch.zeros((H, M, F), dtype=dtype, device=device)

    idx = 0
    for t in tensor_list:
        B, N, F_ = t.shape
        assert F_ == F, "All tensors must have the same feature dimension F"

        if N < M:
            # create a (B, M, F) zero tensor and copy `t` into its first N slots
            padded = torch.zeros((B, M, F), dtype=dtype, device=device)
            padded[:, :N, :] = t
        else:
            # truncate to the first M points
            padded = t[:, :M, :]

        out[idx : idx + B] = padded
        idx += B

    return out


def get_class_loss(weight, pred, y, class_cost, use_event_loss=False, logs={}):
    loss = 0.0
    if use_event_loss:
        event_mask = y >= 200
        if event_mask.any():
            loss_event = torch.mean(
                weight[event_mask]
                * class_cost(pred[event_mask][:, 200:], y[event_mask] - 200)
            )
            logs["loss_class_event"] += loss_event.detach()
            loss = loss + loss_event
        if (~event_mask).any():
            loss_class = torch.mean(
                weight[~event_mask]
                * class_cost(pred[~event_mask][:, :200], y[~event_mask])
            )
            logs["loss_class"] += loss_class.detach()
            loss = loss + loss_class
    else:
        loss_class = torch.mean(weight * class_cost(pred, y))
        loss = loss + loss_class
        logs["loss_class"] += loss_class.detach()

    return loss


# def wd_loss(x_hat, x, mask, mask_logits=None,
#             blur=0.01, blur_start=None, n_iter=100,
#             scaling=0.5, eps=1e-8, chunk_size=128):
#     """
#     Memory-efficient entropic Wasserstein (Sinkhorn) distance for cosmological
#     halo point clouds with 3D position + 3D velocity, both standardized
#     (mean 0, std 1), so the ground cost is plain 6D Euclidean. No dependencies.

#     x_hat:  (B, M, 6) reconstructed halos  (x, y, z, vx, vy, vz)
#     x:      (B, N, 6) real halos
#     mask:        (B, N, 1) real-particle mask (1 keep, 0 padded)   [required]
#     mask_logits: (B, M)    optional predicted presence logits for x_hat slots.
#                            If None, x_hat is treated as uniform over M slots.

#     Ground cost: Euclidean distance in 6D (p=1). Returns mean-over-batch
#     entropic OT cost.

#     Memory strategy:
#       - the (B, M, N) cost matrix is never fully materialized; it is computed
#         in column chunks of width `chunk_size`.
#       - the Sinkhorn fixed-point iterations run under torch.no_grad(); a single
#         final differentiable update lets gradients flow to the inputs. This
#         avoids storing ~n_iter copies of intermediate (B, M, chunk) tensors.
#     """
#     B, M, F = x_hat.shape
#     N = x.shape[1]

#     mask = mask.squeeze(-1)                               # (B, N)

#     # --- keep coordinates finite (padded points may hold garbage) ---
#     x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
#     x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)

#     # --- measures ---
#     if mask_logits is not None:
#         w_hat = torch.sigmoid(mask_logits)               # (B, M)
#     else:
#         w_hat = torch.ones(B, M, device=x.device, dtype=x.dtype)
#     w = mask                                             # (B, N); padded -> 0

#     a = w_hat / (w_hat.sum(dim=1, keepdim=True) + eps)   # (B, M)
#     b = w     / (w.sum(dim=1, keepdim=True) + eps)       # (B, N)

#     log_a = torch.log(a + eps)                           # (B, M)
#     log_b = torch.log(b + eps)                           # (B, N)

#     # boolean mask for padded columns, used to neutralize them in the updates
#     pad = (mask == 0)                                     # (B, N)
#     log_b = log_b.masked_fill(pad, -1e9)

#     # --- chunked ground cost columns  C[:, :, j0:j1]  (B, M, c) ---
#     def cost_block(j0, j1):
#         yj = x[:, j0:j1, :]                               # (B, c, F)
#         yh2 = (x_hat ** 2).sum(-1, keepdim=True)          # (B, M, 1)
#         yj2 = (yj ** 2).sum(-1).unsqueeze(1)              # (B, 1, c)
#         cross = torch.bmm(x_hat, yj.transpose(1, 2))      # (B, M, c)
#         d2 = (yh2 + yj2 - 2 * cross).clamp_min(0.0)
#         return (d2 + 1e-12).sqrt()                        # (B, M, c), safe grad

#     # --- epsilon annealing schedule (multiscale) ---
#     eps_final = blur
#     if blur_start is None:
#         with torch.no_grad():
#             cmax = cost_block(0, min(N, 256)).max().item()
#         eps_start = max(blur, cmax * 0.5 + eps)
#     else:
#         eps_start = max(blur, blur_start)

#     def make_schedule(n):
#         if scaling <= 0 or scaling >= 1:
#             return [eps_final] * n
#         sched, e = [], eps_start
#         while e > eps_final:
#             sched.append(e)
#             e *= scaling
#         sched.append(eps_final)
#         if len(sched) < n:
#             sched += [eps_final] * (n - len(sched))
#         else:
#             sched = sched[:n]
#         return sched

#     schedule = make_schedule(n_iter)

#     # --- chunked log-domain dual updates ---
#     # f_i = -reg * logsumexp_j ( log_b_j + (g_j - C_ij)/reg )
#     def update_f(g, reg):
#         parts = []
#         for j0 in range(0, N, chunk_size):
#             j1 = min(N, j0 + chunk_size)
#             C = cost_block(j0, j1)                        # (B, M, c)
#             t = log_b[:, j0:j1].unsqueeze(1) + (g[:, j0:j1].unsqueeze(1) - C) / reg
#             parts.append(torch.logsumexp(t, dim=2, keepdim=True))   # (B, M, 1)
#             del C, t
#         lse = torch.logsumexp(torch.cat(parts, dim=2), dim=2)       # (B, M)
#         return -reg * lse

#     # g_j = -reg * logsumexp_i ( log_a_i + (f_i - C_ij)/reg )
#     def update_g(f, reg):
#         parts = []
#         for j0 in range(0, N, chunk_size):
#             j1 = min(N, j0 + chunk_size)
#             C = cost_block(j0, j1)                        # (B, M, c)
#             t = log_a.unsqueeze(2) + (f.unsqueeze(2) - C) / reg
#             parts.append(-reg * torch.logsumexp(t, dim=1))          # (B, c)
#             del C, t
#         g = torch.cat(parts, dim=1)                       # (B, N)
#         return g.masked_fill(pad, -1e9)

#     # --- initialize duals ---
#     f = torch.zeros_like(log_a)
#     g = torch.zeros_like(log_b).masked_fill(pad, -1e9)

#     # --- converge the duals WITHOUT building a graph (cheap memory) ---
#     with torch.no_grad():
#         for reg in schedule:
#             f = update_f(g, reg)
#             g = update_g(f, reg)

#     # --- one final DIFFERENTIABLE update so gradients reach the inputs ---
#     reg = schedule[-1]
#     f = update_f(g, reg)
#     g = update_g(f, reg)

#     # --- final transport cost, chunked; padded columns contribute ~0 ---
#     total = torch.zeros(B, device=x.device, dtype=x.dtype)
#     for j0 in range(0, N, chunk_size):
#         j1 = min(N, j0 + chunk_size)
#         C = cost_block(j0, j1)
#         log_P = (log_a.unsqueeze(2) + log_b[:, j0:j1].unsqueeze(1)
#                  + (f.unsqueeze(2) + g[:, j0:j1].unsqueeze(1) - C) / reg)
#         total = total + (log_P.exp() * C).sum(dim=(1, 2))
#         del C, log_P

#     return total.mean()


def wd_loss(
    x_hat,
    x,
    mask,
    mask_logits=None,
    n_clusters=64,
    cap=256,
    blur=0.01,
    blur_start=None,
    n_iter=25,
    scaling=0.5,
    eps=1e-8,
):
    """
    Clustered entropic Wasserstein reconstruction loss for point clouds.

    Points are assigned to real-cloud centroids by ABSOLUTE position, so a
    globally-misplaced reconstructed point is penalized (unlike per-cluster
    nearest-k gathering, which would erase global displacement). OT is then
    solved within each cluster. Memory is bounded by (B * n_clusters * cap^2)
    instead of (B * M * N), and the clustering makes the loss sensitive to
    local densities.

    x_hat:  (B, M, F)  reconstructed points
    x:      (B, N, F)  real points
    mask:        (B, N, 1) or (B, N)  real-point mask (1 keep, 0 pad)  [required]
    mask_logits: (B, M) or (B, M, 1)  presence logits for x_hat (optional)
    n_clusters:  number of centroids built from x
    cap:         max points per cluster per cloud (overflow dropped by distance)
    blur:        final entropic regularization (also the ground-cost scale)
    blur_start:  starting reg for annealing (auto if None)
    n_iter:      total Sinkhorn iterations
    scaling:     geometric annealing factor in (0, 1)

    Returns mean-over-live-groups entropic OT cost.
    """
    NEG = -1e4  # finite "log(0)" surrogate: safe under /reg and logsumexp

    B, M, F = x_hat.shape
    N = x.shape[1]

    # ---- normalize input shapes ----
    if mask.dim() == 3:
        mask = mask.squeeze(-1)  # (B, N)
    if mask_logits is not None and mask_logits.dim() == 3:
        mask_logits = mask_logits.squeeze(-1)  # (B, M)

    # ---- keep coordinates finite (padded slots may hold garbage) ----
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- presence / validity weights ----
    if mask_logits is not None:
        w_hat = torch.sigmoid(mask_logits)  # (B, M)
    else:
        w_hat = torch.ones(B, M, device=x.device, dtype=x.dtype)
    w_real = mask.to(x.dtype)  # (B, N); padded -> 0

    # ---- build centroids from x (detached, sampled from valid points) ----
    with torch.no_grad():
        K = min(n_clusters, N)
        probs = w_real + eps
        probs = probs / probs.sum(1, keepdim=True)
        seed = torch.multinomial(probs, K, replacement=True)  # (B, K)
        centroids = torch.gather(
            x, 1, seed.unsqueeze(-1).expand(-1, -1, F)
        )  # (B, K, F)

        # hard assignment by ABSOLUTE position
        d_x = torch.cdist(x, centroids)  # (B, N, K)
        d_hat = torch.cdist(x_hat, centroids)  # (B, M, K)
        d_x = d_x.masked_fill(mask.unsqueeze(-1) == 0, 1e9)
        asg_x = d_x.argmin(-1)  # (B, N)
        asg_hat = d_hat.argmin(-1)  # (B, M)

    # ---- pack each (batch, cluster) into a fixed-capacity slate ----
    # Overflow beyond `cap` is dropped, keeping the points closest to the
    # centroid. Dropped / empty slots carry zero weight.
    def pack(pts, w_pts, asg, dist):
        P = pts.shape[1]
        # distance of each point to its assigned centroid
        d_self = torch.gather(dist, 2, asg.unsqueeze(-1)).squeeze(-1)  # (B,P)
        # composite sort key: group first, then distance within group
        span = (d_self.max() + 1.0).detach()
        key = asg.to(pts.dtype) * span + d_self  # (B,P)
        order = key.argsort(dim=1)  # (B,P)

        asg_s = torch.gather(asg, 1, order)  # (B,P)
        w_s = torch.gather(w_pts, 1, order)  # (B,P)
        pts_s = torch.gather(pts, 1, order.unsqueeze(-1).expand(-1, -1, F))

        # rank within each cluster via per-cluster cumulative count
        onehot = torch.zeros(B, P, K, device=pts.device, dtype=pts.dtype)
        onehot.scatter_(2, asg_s.unsqueeze(-1), 1.0)
        rank = onehot.cumsum(1) - 1  # (B,P,K)
        rank = torch.gather(rank, 2, asg_s.unsqueeze(-1)).squeeze(-1).long()
        keep = rank < cap  # (B,P)
        rank = rank.clamp(max=cap - 1)

        slate = torch.zeros(B, K, cap, F, device=pts.device, dtype=pts.dtype)
        wt = torch.zeros(B, K, cap, device=pts.device, dtype=pts.dtype)
        bidx = torch.arange(B, device=pts.device).view(B, 1).expand(B, P)

        w_eff = w_s * keep.to(w_s.dtype)  # zero-weight the dropped overflow
        slate[bidx, asg_s, rank] = pts_s
        wt[bidx, asg_s, rank] = w_eff
        return slate.reshape(B * K, cap, F), wt.reshape(B * K, cap)

    Xr, Wr = pack(x, w_real, asg_x, d_x)  # (G, cap, F), (G, cap)
    Xh, Wh = pack(x_hat, w_hat, asg_hat, d_hat)

    # ---- normalize to probability measures per group ----
    # groups with zero total weight are "dead"; give them a dummy uniform
    # measure so arithmetic stays finite, then exclude them from the mean.
    sum_r = Wr.sum(1, keepdim=True)
    sum_h = Wh.sum(1, keepdim=True)
    live = (sum_r.squeeze(1) > 0) & (sum_h.squeeze(1) > 0)  # (G,)

    a = Wh / (sum_h + eps)  # source = reconstruction
    b = Wr / (sum_r + eps)  # target = real

    pad_a = Wh == 0
    pad_b = Wr == 0

    log_a = torch.log(a + eps).masked_fill(pad_a, NEG)
    log_b = torch.log(b + eps).masked_fill(pad_b, NEG)

    # ---- ground cost within each group (p=1 Euclidean) ----
    C = torch.cdist(Xh, Xr) + 1e-12  # (G,cap,cap)
    C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- epsilon-annealing schedule ----
    eps_final = max(blur, eps)
    if blur_start is None:
        with torch.no_grad():
            cmax = C.amax().item()
            if not (cmax == cmax) or cmax <= 0:  # NaN or nonpositive guard
                cmax = 1.0
        eps_start = max(eps_final, cmax * 0.5)
    else:
        eps_start = max(eps_final, blur_start)

    sched, e = [], eps_start
    if 0.0 < scaling < 1.0:
        while e > eps_final:
            sched.append(e)
            e *= scaling
    sched.append(eps_final)
    if len(sched) < n_iter:
        sched += [eps_final] * (n_iter - len(sched))
    else:
        sched = sched[:n_iter]

    # ---- log-domain Sinkhorn duals ----
    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)

    def step(reg):
        nonlocal f, g
        # f_i = -reg * logsumexp_j ( log_b_j + (g_j - C_ij)/reg )
        f = -reg * torch.logsumexp(
            log_b.unsqueeze(1) + (g.unsqueeze(1) - C) / reg, dim=2
        )
        # padded source slots have zero mass -> dual is irrelevant; pin it to 0
        # so a fully-padded logsumexp can never blow up the next half-step.
        f = f.masked_fill(pad_a, 0.0).clamp(-1e4, 1e4)

        # g_j = -reg * logsumexp_i ( log_a_i + (f_i - C_ij)/reg )
        g = -reg * torch.logsumexp(
            log_a.unsqueeze(2) + (f.unsqueeze(2) - C) / reg, dim=1
        )
        g = g.masked_fill(pad_b, 0.0).clamp(-1e4, 1e4)

    # converge duals without building a graph (memory-cheap)
    with torch.no_grad():
        for reg in sched:
            step(reg)

    # one final DIFFERENTIABLE update so gradients reach the inputs
    reg = sched[-1]
    step(reg)

    # ---- transport plan and cost ----
    log_P = (
        log_a.unsqueeze(2)
        + log_b.unsqueeze(1)
        + (f.unsqueeze(2) + g.unsqueeze(1) - C) / reg
    )  # (G,cap,cap)
    # dead groups: force ~zero transported mass so they contribute nothing
    log_P = log_P.masked_fill(~live.view(-1, 1, 1), NEG)

    P = log_P.exp()
    P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    cost = (P * C).sum(dim=(1, 2))  # (G,)

    cost = cost * live.to(cost.dtype)
    denom = live.to(cost.dtype).sum().clamp_min(1.0)
    loss = cost.sum() / denom

    # final belt-and-suspenders guard
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    return loss


def emd_loss(
    x_hat,
    x,
    mask,
    mask_logits,
    R=0.8,
    blur=0.015,
    blur_start=None,
    n_iter=100,
    n_iter_self=None,
    scaling=0.5,
    eps=1e-8,
):
    """
    Debiased Sinkhorn-divergence EMD loss with presence gating. No deps.

    x_hat:       (B, M, F) reconstructed particles
    x:           (B, N, F) real particles
    mask:        (B, N, 1) real-particle mask for x (1 = keep, 0 = padded)
    mask_logits: (B, M)    predicted presence logits for x_hat slots
    features assumed ordered (eta, phi, log pT, log E)

    Ground cost: Euclidean distance in (eta, phi)/R  (~ dR),  p=1.

    Returns mean over batch of:
        Sinkhorn divergence  S(a,b) = OT(a,b)
    """
    mask = mask.squeeze(-1)  # (B, N)
    p_hat = torch.sigmoid(mask_logits)  # (B, M)

    pos_hat = x_hat[..., 0:2] / R  # (B, M, 2)
    pos = x[..., 0:2] / R  # (B, N, 2)

    w_hat = x_hat[..., 2].exp() * p_hat  # (B, M)
    w = x[..., 2].exp() * mask  # (B, N)

    # --- normalize to probability measures (balanced OT) ---
    a = w_hat / (w_hat.sum(dim=1).unsqueeze(1) + eps)  # (B, M)
    b = w / (w.sum(dim=1).unsqueeze(1) + eps)  # (B, N)

    # --- cost matrices (p=1 Euclidean in (eta,phi)/R) ---
    C_ab = torch.cdist(pos_hat, pos, p=2)  # (B, M, N)

    # --- log-domain measures; hard-mask padded real particles ---
    log_a = torch.log(a + eps)  # (B, M)
    log_b = torch.log(b + eps)  # (B, N)
    log_b = log_b.masked_fill(mask == 0, -1e9)  # (B, N)

    # --- epsilon annealing schedule (multiscale), shared across solves ---
    eps_final = blur
    if blur_start is None:
        cmax = C_ab.detach().max().item()
        eps_start = max(blur, cmax * 0.5 + eps)
    else:
        eps_start = max(blur, blur_start)

    def make_schedule(n):
        if scaling <= 0 or scaling >= 1:
            return [eps_final] * n
        sched, e = [], eps_start
        while e > eps_final:
            sched.append(e)
            e *= scaling
        sched.append(eps_final)
        if len(sched) < n:
            sched += [eps_final] * (n - len(sched))
        else:
            sched = sched[:n]
        return sched

    schedule = make_schedule(n_iter)

    def sinkhorn_cost(log_p, log_q, C, sched):
        """Log-domain Sinkhorn; returns entropic OT cost <P, C> per batch."""
        f = torch.zeros_like(log_p)  # dual for rows
        g = torch.zeros_like(log_q)  # dual for cols
        for reg in sched:
            M_g = (g.unsqueeze(1) - C) / reg  # (B, |p|, |q|)
            f = -reg * torch.logsumexp(log_q.unsqueeze(1) + M_g, dim=2)
            M_f = (f.unsqueeze(2) - C) / reg  # (B, |p|, |q|)
            g = -reg * torch.logsumexp(log_p.unsqueeze(2) + M_f, dim=1)
        reg = sched[-1]
        log_P = (
            log_p.unsqueeze(2)
            + log_q.unsqueeze(1)
            + (f.unsqueeze(2) + g.unsqueeze(1) - C) / reg
        )
        P = log_P.exp()
        return (P * C).sum(dim=(1, 2))  # (B,)

    loss = sinkhorn_cost(log_a, log_b, C_ab, schedule)  # (B,)
    return loss.mean()


def chamfer_loss(x_hat, x, mask):
    """
    x_hat: (B, M, F) reconstructed/generated particles
    x:     (B, N, F) real particles
    mask:  (B, N, 1) real-particle mask for x
    """

    mask = mask.squeeze(-1).bool()

    # pairwise squared distances: (B, M, N)
    dist = torch.cdist(x_hat, x) ** 2

    # ignore padded real particles
    dist = dist.masked_fill(~mask[:, None, :], float("inf"))

    # each generated point should match some real point
    loss_hat_to_x = dist.min(dim=2).values.mean()

    # each real point should be matched by some generated point
    loss_x_to_hat = dist.min(dim=1).values
    loss_x_to_hat = loss_x_to_hat[mask].mean()

    return loss_hat_to_x + loss_x_to_hat


def get_loss(
    outputs,
    y,
    mode,
    class_cost,
    gen_cost,
    use_event_loss,
    use_clip,
    clip_loss,
    logs,
    data_pid=None,
):
    loss = 0.0
    if outputs["y_pred"] is not None:
        if mode == "regression":
            loss_class = torch.mean(class_cost(outputs["y_pred"], y))
            logs["loss_class"] += loss_class.detach()
        else:
            counts = torch.bincount(
                y.int(), minlength=outputs["y_pred"].shape[-1]
            ).float()
            class_weights = 1.0 / (counts + 1e-6)
            weights = class_weights[y.int()]
            weights = weights / weights.mean()

            loss_class = get_class_loss(
                # torch.ones_like(y),
                weights,
                outputs["y_pred"],
                y.long(),
                class_cost,
                use_event_loss,
                logs,
            )

        loss = loss + loss_class

    if outputs["z_pred"] is not None:
        if data_pid is not None:
            if mode == "segmentation":
                data_pid = data_pid.reshape((-1, data_pid.shape[-1]))
                loss_gen = torch.mean(
                    gen_cost(
                        outputs["z_pred"].reshape((-1, outputs["z_pred"].shape[-1])),
                        data_pid,
                    )
                )
            else:
                data_pid = data_pid.reshape((-1))
                mask = data_pid != -1  # -1 is associated to zero-pdded entries
                loss_gen = torch.mean(
                    gen_cost(
                        outputs["z_pred"].reshape((-1, outputs["z_pred"].shape[-1]))[
                            mask
                        ],
                        data_pid[mask],
                    )
                )
        else:
            nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
            loss_gen = outputs["v_weight"] * gen_cost(outputs["v"], outputs["z_pred"])
            loss_gen = loss_gen.sum((1, 2)) / nonzero
            loss_gen = loss_gen.mean()

        logs["loss_gen"] += loss_gen.detach()
        loss = loss + loss_gen

    if outputs["y_perturb"] is not None and data_pid is None:
        counts = torch.bincount(y.int(), minlength=outputs["y_pred"].shape[-1]).float()
        class_weights = 1.0 / (counts + 1e-6)
        weights = class_weights[y.int()]
        weights = outputs["alpha"].squeeze() * weights / weights.mean()
        loss = loss + get_class_loss(
            weights, outputs["y_perturb"], y.long(), class_cost, use_event_loss, logs
        )

    if outputs["x_hat"] is not None:
        target_mask = (outputs["x"][..., 2:3] != 0).float()
        if outputs["x"].shape[1] > 200:
            loss_ot = wd_loss(
                outputs["x_hat"], outputs["x"], target_mask, outputs["mask_logits"]
            )
        else:
            loss_ot = emd_loss(
                outputs["x_hat"], outputs["x"], target_mask, outputs["mask_logits"]
            )

        loss_mask = F.binary_cross_entropy_with_logits(
            outputs["mask_logits"],
            target_mask.squeeze(-1),
        )
        loss_ae = loss_ot + loss_mask
        logs["loss_ae"] += loss_ae.detach()
        loss = loss + loss_ae

    if use_clip and outputs["z_body"] is not None and outputs["x_body"] is not None:
        loss_clip = clip_loss(
            outputs["x_body"].view(outputs["x_body"].shape[0], -1),
            outputs["z_body"].view(outputs["x_body"].shape[0], -1),
            weight=outputs["alpha"],
        )
        loss = loss + loss_clip
        logs["loss_clip"] += loss_clip.detach()

    logs["loss"] += loss.detach()
    return loss


def save_checkpoint(
    model,
    ema_model,
    epoch,
    optimizer,
    loss,
    lr_scheduler,
    checkpoint_dir,
    checkpoint_name,
    best_loss=None,
    best_epoch=None,
):
    save_dict = {
        "body": model.module.body.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "sched": lr_scheduler.state_dict(),
    }
    if best_loss is not None:
        save_dict["best_loss"] = best_loss
    if best_epoch is not None:
        save_dict["best_epoch"] = best_epoch

    if model.module.classifier is not None:
        save_dict["classifier_head"] = model.module.classifier.state_dict()

    if model.module.generator is not None:
        save_dict["generator_head"] = model.module.generator.state_dict()
    if ema_model is not None:
        save_dict["ema_body"] = ema_model.body.state_dict()
        if model.module.generator is not None:
            save_dict["ema_generator"] = ema_model.generator.state_dict()

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    torch.save(save_dict, os.path.join(checkpoint_dir, checkpoint_name))
    print(
        f"Epoch {epoch} | Training checkpoint saved at {os.path.join(checkpoint_dir, checkpoint_name)}"
    )


def restore_checkpoint(
    model,
    checkpoint_dir,
    checkpoint_name,
    device,
    is_main_node=False,
    restore_ema_model=False,
    ema_model=None,
    fine_tune=False,
    optimizer=None,
    lr_scheduler=None,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"

    if fine_tune and not os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)):
        print(f"Fetching pretrained checkpoint {checkpoint_name}")
        file_url = f"https://portal.nersc.gov/cfs/m4567/checkpoints/{checkpoint_name}"
        file_path = os.path.join(checkpoint_dir, checkpoint_name)
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {checkpoint_name}")

    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)

    if restore_ema_model:
        body_name = "ema_body"
        generator_name = "ema_generator"
    else:
        body_name = "body"
        generator_name = "generator_head"

    if not fine_tune:
        base_model.body.load_state_dict(checkpoint[body_name], strict=False)

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            base_model.classifier.load_state_dict(
                checkpoint["classifier_head"], strict=True
            )

        if base_model.generator is not None:
            base_model.generator.load_state_dict(
                checkpoint[generator_name], strict=True
            )

        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["sched"])
        # checkpoint["epoch"] is already the count of epochs completed
        # (save_checkpoint is called with epoch + 1), so it is exactly the
        # 0-indexed epoch to resume at.
        startEpoch = checkpoint["epoch"]
        # best_loss / best_epoch are written only by the every-epoch
        # last_model_ checkpoint; older best_model_ checkpoints predate them,
        # so fall back to what those do carry.
        best_loss = checkpoint.get("best_loss", checkpoint["loss"])
        best_epoch = checkpoint.get("best_epoch", checkpoint["epoch"] - 1)

    else:

        def filter_partial_model(state, model_state, is_main_node=False):
            filtered = {}

            for k, v in state.items():
                if "out." in k:
                    if is_main_node:
                        print(f"Skipping {k}: explicitly excluded")
                elif k not in model_state:
                    if is_main_node:
                        print(f"Skipping {k}: not present in new model")
                elif v.shape != model_state[k].shape:
                    if is_main_node:
                        print(f"Skipping {k}: {v.shape} -> {model_state[k].shape}")
                else:
                    filtered[k] = v

            if is_main_node:
                for k in model_state.keys() - filtered.keys():
                    if k not in state:
                        print(f"Random {k}: not present in checkpoint")

                total = sum(v.numel() for v in model_state.values())
                loaded = sum(model_state[k].numel() for k in filtered)
                print(f"Loaded: {loaded:,}/{total:,} ({100 * loaded / total:.2f}%)")
                print(
                    f"Random: {total - loaded:,}/{total:,} ({100 * (total - loaded) / total:.2f}%)"
                )

            return filtered

        if base_model.body is not None and "body" in checkpoint:
            filtered_state = filter_partial_model(
                checkpoint["body"], base_model.body.state_dict(), is_main_node
            )
            base_model.body.load_state_dict(filtered_state, strict=False)

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            filtered_state = filter_partial_model(
                checkpoint["classifier_head"],
                base_model.classifier.state_dict(),
                is_main_node,
            )
            base_model.classifier.load_state_dict(filtered_state, strict=False)

        if base_model.generator is not None:
            filtered_state = filter_partial_model(
                checkpoint["generator_head"],
                base_model.generator.state_dict(),
                is_main_node,
            )
            base_model.generator.load_state_dict(filtered_state, strict=False)

        startEpoch = 0.0
        best_loss = np.inf
        best_epoch = 0.0

    if ema_model is not None:
        if fine_tune:
            ema_model.load_state_dict(base_model.state_dict())
        elif "ema_body" in checkpoint:
            ema_model.body.load_state_dict(checkpoint["ema_body"], strict=True)

            if base_model.generator is not None:
                ema_model.generator.load_state_dict(
                    checkpoint["ema_generator"], strict=True
                )

    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            if is_main_node:
                print("Optimizer cannot be loaded back, skipping...")

    return startEpoch, best_loss, best_epoch


def shadow_copy(model):
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def gather_tensors(x):
    """
    If running under DDP, all_gather x from every rank, concat, then return as numpy.
    Otherwise just .cpu().numpy().
    """
    if dist.is_initialized():
        ws = dist.get_world_size()
        # pre‐allocate one buffer per rank
        buf = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(buf, x)
        x = torch.cat(buf, dim=0)
    return x.cpu()


def get_param_groups(model, wd, lr, lr_factor=1.0, fine_tune=False, freeze=False):
    no_decay, decay = [], []
    new_layer_no_decay, new_layer_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_new_layer = name.startswith(
            (
                "classifier.out",
                "generator",
                "body.cond",
                "body.add_embed",
                "body.local_physics",
                "body.embed",
                "body.interaction",
                # "body.token",
            )
        )

        if any(keyword in name for keyword in model.no_weight_decay()):
            if is_new_layer:
                new_layer_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if is_new_layer:
                new_layer_decay.append(param)
            else:
                decay.append(param)
    # Base learning rate groups
    param_groups = [
        {"params": decay, "weight_decay": wd, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]

    # Adjust learning rate for new layer if fine-tuning
    new_layer_lr = lr * lr_factor if fine_tune else lr

    if new_layer_decay:
        param_groups.append(
            {"params": new_layer_decay, "weight_decay": wd, "lr": new_layer_lr}
        )
    if new_layer_no_decay:
        param_groups.append(
            {"params": new_layer_no_decay, "weight_decay": 0.0, "lr": new_layer_lr}
        )

    if fine_tune and freeze:
        # Freeze body parts but input embeddings
        for name, param in model.body.named_parameters():
            if (
                name.startswith("embed.")
                or name.startswith("local_physics.")
                or name.startswith("cond.")
                or name.startswith("add_embed.")
                or name.startswith("token")
            ):
                continue
            else:
                param.requires_grad = False

    return param_groups


def get_checkpoint_name(tag):
    return f"best_model_{tag}.pt"


def get_last_checkpoint_name(tag):
    return f"last_model_{tag}.pt"


def is_master_node():
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    else:
        return True


def ddp_setup():
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2900"
        os.environ["RANK"] = "0"
        init_process_group(rank=0, world_size=1)
        rank = local_rank = 0
    else:
        init_process_group(init_method="env://")
        # overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    return local_rank, rank, dist.get_world_size()
