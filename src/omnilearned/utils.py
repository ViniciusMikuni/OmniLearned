import math
import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


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


class CLIPLoss(nn.Module):
    def get_logits(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        logit_scale: float,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        clean_features = F.normalize(clean_features, dim=-1, eps=1e-3)

        perturbed_features = F.normalize(perturbed_features, dim=-1, eps=1e-3)

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
        logits_per_clean, logits_per_perturbed = self.get_logits(
            clean_features, perturbed_features, logit_scale
        )

        labels = torch.arange(
            logits_per_clean.shape[0], device=clean_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_clean, labels, reduction="none")
            + F.cross_entropy(logits_per_perturbed, labels, reduction="none")
        ) / 2
        if weight is not None:
            total_loss = torch.mean(weight.reshape(-1) * total_loss)
        else:
            total_loss = total_loss.mean()
        return {"contrastive_loss": total_loss} if output_dict else total_loss


def emd_loss(
    x_hat,
    x,
    mask,
    mask_logits,
    R=0.8,
    blur=0.015,
    blur_start=None,
    n_iter=10,
    scaling=0.5,
    wrap_phi=False,
    unroll=True,
    eps=1e-8,
):
    """Angular, normalized-pT Sinkhorn transport cost.

    Differentiates every iteration by default, with checkpoint recomputation.
    Inputs use (eta, phi, log pT, ...); other features do not enter this term.
    """
    if x_hat.dim() != 3 or x.dim() != 3:
        raise ValueError("x_hat and x must be (B, *, F)")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")

    MIN_MASS = 1e-12
    NEG = -1e9  # log-space "zero"; safe in fp32, do not use this loss in fp16

    mask = mask.squeeze(-1).to(x.dtype)  # (B, N)

    valid_jet = mask.sum(dim=1) > 0  # (B,)
    mask_eff = torch.where(valid_jet.unsqueeze(1), mask, torch.ones_like(mask))

    p_hat = torch.sigmoid(mask_logits)  # (B, M)

    eta = torch.where(mask.bool(), x[..., 0], 0.0)
    phi = torch.where(mask.bool(), x[..., 1], 0.0)
    eta_hat, phi_hat = x_hat[..., 0], x_hat[..., 1]

    log_pt = torch.where(
        mask.bool(), x[..., 2], 0.0
    )  # padded -> exp(0) = 1, killed by mask_eff below
    w = log_pt.exp() * mask_eff  # (B, N)
    w_hat = x_hat[..., 2].exp() * p_hat  # (B, M)

    def _normalize(v, support):
        tot = v.sum(dim=1, keepdim=True)
        degenerate = tot <= MIN_MASS
        n_sup = support.sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform = support / n_sup
        return torch.where(degenerate, uniform, v / tot.clamp_min(MIN_MASS))

    a = _normalize(w_hat, torch.ones_like(w_hat))  # (B, M)
    b = _normalize(w, mask_eff)  # (B, N)

    d_eta = eta_hat.unsqueeze(2) - eta.unsqueeze(1)  # (B, M, N)
    d_phi = phi_hat.unsqueeze(2) - phi.unsqueeze(1)  # (B, M, N)
    if wrap_phi:
        d_phi = torch.remainder(d_phi + math.pi, 2.0 * math.pi) - math.pi
    C = torch.sqrt((d_eta * d_eta + d_phi * d_phi) / (R * R) + 1e-12)  # (B,M,N)

    log_a = torch.log(a + eps)
    log_b = torch.log(b + eps).masked_fill(mask_eff == 0, NEG)

    return _transport_cost(
        C, log_a, log_b, valid_jet, blur, blur_start, n_iter, scaling, unroll
    )


def sliced_wasserstein_loss(
    x_hat,
    x,
    mask,
    mask_logits,
    n_projections=32,
    projections=None,
):
    """
    Weighted sliced Wasserstein-1 reconstruction loss.

    Args:
        x_hat: Predicted point cloud, (B, M, D).
        x: Target point cloud, (B, N, D).
        mask: Valid target particles, (B, N) or (B, N, 1).
        mask_logits: Predicted presence logits, (B, M) or (B, M, 1).
        n_projections: Number of random projection directions.
        projections: Optional fixed directions, (D, K).

    Returns:
        Scalar loss averaged over directions and nonempty events.

    Each projected weighted W1 distance is computed exactly.
    The average over finitely many directions approximates sliced W1.
    """
    B, M, D = x_hat.shape
    N = x.shape[1]

    if n_projections < 1:
        raise ValueError("n_projections must be positive")

    mask = mask.reshape(B, N).bool()
    logits = mask_logits.reshape(B, M)
    nonempty = mask.any(dim=1)

    if M == 0 or N == 0 or not nonempty.any():
        return x_hat.sum() * 0.0 + logits.sum() * 0.0

    with torch.autocast(device_type=x_hat.device.type, enabled=False):
        dtype = torch.float64 if x_hat.dtype == torch.float64 else torch.float32

        pred = x_hat[nonempty].to(dtype)
        valid = mask[nonempty]
        target = x[nonempty].to(dtype)
        target = torch.where(valid[..., None], target, 0.0)
        logits = logits[nonempty].to(dtype)

        # Normalize sigmoid probabilities without numerical underflow.
        pred_weights = F.logsigmoid(logits).softmax(dim=1)
        target_weights = valid.to(dtype)
        target_weights = target_weights / target_weights.sum(dim=1, keepdim=True)

        if projections is None:
            directions = torch.randn(D, n_projections, device=pred.device, dtype=dtype)
        else:
            directions = projections.to(device=pred.device, dtype=dtype)
            if (
                directions.ndim != 2
                or directions.shape[0] != D
                or directions.shape[1] == 0
            ):
                raise ValueError("projections must have shape (D, K), K > 0")

        norms = directions.norm(dim=0, keepdim=True)
        if (norms == 0).any():
            raise ValueError("Projection directions must be nonzero")
        directions = directions / norms

        # Merge the projected supports of the two distributions.
        positions = torch.cat((pred @ directions, target @ directions), dim=1)
        signed_weights = torch.cat((pred_weights, -target_weights), dim=1)
        signed_weights = signed_weights[..., None].expand_as(positions)

        positions, order = positions.sort(dim=1)
        signed_weights = signed_weights.gather(1, order)

        # W1 = integral of the absolute difference between the two CDFs.
        cdf_difference = signed_weights.cumsum(dim=1)[:, :-1]
        interval_widths = positions[:, 1:] - positions[:, :-1]

        return (cdf_difference.abs() * interval_widths).sum(dim=1).mean()


def _transport_cost(
    C, log_a, log_b, valid_jet, blur, blur_start, n_iter, scaling, unroll
):
    """Shared annealed Sinkhorn solve; returns the transport part of the cost."""
    eps_final = float(blur)
    if blur_start is None:
        with torch.no_grad():
            cmax = C.max().item()
        eps_start = max(eps_final, 0.5 * cmax)
    else:
        eps_start = max(eps_final, float(blur_start))

    def make_schedule(n):
        if eps_start <= eps_final or not (0.0 < scaling < 1.0):
            return [eps_final] * n
        n_anneal = math.ceil(math.log(eps_final / eps_start) / math.log(scaling))
        n_anneal = max(1, min(int(n_anneal), n))
        ratio = (eps_final / eps_start) ** (1.0 / n_anneal)
        sched = [eps_start * ratio**k for k in range(1, n_anneal + 1)]
        sched[-1] = eps_final  # exact, no float drift
        return sched + [eps_final] * (n - n_anneal)

    schedule = make_schedule(n_iter)

    def softmin_rows(g_, reg):  # f_i = -reg * log sum_j b_j exp((g_j - C_ij)/reg)
        return -reg * torch.logsumexp(
            log_b.unsqueeze(1) + (g_.unsqueeze(1) - C) / reg, dim=2
        )

    def softmin_cols(f_, reg):  # g_j = -reg * log sum_i a_i exp((f_i - C_ij)/reg)
        return -reg * torch.logsumexp(
            log_a.unsqueeze(2) + (f_.unsqueeze(2) - C) / reg, dim=1
        )

    def solve(f_, g_):
        for reg in schedule:
            f_ = softmin_rows(g_, reg)  # Gauss-Seidel: g then fresh f
            g_ = softmin_cols(f_, reg)
        return f_, g_

    f = torch.zeros_like(log_a)
    g = torch.zeros_like(log_b)
    reg = schedule[-1]

    if unroll:
        if torch.is_grad_enabled() and any(t.requires_grad for t in (C, log_a, log_b)):
            f, g = checkpoint(solve, f, g, use_reentrant=False)
        else:
            f, g = solve(f, g)
    else:
        with torch.no_grad():
            f, g = solve(f, g)
        g_d = g.detach()
        f = softmin_rows(g_d, reg)
        g = softmin_cols(f, reg)

    log_P = (
        log_a.unsqueeze(2)
        + log_b.unsqueeze(1)
        + (f.unsqueeze(2) + g.unsqueeze(1) - C) / reg
    )
    cost = (log_P.exp() * C).sum(dim=(1, 2))  # (B,)

    cost = torch.where(valid_jet, cost, torch.zeros_like(cost))
    return cost.sum() / valid_jet.sum().clamp_min(1)


def shadow_copy(model):
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def get_checkpoint_name(tag):
    return f"best_model_{tag}.pt"


def ddp_setup():
    local = int(os.environ.get("LOCAL_RANK", "0"))
    rank, size = (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    return local, rank, size


def get_param_groups(model, wd, lr, lr_factor=1.0, fine_tune=False, freeze=False):
    groups = {}
    for name, p in model.named_parameters():
        if freeze and name.startswith("body."):
            keep_trainable = name.startswith(
                (
                    "body.embed.",
                    "body.local_physics.",
                    "body.cond_embed.",
                    "body.add_embed.",
                    "body.token",
                )
            )

            if not keep_trainable:
                p.requires_grad = False
        if p.requires_grad:
            decay = wd if p.ndim > 1 and "token" not in name else 0.0
            is_new_layer = name.startswith(
                (
                    "classifier.out",
                    "generator",
                    "body.cond",
                    "body.add_embed",
                    "body.local_physics",
                    "body.embed",
                    "body.interaction",
                )
            )
            rate = lr * lr_factor if fine_tune and is_new_layer else lr
            groups.setdefault((decay, rate), []).append(p)
    return [{"params": ps, "weight_decay": d, "lr": r} for (d, r), ps in groups.items()]


def get_loss(
    outputs,
    y,
    mode,
    class_cost=None,
    gen_cost=None,
    use_event_loss=False,
    use_clip=False,
    clip_loss=None,
    logs=None,
    data_pid=None,
):
    terms, zero = {}, outputs["x"].sum() * 0

    def class_loss(pred, weight=None):
        if "regression" in mode:
            per = (pred.float() - y.reshape_as(pred)).square().flatten(1).mean(1)
        else:
            target = y.long().reshape(-1)
            balance = (
                torch.bincount(target, minlength=pred.shape[-1])
                .float()
                .clamp_min(1)
                .reciprocal()[target]
            )
            balance = balance / balance.mean()
            if use_event_loss:
                per = torch.zeros(len(y), device=pred.device, dtype=torch.float32)
                for select, start, stop in (
                    (target < 200, 0, 200),
                    (target >= 200, 200, pred.shape[-1]),
                ):
                    if select.any():
                        per[select] = F.cross_entropy(
                            pred[select, start:stop],
                            target[select] - start,
                            reduction="none",
                        )
            else:
                per = F.cross_entropy(pred, target, reduction="none")
            per *= balance
        return (per if weight is None else per * weight.reshape(-1)).mean()

    if outputs.get("y_pred") is not None:
        terms["loss_class"] = class_loss(outputs["y_pred"])
    if outputs.get("y_perturb") is not None:
        terms["loss_perturbed"] = class_loss(
            outputs["y_perturb"], None if "regression" in mode else outputs["alpha"]
        )
    mask = outputs["mask"].squeeze(-1)
    if outputs.get("z_pred") is not None:
        pred = outputs["z_pred"]
        if data_pid is not None:
            if mode == "ftag":
                target = data_pid.reshape(mask.shape).long()
                live = mask & (target != -1)
                terms["loss_gen"] = (
                    F.cross_entropy(pred[live], target[live])
                    if live.any()
                    else pred.sum() * 0
                )
            else:
                target = data_pid.reshape_as(pred)
                err = (
                    (
                        torch.where(mask.unsqueeze(-1), pred.float(), 0.0)
                        - torch.where(mask.unsqueeze(-1), target, 0.0)
                    )
                    .square()
                    .mean(-1)
                )
                terms["loss_gen"] = (
                    (err * mask).sum(1) / mask.sum(1).clamp_min(1)
                ).mean()
        else:
            err = outputs["v_weight"] * (pred.float() - outputs["v"]).square()
            terms["loss_gen"] = (
                (err * mask.unsqueeze(-1)).sum((1, 2)) / mask.sum(1).clamp_min(1)
            ).mean()
    if outputs.get("x_hat") is not None:
        counts = mask.sum(1)
        with torch.autocast(device_type=outputs["x"].device.type, enabled=False):
            args = (
                outputs["x_hat"].float(),
                outputs["x"].float(),
                mask.unsqueeze(-1),
                outputs["mask_logits"].float(),
            )
            reconstruction_emd = emd_loss(*args)
            reconstruction_ot = sliced_wasserstein_loss(*args)
        reconstruction = reconstruction_emd + reconstruction_ot
        if logs is not None:
            logs["loss_emd"] = reconstruction_emd.detach()
            logs["loss_ot"] = reconstruction_ot.detach()
        logits = outputs["mask_logits"]
        if (counts > logits.shape[1]).any():
            raise ValueError(
                "num_part must accommodate every real particle in encoder mode"
            )
        target = (
            torch.arange(logits.shape[1], device=logits.device)[None] < counts[:, None]
        )
        terms["loss_ae"] = reconstruction + F.binary_cross_entropy_with_logits(
            logits, target.float()
        )
    if (
        use_clip
        and outputs.get("x_body") is not None
        and outputs.get("z_body") is not None
    ):
        terms["loss_clip"] = (clip_loss or CLIPLoss())(
            outputs["x_body"].flatten(1), outputs["z_body"].flatten(1), outputs["alpha"]
        )

    loss = sum(terms.values(), zero)
    terms["loss"] = loss
    if logs is not None:
        for key, value in terms.items():
            logs[key] = logs.get(key, 0.0) + value.detach()
    return loss
