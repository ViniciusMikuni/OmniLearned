import einops
import torch
import torch.nn.functional as F
from torch import nn

from omnilearned.attention import Attention
from omnilearned.neighbors import knn


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last=True, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight
        else:
            x = x * self.weight[:, None, None]
        return x

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"


class NoScaleDropout(nn.Module):
    """
    Dropout without rescaling.
    """

    def __init__(self, rate: float) -> None:
        super().__init__()
        self.rate = rate

    def forward(self, x: torch.Tensor, offsets=None) -> torch.Tensor:
        if not self.training or self.rate == 0:
            return x
        else:
            mask_shape = (x.shape[0] if offsets is None else len(offsets) - 1,) + (
                1,
            ) * (x.ndim - 1)
            mask = torch.empty(mask_shape, device=x.device).bernoulli_(1 - self.rate)
            if offsets is not None:
                mask = torch.repeat_interleave(mask, offsets[1:] - offsets[:-1], dim=0)
            return x * mask


def get_mass(xi, xj, mask, is_log=False):
    m2 = (
        2
        * torch.exp(xi[:, :, :, 2])
        * torch.exp(xj[:, :, :, 2])
        * (
            torch.cosh(xi[:, :, :, 0] - xj[:, :, :, 0])
            - torch.cos(xi[:, :, :, 1] - xj[:, :, :, 1])
        )
    )
    if is_log:
        return torch.log(torch.where(m2 > 0, m2, 1.0)).unsqueeze(-1) * mask
    else:
        return torch.sqrt(m2).unsqueeze(-1) * mask


def get_dr(xi, xj, mask, is_log=True):
    d_eta = xi[:, :, :, 0] - xj[:, :, :, 0]
    d_phi = xi[:, :, :, 1] - xj[:, :, :, 1]
    dr2 = d_eta.square() + d_phi.square()

    if is_log:
        return 0.5 * torch.log(torch.where(dr2 > 0, dr2, 1.0)).unsqueeze(-1) * mask
    else:
        return torch.sqrt(dr2).unsqueeze(-1) * mask


def get_d3(xi, xj, mask, is_log=True):
    dx = xi[:, :, :, 0] - xj[:, :, :, 0]
    dy = xi[:, :, :, 1] - xj[:, :, :, 1]
    dz = xi[:, :, :, 2] - xj[:, :, :, 2]

    dr2 = dx.square() + dy.square() + dz.square()

    if is_log:
        return 0.5 * torch.log(torch.where(dr2 > 0, dr2, 1.0)).unsqueeze(-1) * mask
    else:
        return torch.sqrt(dr2).unsqueeze(-1) * mask


def get_dot(xi, xj, mask, is_log=True):
    dot = (xi[:, :, :, :3] * xj[:, :, :, :3]).sum(dim=-1)
    dr2 = 1.0 - dot / (
        xi[:, :, :, :3].norm(dim=-1) * xj[:, :, :, :3].norm(dim=-1) + 1e-8
    )

    if is_log:
        return torch.log(torch.where(dr2 > 0, dr2, 1.0)).unsqueeze(-1) * mask
    else:
        return dr2.unsqueeze(-1) * mask


def get_dot_diff(xi, xj, mask, is_log=True):
    dot = (xi[:, :, :, :3] * (xi[:, :, :, :3] - xj[:, :, :, :3])).sum(dim=-1)
    dr2 = 1.0 - dot / (
        xi[:, :, :, :3].norm(dim=-1) * (xi[:, :, :, :3] - xj[:, :, :, :3]).norm(dim=-1)
        + 1e-8
    )

    if is_log:
        return torch.log(torch.where(dr2 > 0, dr2, 1.0)).unsqueeze(-1) * mask
    else:
        return dr2.unsqueeze(-1) * mask


def get_kt(xi, xj, mask, is_log=False):
    kt = torch.min(
        torch.stack([torch.exp(xi[:, :, :, 2:3]), torch.exp(xj[:, :, :, 2:3])], -1), -1
    )[0] * get_dr(xi, xj, mask, is_log=False)
    if is_log:
        return torch.log(torch.where(kt > 0, kt, 1.0)) * mask
    else:
        return kt

def _ot_block(left, right, la, lb, epsilon, n_iters, gram=None):
    if gram is None:
        cost = (1 - torch.einsum("bikf,bjlf->bijkl", left, right)).clamp_min(0)
    else:
        batch = torch.arange(gram.shape[0], device=gram.device)[
            :, None, None, None, None
        ]
        cost = (
            1 - gram[batch, left[:, :, None, :, None], right[:, None, :, None, :]]
        ).clamp_min(0)

    kernel = -cost / epsilon
    u = torch.zeros_like(kernel[..., :1])
    v = torch.zeros_like(kernel[..., :1, :])

    for _ in range(n_iters):
        u = la[:, :, None, :, None] - torch.logsumexp(
            kernel + v, dim=-1, keepdim=True
        )
        v = lb[:, None, :, None, :] - torch.logsumexp(
            kernel + u, dim=-2, keepdim=True
        )

    plan = (kernel + u + v).exp()
    rows = plan.sum(dim=-1)
    cols = plan.sum(dim=-2)

    return epsilon * (
        (u.squeeze(-1) * rows).sum(-1)
        + (v.squeeze(-2) * cols).sum(-1)
        - rows.sum(-1)
    )

def _debias(distance):
    """Symmetric, self-debiased entropic OT; clip finite-iteration negatives."""
    distance = 0.5 * (distance + distance.transpose(1, 2))
    diagonal = distance.diagonal(dim1=1, dim2=2)
    distance = distance - 0.5 * (diagonal[:, :, None] + diagonal[:, None, :])
    return distance.clamp_min(0)


def get_ot(
    x,
    mask,
    indices,
    n_iters=5,
    epsilon=1.0,
    is_log=True,
    chunk_size=128,
    cache_cosines=True,
):
    """Return cosine Sinkhorn divergences: full, relative, coordinates.

    Shape is (B,N,N,3). All three
    channels use the entropic objective <P,C> + epsilon*sum(P*(log(P)-1)),
    symmetrize, and subtract the mean self-objective using existing diagonals.
    With finite n_iters these are approximate divergences; negative estimates
    are clamped to zero before the optional logarithm. No extra OT solves.

    indices: flattened batch-offset kNN indices, as returned by this package.
    x[..., :3]: Cartesian positions used by the coordinate-only third channel.
    cache_cosines: reuse a B*N*N cosine matrix for the absolute channel.
    All channels use uniform valid-neighbor weights, including unequal counts.
    epsilon and n_iters apply to all three channels; no radial W1 is computed.
    is_log applies the existing log(clamp_min(1e-12)) to all three channels.
    Uses eager PyTorch only; no torch.compile or POT dependency.
    """
    b, n, features = x.shape
    if chunk_size < 1 or epsilon <= 0 or n_iters < 0:
        raise ValueError("Require chunk_size >= 1, epsilon > 0, n_iters >= 0")
    if features < 3:
        raise ValueError("The first three features must be Cartesian positions")
    if n == 0:
        return x.new_zeros((b, 0, 0, 3))
    idx = indices.reshape(b, n, -1)
    k = idx.shape[-1]
    if k == 0:
        raise ValueError("At least one neighbor slot is required")
    mask = mask.reshape(b, n, 1).bool()
    centers = torch.arange(b * n, device=x.device).reshape(b, n, 1)
    valid = mask.reshape(-1)[idx] & (idx != centers) & mask
    weights = valid.float()
    live = valid.any(-1)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
    weights = torch.where(live[..., None], weights, 1 / k)
    log_weights = weights.clamp_min(1e-30).log()
    local_idx = idx - torch.arange(b, device=x.device)[:, None, None] * n
    needs_grad = torch.is_grad_enabled() and x.requires_grad
    chunks = [slice(i, i + chunk_size) for i in range(0, n, chunk_size)]

    def distance(values, gram=None):
        rows = []
        for i in chunks:
            columns = []
            for j in chunks:
                args = (
                    values[:, i], values[:, j], log_weights[:, i], log_weights[:, j],
                    epsilon, n_iters, gram,
                )
                columns.append(_ot_block(*args))
            rows.append(torch.cat(columns, dim=2))
        return torch.cat(rows, dim=1)

    neighbors = x.reshape(b * n, features)[idx]

    def absolute(values, gathered):
        if needs_grad:
            # Preserve the original gradient accumulation at zero displacements.
            return distance(F.normalize(gathered.float(), dim=-1, eps=1e-12))
        # Normalize each particle once, before repeated neighborhood gathering.
        values = F.normalize(values.float(), dim=-1, eps=1e-12)
        if cache_cosines:
            return distance(local_idx, values @ values.transpose(-1, -2))
        return distance(values.reshape(b * n, -1)[idx])

    full = _debias(absolute(x, neighbors))
    # Subtract in the original dtype, matching the old implementation.
    relative = _debias(distance(
        F.normalize((x.unsqueeze(2) - neighbors).float(), dim=-1, eps=1e-12)
    ))
    geometric = _debias(absolute(x[..., :3], neighbors[..., :3]))
    
    result = torch.stack((full, relative, geometric), dim=-1)
    if is_log:
        result = result.clamp_min(1e-12).log()
    return result.to(x.dtype) * (live[:, :, None] & live[:, None, :]).unsqueeze(-1)


class InputBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        act_layer=nn.LeakyReLU,
        mlp_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.mlp = MLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            drop=mlp_drop,
            act_layer=act_layer,
        )
        self.norm = norm_layer(in_features)

    def forward(self, x, mask):
        x_mlp = self.mlp(self.norm(x), mask)
        return x_mlp, x


class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_features,
        out_features,
        act_layer=nn.LeakyReLU,
        mlp_drop=0.0,
        norm_layer=nn.LayerNorm,
        int_type="lhc",
        feature_drop=0,
    ):
        super().__init__()
        self.int_type = int_type
        self.mlp = MLP(
            in_features=3,
            hidden_features=2 * hidden_features,
            out_features=out_features,
            drop=mlp_drop,
            act_layer=act_layer,
        )
        self.feature_drop = (
            NoScaleDropout(feature_drop) if feature_drop > 0.0 else nn.Identity()
        )

    def forward(self, x, mask, indices=None):
        B, M, C = x.shape
        xi = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        xj = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        mask_event = (mask.float() @ mask.float().transpose(-1, -2)).unsqueeze(-1)

        if self.int_type == "lhc":
            x_int = torch.cat(
                [
                    get_mass(xi, xj, mask_event, is_log=True),
                    get_dr(xi, xj, mask_event, is_log=True),
                    get_kt(xi, xj, mask_event, is_log=True),
                ],
                -1,
            )
        elif self.int_type == "astro":
            x_int = torch.cat(
                [
                    get_d3(xi, xj, mask_event, is_log=True),
                    get_dot(xi, xj, mask_event, is_log=True),
                    get_dot_diff(xi, xj, mask_event, is_log=True),
                ],
                -1,
            )
        elif self.int_type == "ot":
            x_int = get_ot(x, mask, indices) * mask_event

        out = self.mlp(x_int)
        # mask again
        out = out * mask_event
        out = self.feature_drop(out)

        # reshape to [B*H, M, M]
        out = einops.rearrange(out, "b n1 n2 h -> (b h) n1 n2")
        # bias = out.reshape(-1, M, M)

        return out


class LocalEmbeddingBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        mlp_drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
        K=10,
        num_heads=4,
        local_int=False,
        int_type="lhc",
        num_transformers=2,
        feature_drop=0,
    ):
        super().__init__()
        self.K = K
        self.num_heads = num_heads
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.local_int = local_int
        self.int_type = int_type
        k_features = in_features if not self.local_int else in_features + 3

        self.mlp = MLP(
            in_features=k_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            act_layer=act_layer,
            drop=mlp_drop,
        )

        self.in_blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=out_features,
                    num_heads=num_heads,
                    mlp_ratio=2,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    skip=False,
                    use_int=False,
                )
                for _ in range(num_transformers)
            ]
        )

    def pairwise_distance(self, points):
        r = torch.sum(points * points, dim=2, keepdim=True)
        m = torch.bmm(points, points.transpose(1, 2))
        D = r - 2 * m + r.transpose(1, 2)
        return D

    def forward(self, points, features, mask, indices=None):
        batch_size, num_points, num_dims = features.shape
        if indices is None:
            indices, neighbor_valid = knn(points, mask, self.K)
        else:
            neighbor_valid = mask.reshape(batch_size * num_points, 1)[indices].reshape(
                batch_size, num_points, self.K, 1
            ) & mask.unsqueeze(2)

        neighbors = features.view(batch_size * num_points, -1)[indices, :]
        neighbors = neighbors.view(batch_size, num_points, self.K, num_dims)

        mask_neighbors = neighbor_valid

        knn_fts_center = features.unsqueeze(2).expand_as(neighbors)
        local_features = knn_fts_center - neighbors

        if self.local_int:
            # Add the information of the interaction matrix
            if self.int_type == "lhc":
                x_int = torch.cat(
                    [
                        get_mass(
                            knn_fts_center, neighbors, mask_neighbors, is_log=True
                        ),
                        get_dr(knn_fts_center, neighbors, mask_neighbors, is_log=True),
                        get_kt(knn_fts_center, neighbors, mask_neighbors, is_log=True),
                    ],
                    -1,
                )
            elif self.int_type == "astro" or self.int_type == "ot":
                x_int = torch.cat(
                    [
                        get_d3(knn_fts_center, neighbors, mask_neighbors, is_log=True),
                        get_dot(knn_fts_center, neighbors, mask_neighbors, is_log=True),
                        get_dot_diff(
                            knn_fts_center, neighbors, mask_neighbors, is_log=True
                        ),
                    ],
                    -1,
                )

            local_features = (
                torch.cat(
                    [local_features, x_int],
                    -1,
                )
                * mask_neighbors
            )

        mask_neighbors = mask_neighbors.view(batch_size * num_points, self.K, 1)
        local_features = local_features.view(batch_size * num_points, self.K, -1)

        centers = mask.reshape(-1).bool()
        local_mask = mask_neighbors[centers]
        x_real = self.mlp(local_features[centers]) * local_mask
        for blk in self.in_blocks:
            x_real = blk(x_real, mask=local_mask)
        x = x_real.new_zeros((batch_size * num_points, self.K, x_real.shape[-1]))
        x[centers] = x_real
        x = x.view((batch_size, num_points, self.K, -1))
        mask_neighbors = mask_neighbors.view((batch_size, num_points, self.K, -1))
        x = torch.sum(x, dim=2) / torch.sum(1e-9 + mask_neighbors, dim=2) * mask
        return x, indices


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.LeakyReLU,
        drop=0.0,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define layers
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )

    def forward(self, x, mask=None):
        # Apply the first linear layer, activation, dropout, and norm
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        # Apply the second linear layer, norm, and dropout
        x = self.fc2(x)
        x = self.drop(x)
        if mask is not None:
            x = x * mask
        return x


class AttBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_int=False,
        skip=False,
        use_attn=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.use_attn = use_attn

        if self.use_attn:
            self.attn = Attention(
                embed_dim=dim,
                num_heads=num_heads,
                bias=False,
                batch_first=True,
            )

            self.attn_drop = (
                NoScaleDropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.use_int = use_int
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.num_heads = num_heads

    def forward(self, x, mask=None, attn_mask=None, skip=None, offsets=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1)) * mask

        if self.use_attn:
            x_norm = self.norm1(x * mask)
            x_attn = (
                self.attn(
                    query=x_norm,
                    key=x_norm,
                    value=x_norm,
                    key_padding_mask=(
                        ~mask[:, :, 0]
                        if offsets is None and attn_mask is None
                        else None
                    ),
                    offsets=offsets,
                    attn_mask=attn_mask,
                    need_weights=False,
                )[0]
                * mask
            )

            x = x + (
                self.attn_drop(x_attn, offsets=offsets)
                if isinstance(self.attn_drop, NoScaleDropout)
                else self.attn_drop(x_attn)
            )

        x = x + self.mlp(self.norm2(x), mask)
        return x


class TokenAttBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_tokens=1,
        skip=False,
        use_attn=True,
    ):
        super().__init__()

        self.norm = norm_layer(dim)
        self.num_tokens = num_tokens
        self.use_attn = use_attn

        if self.use_attn:
            self.attn = Attention(
                embed_dim=dim,
                num_heads=num_heads,
                bias=False,
                batch_first=True,
            )
            self.attn_drop = (
                NoScaleDropout(attn_drop) if attn_drop > 0.0 else nn.Identity()
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.num_heads = num_heads

    def forward(self, x, mask=None, skip=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1)) * mask

        tokens = x[:, : self.num_tokens]
        x = x[:, self.num_tokens :]

        if self.use_attn:
            token_attn = self.attn(
                query=tokens,
                key=x,
                value=x,
                key_padding_mask=~mask[:, :, 0] if mask is not None else None,
                need_weights=False,
            )[0]

            tokens = tokens + self.attn_drop(token_attn)

        if not self.use_attn:
            weights = torch.ones_like(x[..., :1]) if mask is None else mask
            tokens = tokens + (x * weights).sum(1, keepdim=True) / weights.sum(
                1, keepdim=True
            ).clamp_min(1)
        tokens = tokens + self.mlp(self.norm(tokens))
        return torch.cat([tokens, x], 1)
