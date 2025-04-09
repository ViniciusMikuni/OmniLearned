import torch
import torch.nn as nn
import einops
import math
import torch.nn.functional as F


class LayerScale(nn.Module):
    def __init__(self, projection_dim, init_values=1e-3):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim))

    def forward(self, x, mask=None):
        if mask is not None:
            return x * self.gamma * mask.unsqueeze(-1)
        else:
            return x * self.gamma


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.rate == 0:
            return x
        else:
            mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = torch.empty(mask_shape, device=x.device).bernoulli_(1 - self.rate)
            return x * mask


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return self.scale * (x / (norm + self.eps))


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
    dr2 = (xi[:, :, :, 0] - xj[:, :, :, 0]) ** 2 + (
        xi[:, :, :, 1] - xj[:, :, :, 1]
    ) ** 2
    if is_log:
        return 0.5 * torch.log(torch.where(dr2 > 0, dr2, 1.0)).unsqueeze(-1) * mask
    else:
        return torch.sqrt(dr2).unsqueeze(-1) * mask


def get_kt(xi, xj, mask, is_log=False):
    kt = torch.min(
        torch.stack([torch.exp(xi[:, :, :, 2:3]), torch.exp(xj[:, :, :, 2:3])], -1), -1
    )[0] * get_dr(xi, xj, mask, is_log=False)
    if is_log:
        return torch.log(torch.where(kt > 0, kt, 1.0)) * mask
    else:
        return kt


def get_local_mass(xi, jet, mask, is_log=False, radius=1.0):
    pt = jet[:, :, :, 0]
    z = torch.exp(xi[:, :, :, 2]) / (1e-9 + pt)
    dr2 = (xi[:, :, :, 0] - jet[:, :, :, 1]) ** 2 + (
        xi[:, :, :, 1] - jet[:, :, :, 2]
    ) ** 2
    if is_log:
        return (
            torch.log(torch.where(dr2 > 0, dr2 * z / radius**2, 1.0)).unsqueeze(-1)
            * mask
        )
    else:
        return (z * torch.sqrt(dr2)).unsqueeze(-1) * mask


def get_local_pt(xi, jet, mask, is_log=False):
    pt = jet[:, :, :, 0]
    pt = torch.exp(xi[:, :, :, 2]) / (1e-9 + pt)
    if is_log:
        return torch.log(torch.where(pt > 0, pt, 1.0)).unsqueeze(-1) * mask
    else:
        return pt.unsqueeze(-1) * mask


def get_local_jet(xi, mask, is_log=False):
    def to_cartesian(data, mask=None):
        # Assume the inputs are eta, phi, log(pT), log(E)
        pt = torch.exp(data[:, :, :, 2])
        px = pt * torch.cos(data[:, :, :, 1])
        py = pt * torch.sin(data[:, :, :, 1])
        pz = pt * torch.sinh(data[:, :, :, 0])
        e = torch.exp(data[:, :, :, 3])
        x = torch.stack([e, px, py, pz], -1)
        if mask is not None:
            x = x * mask
        return x

    def get_cylindrical(data, mask=None):
        # Assume the inputs are eta, phi, pT, E

        pt = torch.sqrt(data[:, :, :, 1] ** 2 + data[:, :, :, 2] ** 2)
        phi = torch.arctan2(data[:, :, :, 2], data[:, :, :, 1] + 1e-6)
        eta = torch.arcsinh(data[:, :, :, 3] / (pt + 1e-9))
        m = torch.abs(data[:, :, :, 0] ** 2 - torch.sum(data[:, :, :, 1:] ** 2, -1))
        x = torch.stack([pt, eta, phi, m], -1)
        if mask is not None:
            x = x * mask
        return x

    j_c = to_cartesian(xi, mask).sum(2, keepdims=True)
    j_c = get_cylindrical(j_c)
    return j_c


class InputBlock(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        act_layer=nn.GELU,
        mlp_drop=0.0,
        norm_layer=nn.LayerNorm,
        use_cond=True,
    ):
        super().__init__()
        self.use_cond = use_cond
        self.mlp = MLP(
            in_features=in_features + 3 if use_cond else in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            drop=mlp_drop,
            act_layer=act_layer,
        )

    def forward(self, x, j, mask):
        if j is not None and self.use_cond:
            x_physics = torch.stack(
                [
                    x[:, :, 2] - j[:, 0].unsqueeze(-1),
                    x[:, :, 3] - j[:, 1].unsqueeze(-1),
                    torch.hypot(x[:, :, 0], x[:, :, 1]),
                ],
                -1,
            )
            x = torch.cat([x, x_physics], -1)
        x_mlp = x
        x_mlp = self.mlp(x_mlp, mask)
        return x_mlp, x


class InteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_features,
        out_features,
        act_layer=nn.GELU,
        mlp_drop=0.0,
        norm_layer=nn.LayerNorm,
        cut=0.4,
    ):
        super().__init__()
        self.cut = cut
        self.mlp_int1 = MLP(
            in_features=3,
            hidden_features=hidden_features,
            out_features=hidden_features,
            norm_layer=norm_layer,
            drop=mlp_drop,
            act_layer=act_layer,
        )

        self.mlp_int2 = MLP(
            in_features=hidden_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            drop=mlp_drop,
            act_layer=act_layer,
        )
        self.mlp_local = MLP(
            in_features=2,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            drop=mlp_drop,
            act_layer=act_layer,
        )

    def forward(self, x, mask):
        xi = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        xj = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        mask_event = (mask.float() @ mask.float().transpose(-1, -2)).unsqueeze(-1)
        x_int = torch.cat(
            [
                get_mass(xi, xj, mask_event, is_log=True),
                get_dr(xi, xj, mask_event, is_log=True),
                get_kt(xi, xj, mask_event, is_log=True),
            ],
            -1,
        )

        x_int = self.mlp_int1(x_int) * mask_event
        x_glob = x_int

        x_int = self.mlp_int2(x_int) * mask_event

        mask_local = get_dr(xi, xj, mask_event, is_log=False) < self.cut
        mask_local = mask_local * mask_event
        local_jet = get_local_jet(xj * mask_local, mask_local)

        x_local = torch.cat(
            [
                get_local_pt(xj, local_jet, mask_local, is_log=True),
                get_local_mass(xj, local_jet, mask_local, is_log=True, radius=self.cut),
            ],
            -1,
        )

        x_local = self.mlp_local(x_local) * mask_local
        x_glob = mask * (
            torch.sum(x_glob, 2) / (1e-9 + torch.sum(mask, 1, keepdims=True))
        )
        return x_int + x_local, x_glob


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
        physics=False,
    ):
        super().__init__()
        self.K = K
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.physics = physics
        k_features = in_features if not physics else in_features + 3

        self.mlp1 = MLP(
            in_features=k_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            act_layer=act_layer,
            drop=mlp_drop,
        )

        self.norm1 = norm_layer(out_features)
        self.norm2 = norm_layer(out_features)
        self.norm3 = norm_layer(out_features)

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=out_features,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=True,
            batch_first=True,
        )
        self.mlp2 = MLP(
            in_features=out_features,
            hidden_features=hidden_features,
            out_features=out_features,
            norm_layer=norm_layer,
            act_layer=act_layer,
            drop=mlp_drop,
        )

    def pairwise_distance(self, points):
        r = torch.sum(points * points, dim=2, keepdim=True)
        m = torch.bmm(points, points.transpose(1, 2))
        D = r - 2 * m + r.transpose(1, 2)
        return D

    def forward(self, points, features, mask, indices=None):

        batch_size, num_points, num_dims = features.shape
        if indices is None:
            distances = self.pairwise_distance(
                points
            )  # uses custom pairwise function, not torch.cdist
            _, indices = torch.topk(-distances, k=self.K + 1, dim=-1)
            indices = indices[:, :, 1:]  # Exclude self

            idx_base = (
                torch.arange(0, batch_size, device=features.device).view(-1, 1, 1)
                * num_points
            )
            indices = indices + idx_base
            indices = indices.view(-1)

        neighbors = features.view(batch_size * num_points, -1)[indices, :]
        neighbors = neighbors.view(batch_size, num_points, self.K, num_dims)

        mask_neighbors = mask.view(batch_size * num_points, -1)[indices, :]

        mask_neighbors = mask_neighbors.view(batch_size, num_points, self.K, 1)
        mask_neighbors = mask_neighbors * mask.unsqueeze(2).expand_as(mask_neighbors)

        knn_fts_center = features.unsqueeze(2).expand_as(neighbors)
        # local_features = torch.cat([knn_fts_center-neighbors, neighbors], dim=-1)
        local_features = knn_fts_center - neighbors
        if self.physics:
            local_features = (
                torch.cat(
                    [
                        local_features * mask_neighbors,
                        get_mass(
                            knn_fts_center, neighbors, mask_neighbors, is_log=True
                        ),
                        get_dr(knn_fts_center, neighbors, mask_neighbors, is_log=True),
                        get_kt(knn_fts_center, neighbors, mask_neighbors, is_log=True),
                    ],
                    -1,
                )
                * mask_neighbors
            )
        local_features = self.mlp1(local_features) * mask_neighbors

        local_features_norm = self.norm1(
            local_features.view((batch_size * num_points, self.K, -1))
        )
        local_features = (
            self.multi_head_attention(
                query=local_features_norm,
                key=local_features_norm,
                value=local_features_norm,
                need_weights=False,
            )[0].view((batch_size, num_points, self.K, -1))
            * mask_neighbors
        )

        local_features = local_features + self.mlp2(
            self.norm2(local_features), mask_neighbors
        )

        local_features = (
            torch.sum(local_features, dim=2)
            / torch.sum(1e-9 + mask_neighbors, dim=2)
            * mask
        )
        local_features = self.norm3(local_features)
        return local_features, indices


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
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
        if norm_layer is not None:
            self.norm = norm_layer(hidden_features)
        else:
            self.norm = None

    def forward(self, x, mask=None):
        # Apply the first linear layer, activation, dropout, and norm
        x = self.fc1(x)
        x = self.act(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.drop(x)
        # Apply the second linear layer, norm, and dropout
        x = self.fc2(x)
        x = self.drop(x)
        if mask is not None:
            x = x * mask
        return x


def modulate(x, shift, scale, mask=None):
    x = x * (1 + scale) + shift
    if mask is not None:
        return x * mask
    else:
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
        num_tokens=1,
        use_int=False,
        skip=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.num_tokens = num_tokens
        # self.attn = Attention(dim,num_heads,attn_drop=attn_drop)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=False,
            batch_first=True,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        # self.scale1 = LayerScale(dim)
        # self.scale2 = LayerScale(dim)
        self.use_int = use_int
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.num_heads = num_heads

    def forward(self, x, x_int=None, mask=None, skip=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1)) * mask

        if mask is not None:
            attn_mask = mask.float() @ mask.float().transpose(-1, -2)
            attn_mask = ~(attn_mask.bool()).repeat_interleave(self.num_heads, dim=0)
            attn_mask = attn_mask.float() * -1e9
            if self.use_int:
                attn_mask[:, self.num_tokens :, self.num_tokens :] = (
                    x_int + attn_mask[:, self.num_tokens :, self.num_tokens :]
                )
        else:
            attn_mask = None

        x_norm = self.norm1(x * mask)
        x = (
            x
            + self.attn(
                # x_norm,attn_mask)
                query=x_norm,
                key=x_norm,
                value=x_norm,
                # key_padding_mask=~mask[:,:,0],
                attn_mask=attn_mask,
                need_weights=False,
            )[0]
            * mask
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
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.num_tokens = num_tokens
        # self.attn = Attention(dim,num_heads,attn_drop=attn_drop)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=False,
            batch_first=True,
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

        x_norm = self.norm1(x)

        tokens = x[:, : self.num_tokens]
        x = x[:, self.num_tokens :]

        tokens = (
            tokens
            + self.attn(
                # x_norm,attn_mask)
                query=x_norm[:, : self.num_tokens],
                key=x_norm[:, self.num_tokens :],
                value=x_norm[:, self.num_tokens :],
                key_padding_mask=~mask[:, :, 0],
                need_weights=False,
            )[0]
        )
        tokens = tokens + self.mlp(self.norm2(tokens))
        return torch.cat([tokens, x], 1)


class ScalableSoftmax(nn.Module):
    """Scalable-Softmax (SSMax) implementation from the paper
    'Scalable-Softmax Is Superior for Attention'.

    This is a drop-in replacement for standard Softmax that helps prevent attention
    fading in transformers by incorporating input size scaling. The scaling helps
    maintain focused attention distributions even with large input sizes.

    Args:
        s (float, optional): Scaling parameter that controls attention focusing strength.
            Lower values (e.g. 0.1) produce sharper attention, higher values (e.g. 1.0)
            produce softer attention. Default: 0.43 as used in paper.
        learn_scaling (bool, optional): If True, make scaling parameter learnable.
            Default: True
        bias (bool, optional): If True, adds a learnable bias term. The paper found
            that while bias helps training, it can hurt length generalization.
            Default: False

    Shape:
        - Input: (*, N) where * is any number of dimensions and N is the sequence length
        - Output: Same shape as input
    """

    def __init__(self, s: float = 0.43, learn_scaling: bool = True, bias: bool = False):
        super().__init__()

        # Initialize scaling parameter
        if learn_scaling:
            self.s = nn.Parameter(torch.tensor(s, dtype=torch.float))
        else:
            self.register_buffer("s", torch.tensor(s, dtype=torch.float))

        # Optional bias parameter
        self.has_bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        """Forward pass applying SSMax along specified dimension.

        Args:
            x (torch.Tensor): Input tensor
            dim (int): Dimension along which to apply SSMax. Default: -1

        Returns:
            torch.Tensor: Output tensor with same shape as input
        """

        # Apply scaling factor based on input size
        scale = self.s * math.log(x.size(dim))
        if self.has_bias:
            scale += self.b

        return F.softmax(x.mul(scale) + attn_mask, dim=dim)

    def extra_repr(self) -> str:
        """String representation of module."""
        s_val = self.s.item()
        if self.has_bias:
            return f"s={s_val:.3f}, b={self.b.item():.3f}"
        return f"s={s_val:.3f}, bias=False"


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ssmax = ScalableSoftmax(
            s=0.4,  # scaling parameter
            learn_scaling=True,  # make scaling parameter learnable
            bias=False,  # whether to use bias term
        )

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape

        qkv = self.qkv(x)
        qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.ssmax(attn, attn_mask.reshape((B, self.num_heads, L, L)))
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
