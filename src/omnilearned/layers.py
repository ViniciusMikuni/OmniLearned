import torch
import torch.nn as nn
import einops


class LayerScale(nn.Module):
    def __init__(self, projection_dim, init_values=1e-3, channels_last=True):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(projection_dim))
        self.channels_last = channels_last

    def forward(self, x, mask=None):
        if self.channels_last:
            if mask is not None:
                return x * self.gamma * mask.unsqueeze(-1)
            else:
                return x * self.gamma
        else:
            if mask is not None:
                return x * self.gamma[:, None, None] * mask.unsqueeze(1)
            else:
                return x * self.gamma[:, None, None]


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


def get_kt(xi, xj, mask, is_log=False):
    kt = torch.min(
        torch.stack([torch.exp(xi[:, :, :, 2:3]), torch.exp(xj[:, :, :, 2:3])], -1), -1
    )[0] * get_dr(xi, xj, mask, is_log=False)
    if is_log:
        return torch.log(torch.where(kt > 0, kt, 1.0)) * mask
    else:
        return kt


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
    ):
        super().__init__()
        self.mlp = MLP(
            in_features=3,
            hidden_features=2 * hidden_features,
            out_features=out_features,
            drop=mlp_drop,
            act_layer=act_layer,
        )

    def forward(self, x, mask):
        B, M, C = x.shape
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

        out = self.mlp(x_int)
        # mask again
        out = out * mask_event

        # reshape to [B*H, M, M]
        out = einops.rearrange(out, "b n1 n2 h -> (b h) n1 n2")
        # bias = out.reshape(-1, M, M)

        # x_int = self.mlp_int(x_int) * mask_event
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
        physics=False,
        num_transformers=2,
    ):
        super().__init__()
        self.K = K
        self.num_heads = num_heads
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.physics = physics
        k_features = in_features if not self.physics else in_features + 3

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
                    num_tokens=0,
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

        knn_fts_center = features.unsqueeze(2).expand_as(neighbors)
        local_features = knn_fts_center - neighbors

        if self.physics:
            # Add the information of the interaction matrix
            local_features = (
                torch.cat(
                    [
                        local_features,
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

        mask_neighbors = mask_neighbors.view(batch_size * num_points, self.K, 1)
        local_features = local_features.view(batch_size * num_points, self.K, -1)

        attn_mask = mask_neighbors.float() @ mask_neighbors.float().transpose(-1, -2)
        attn_mask = ~(attn_mask.bool()).repeat_interleave(self.num_heads, dim=0)
        attn_mask = attn_mask.float() * -1e9

        x = self.mlp(local_features) * mask_neighbors

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask_neighbors, attn_mask=attn_mask)

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
        num_tokens=1,
        use_int=False,
        skip=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.num_tokens = num_tokens

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

        self.use_int = use_int
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.num_heads = num_heads

    def forward(self, x, mask=None, attn_mask=None, skip=None):
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1)) * mask

        x_norm = self.norm1(x * mask)

        x = (
            x
            + self.attn(
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

        self.norm = norm_layer(dim)
        self.num_tokens = num_tokens

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

        tokens = x[:, : self.num_tokens]
        x = x[:, self.num_tokens :]

        tokens = (
            tokens
            + self.attn(
                query=tokens,
                key=x,
                value=x,
                key_padding_mask=~mask[:, :, 0],
                need_weights=False,
            )[0]
        )
        tokens = tokens + self.mlp(self.norm(tokens))
        return torch.cat([tokens, x], 1)
