import torch
import torch.nn as nn

from omnilearned.layers import (
    NoScaleDropout,
    InteractionBlock,
    LocalEmbeddingBlock,
    MLP,
    AttBlock,
    DynamicTanh,
    InputBlock,
    TokenAttBlock,
)
from omnilearned.diffusion import MPFourier, perturb, get_logsnr_alpha_sigma


class PET2(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_transformers=2,
        num_transformers_head=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        feature_drop=0.0,
        num_tokens=4,
        K=15,
        use_int=True,
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        use_time=False,
        mode="classifier",
        num_classes=2,
    ):
        super().__init__()
        self.mode = mode
        if self.mode not in ["classifier", "generator", "pretrain"]:
            raise ValueError(f"Mode '{self.mode}' not supported.")

        self.body = PET_body(
            input_dim,
            hidden_size,
            num_transformers=num_transformers,
            num_transf_local=num_transformers_head,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            feature_drop=feature_drop,
            num_tokens=num_tokens,
            K=K,
            use_int=use_int,
            conditional=conditional,
            cond_dim=cond_dim,
            pid=pid,
            pid_dim=pid_dim,
            add_info=add_info,
            add_dim=add_dim,
            use_time=use_time,
        )

        self.num_add = self.body.num_add
        self.classifier = None
        self.generator = None
        if self.mode == "classifier" or self.mode == "pretrain":
            self.classifier = PET_classifier(
                hidden_size,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_classes=num_classes,
            )

        if self.mode == "generator" or self.mode == "pretrain":
            self.generator = PET_generator(
                input_dim,
                hidden_size,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_add=self.num_add,
                num_classes=num_classes,
            )
        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm", "scale", "token"}

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        y_pred, y_perturb, z_pred, v, x_body, z_body = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        time = torch.rand(size=(x.shape[0],)).to(x.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(time)
        if self.mode == "generator" or self.mode == "pretrain":
            z, v = perturb(x, time)
            z_body = self.body(z, cond, pid, add_info, time)
            z_pred = self.generator(z_body, y)

        if self.mode == "classifier" or self.mode == "pretrain":
            x_body = self.body(x, cond, pid, add_info, torch.zeros_like(time))
            y_pred = self.classifier(x_body)
            if self.mode == "pretrain":
                y_perturb = self.classifier(z_body)

        return {
            "y_pred": y_pred,
            "y_perturb": y_perturb,
            "z_pred": z_pred,
            "v": v,
            "x_body": x_body,
            "z_body": z_body,
            "alpha": alpha**2,
        }


class PET_classifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_classes=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.in_blocks = nn.ModuleList(
            [
                TokenAttBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=self.num_tokens,
                    skip=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = MLP(
            hidden_size * self.num_tokens,
            int(mlp_ratio * self.num_tokens * hidden_size),
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.out = nn.Linear(self.num_tokens * hidden_size, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x):
        B = x.shape[0]
        mask = x[:, self.num_tokens :, 3:4] != 0
        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)

        x = self.fc(x[:, : self.num_tokens].reshape(B, -1))
        return self.out(x)


class PET_generator(nn.Module):
    def __init__(
        self,
        output_size,
        hidden_size,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.LeakyReLU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_add=1,
        num_classes=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_add = num_add
        self.num_classes = num_classes

        self.pid_embed = nn.Sequential(
            nn.Embedding(num_classes, hidden_size),
            MLP(
                hidden_size,
                int(mlp_ratio * hidden_size),
                act_layer=act_layer,
                drop=mlp_drop,
            ),
        )

        self.in_blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=num_tokens,
                    skip=False,
                    use_int=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = nn.Sequential(
            MLP(
                hidden_size,
                int(mlp_ratio * hidden_size),
                act_layer=act_layer,
                drop=mlp_drop,
            ),
        )

        self.out = nn.Linear(hidden_size, output_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, y):
        # Add tokens and label embedding
        mask = x[:, :, 3:4] != 0
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        x = torch.cat([self.pid_embed(y).unsqueeze(1), x], 1) * mask

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)

        x = (
            self.fc(x[:, self.num_add + self.num_tokens + 1 :])
            * mask[:, self.num_add + self.num_tokens + 1 :]
        )
        return self.out(x) * mask[:, self.num_add + self.num_tokens + 1 :]


class PET_body(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_transformers=2,
        num_transf_local=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        feature_drop=0.0,
        num_tokens=4,
        K=15,
        use_int=True,
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        use_time=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_int = use_int
        self.use_time = use_time
        self.conditional = conditional
        self.pid = pid
        self.add_info = add_info

        self.embed = InputBlock(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * hidden_size),
            out_features=hidden_size,
            norm_layer=norm_layer,
            act_layer=act_layer,
            use_cond=conditional,
        )

        if self.use_int:
            self.interaction = InteractionBlock(
                hidden_features=hidden_size,
                out_features=num_heads,
                # mlp_drop=mlp_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

        self.local_physics = LocalEmbeddingBlock(
            in_features=input_dim + 3 if conditional else input_dim,
            hidden_features=mlp_ratio * hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            K=K,
            num_heads=num_heads,
            physics=True,
            num_transformers=num_transf_local,
        )

        self.num_add = 0
        if self.conditional:
            self.cond_embed = MLP(
                in_features=cond_dim,
                hidden_features=int(mlp_ratio * hidden_size),
                out_features=hidden_size,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            self.num_add += 1

        if self.use_time:
            # Time embedding module for diffusion timesteps
            self.MPFourier = MPFourier(hidden_size)
            self.time_embed = MLP(
                in_features=hidden_size,
                hidden_features=int(mlp_ratio * hidden_size),
                out_features=hidden_size,
                norm_layer=norm_layer,
                act_layer=act_layer,
                bias=False,
            )
            self.num_add += 1

        if self.add_info:
            self.add_embed = nn.Sequential(
                MLP(
                    in_features=add_dim,
                    hidden_features=int(mlp_ratio * hidden_size),
                    out_features=hidden_size,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    bias=False,
                ),
                NoScaleDropout(feature_drop),
            )

        if self.pid:
            # Will assume PIDs are just a list of integers and use the embedding layer, notice that zero_pid_idx is used to map zero-padded entries
            self.pid_embed = nn.Sequential(
                nn.Embedding(pid_dim, hidden_size, padding_idx=0),
                NoScaleDropout(feature_drop),
            )

        self.num_tokens = num_tokens
        self.token = nn.Parameter(1e-3 * torch.ones(1, self.num_tokens, hidden_size))

        self.num_heads = num_heads
        self.in_blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    # act_layer=nn.LeakyReLU,
                    norm_layer=norm_layer,
                    num_tokens=num_tokens + self.num_add,
                    skip=False,
                    use_int=use_int,
                )
                for _ in range(num_transformers)
            ]
        )

        self.norm = norm_layer(hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, cond=None, pid=None, add_info=None, time=None):
        B = x.shape[0]
        mask = x[:, :, 3:4] != 0
        token = self.token.expand(B, -1, -1)

        x_embed, x = self.embed(x, cond, mask)

        # Move away zero-padded entries
        coord_shift = 999.0 * (~mask).float()
        local_features, indices = self.local_physics(coord_shift + x[:, :, :2], x, mask)

        if self.use_int:
            x_int = self.interaction(x, mask)

        # Combine local + global info
        x = x_embed + local_features
        # Add classification tokens

        if pid is not None and self.pid:
            # Encode the PID info
            x = x + self.pid_embed(pid) * mask
        if add_info is not None and self.add_info:
            x = x + self.add_embed(add_info) * mask

        if cond is not None and self.conditional:
            # Conditional information: jet level quantities for example
            x = torch.cat([self.cond_embed(cond).unsqueeze(1), x], 1)

        if self.use_time and time is not None:
            # Create time token
            x = torch.cat([self.time_embed(self.MPFourier(time)).unsqueeze(1), x], 1)

        x = torch.cat([token, x], 1)

        # Create a new mask based on the updated point cloud with additional tokens
        mask = x[:, :, 3:4] != 0

        attn_mask = mask.float() @ mask.float().transpose(-1, -2)
        attn_mask = ~(attn_mask.bool()).repeat_interleave(self.num_heads, dim=0)
        attn_mask = attn_mask.float() * -1e9

        if self.use_int:
            # Add the information of the interaction matrix
            attn_mask[
                :, self.num_tokens + self.num_add :, self.num_tokens + self.num_add :
            ] = (
                x_int
                + attn_mask[
                    :,
                    self.num_tokens + self.num_add :,
                    self.num_tokens + self.num_add :,
                ]
            )

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask, attn_mask=attn_mask)

        x = self.norm(x) * mask
        return x
