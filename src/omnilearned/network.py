import torch
import torch.nn as nn
import einops

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
        cut=0.4,
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
            cut=cut,
            use_time=use_time,
        )

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
                num_add=1 if conditional else 0,
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
                num_add=1 if conditional else 0,
                num_classes=num_classes,
            )
        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm", "scale"}

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        mask = x[:, :, 3:4] != 0

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
            z, v = perturb(x, mask, time)
            z_body = self.body(z, cond, pid, add_info, time)
            z_pred = self.generator(z_body, mask, y)

        if self.mode == "classifier" or self.mode == "pretrain":
            x_body = self.body(x, cond, pid, add_info, torch.zeros_like(time))
            y_pred = self.classifier(x_body, mask)
            if self.mode == "pretrain":
                y_perturb = self.classifier(z_body, mask)

        return y_pred, y_perturb, z_pred, v, x_body, z_body, alpha**2


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
        num_add=1,
        num_classes=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_add = num_add

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
                    num_tokens=self.num_tokens + self.num_add,
                    skip=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = nn.Sequential(
            MLP(
                hidden_size * (self.num_tokens + self.num_add),
                int(mlp_ratio * num_tokens * hidden_size),
                act_layer=act_layer,
                drop=mlp_drop,
                norm_layer=norm_layer,
            ),
        )
        self.out = nn.Linear(
            (self.num_tokens + self.num_add) * hidden_size, num_classes
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm"}

    def forward(self, x, mask):
        B = x.shape[0]

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)

        x = self.fc(x[:, : self.num_tokens + self.num_add].reshape(B, -1))
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
                    num_tokens=num_add,
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

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm"}

    def forward(self, x, mask, y):
        label_embed = self.pid_embed(y)

        # remove the classification tokens
        x = x[:, self.num_tokens :]
        if self.num_add > 0:
            mask = torch.cat([torch.ones_like(mask[:, : self.num_add]), mask], 1)
            x = x.clone()
            x_first = x[:, 0] + label_embed
            x = torch.cat([x_first.unsqueeze(1), x[:, 1:]], dim=1)

        else:
            x = x + label_embed.unsqueeze(1) * mask

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)

        x = self.fc(x[:, self.num_add :]) * mask[:, self.num_add :]
        return self.out(x) * mask[:, self.num_add :]


class PET_body(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size,
        num_transformers=2,
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
        cut=0.4,
        use_time=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_int = use_int
        self.use_time = use_time
        self.conditional = conditional
        self.pid = pid
        self.add_info = add_info
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
        if self.pid:
            # Will assume PIDs are just a list of integers and use the embedding layer, notice that zero_pid_idx is used to map zero-padded entries
            self.pid_embed = nn.Sequential(
                nn.Embedding(pid_dim, hidden_size, padding_idx=0),
                NoScaleDropout(feature_drop),
            )

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
                cut=cut,
            )
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size))

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

        # self.in_scales = nn.ModuleList(
        #     [LayerScale(hidden_size) for _ in range(num_transformers)]
        # )

        self.norm = norm_layer(hidden_size)

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

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"token", "norm"}

    def forward(self, x, cond=None, pid=None, add_info=None, time=None):
        B = x.shape[0]
        mask = x[:, :, 3:4] != 0
        token = self.token.expand(B, -1, -1)

        x_embed, x = self.embed(x, cond, mask)

        # Move away zero-padded entries
        coord_shift = 999.0 * (~mask).float()
        local_features, indices = self.local_physics(coord_shift + x[:, :, :2], x, mask)

        if self.use_int:
            x_int, x_glob = self.interaction(x, mask)
            x_int = einops.rearrange(x_int, "b n1 n2 h -> (b h) n1 n2")
        else:
            x_int = None
            x_glob = 0.0

        # Combine local + global info
        x = x_embed + local_features + x_glob
        # Add classification tokens
        if pid is not None and self.pid:
            # Encode the PID info
            x = x + self.pid_embed(pid) * mask
        if add_info is not None and self.add_info:
            x = x + self.add_embed(add_info) * mask

        if cond is not None and self.conditional:
            # Conditional information: jet level quantities for example
            x = torch.cat([self.cond_embed(cond).unsqueeze(1), x], 1)
            mask = torch.cat([torch.ones_like(mask[:, : self.num_add]), mask], 1)

        if self.use_time and time is not None:
            # Create time token
            time_token = self.time_embed(self.MPFourier(time))
            # Add in the condition token
            if self.conditional:
                x = x.clone()
                x_first = x[:, 0] + time_token
                x = torch.cat([x_first.unsqueeze(1), x[:, 1:]], dim=1)
            else:
                x = x + time_token.unsqueeze(1) * mask

        x = torch.cat([token, x], 1)
        mask = torch.cat([torch.ones_like(mask[:, : self.num_tokens]), mask], 1)

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask, x_int=x_int)
            # x = x + self.in_scales[ib](blk(x, mask=mask, x_int=x_int))

        x = self.norm(x) * mask
        return x
