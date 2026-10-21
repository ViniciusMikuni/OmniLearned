import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from omnilearned.attention import Attention
from omnilearned.diffusion import MPFourier, get_logsnr_alpha_sigma, perturb
from omnilearned.layers import (
    MLP,
    AttBlock,
    DynamicTanh,
    InputBlock,
    InteractionBlock,
    LocalEmbeddingBlock,
    NoScaleDropout,
    TokenAttBlock,
)
from omnilearned.neighbors import knn
from omnilearned.packing import offsets_from_mask, padded


class PET2(nn.Module):
    def __init__(
        self,
        input_dim,
        use_int=True,
        local_int=True,
        int_type="lhc",
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        mode="classifier",
        num_classes=2,
        num_gen_classes=1,
        base_dim=128,
        num_transformers=2,
        num_transformers_head=2,
        num_tokens=4,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.0,
        attn_drop=0.0,
        feature_drop=0.0,
        K=15,
        skip=False,
        num_coord=3,
        use_local=True,
        use_attn=True,
        num_particles=30,
        dataset="top",
        qk_norm=False,
        attention_backend="auto",
        activation_checkpointing=False,
    ):
        super().__init__()
        if type(num_tokens) is not int or num_tokens < 1:
            raise ValueError("num_tokens must be a positive integer")
        self.mode = mode
        if self.mode not in [
            "classifier",
            "generator",
            "regression",
            "segmentation",
            "ftag",
            "pretrain",
            "pretrain_regression",
            "pretrain_decoder",
            "encoder",
        ]:
            raise ValueError(f"Mode '{self.mode}' not supported.")
        self.dataset = dataset
        self.cond_dim = cond_dim
        self.body = PET_body(
            input_dim,
            base_dim,
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
            local_int=local_int,
            int_type=int_type,
            conditional=conditional,
            cond_dim=cond_dim,
            pid=pid,
            pid_dim=pid_dim,
            add_info=add_info,
            add_dim=add_dim,
            use_time=self.mode
            in ["generator", "pretrain", "pretrain_regression", "pretrain_decoder"],
            skip=skip,
            num_coord=num_coord,
            use_local=use_local,
            use_attn=use_attn,
        )

        self.num_add = self.body.num_add
        self.num_tokens = num_tokens
        self.classifier = None
        self.generator = None

        use_classifier = self.mode in [
            "classifier",
            "ftag",
            "regression",
            "pretrain",
            "pretrain_regression",
            "pretrain_decoder",
        ]
        use_generator = self.mode in [
            "generator",
            "segmentation",
            "ftag",
            "pretrain",
            "pretrain_decoder",
            "pretrain_regression",
        ]

        if use_classifier:
            self.classifier = PET_classifier(
                base_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_classes=num_classes,
                use_attn=use_attn,
            )

        if use_generator:
            self.generator = PET_generator(
                num_gen_classes if self.mode in ("segmentation", "ftag") else input_dim,
                base_dim,
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
                skip_pid=(num_classes == 1)
                | (self.mode == "segmentation")
                | (self.mode == "pretrain_regression")
                | (self.mode == "ftag"),
                use_attn=use_attn,
            )

        if self.mode in ["encoder", "pretrain_decoder"]:
            self.decoder = PET_decoder(
                input_dim,
                base_dim,
                num_particles=num_particles,
                num_features=input_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
            )
            self.encoder_mlp = MLP(
                base_dim,
                int(mlp_ratio * base_dim),
                out_features=input_dim,
                act_layer=act_layer,
                drop=mlp_drop,
            )
            self.encoder_norm = nn.LayerNorm(input_dim)

        self.body.activation_checkpointing = activation_checkpointing
        for module in self.modules():
            if isinstance(module, Attention):
                module.backend = attention_backend
                if qk_norm:
                    module.enable_qk_norm()
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
        return {"norm", "token"}

    def forward(
        self, x, y, cond=None, pid=None, add_info=None, offsets=None, mask=None
    ):
        if offsets is not None:
            x, mask = padded(x, offsets)
            pid = padded(pid, offsets)[0] if pid is not None else None
            add_info = padded(add_info, offsets)[0] if add_info is not None else None
        if mask is None:
            mask = x[..., 2:3] != 0
        x = torch.where(mask, x, 0.0)
        y_pred = y_perturb = z_pred = v = v_weight = x_body = z_body = x_hat = (
            mask_logits
        ) = latent = None

        time = torch.sigmoid(torch.randn(x.shape[0], device=x.device))
        # time = torch.rand(size=(x.shape[0],)).to(x.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(time)

        if self.mode in [
            "generator",
            "pretrain",
            "pretrain_decoder",
            "pretrain_regression",
        ]:
            z, v, v_weight = perturb(x, time, self.dataset, mask=mask)
            z_body, body_mask = self.body(
                z, cond, pid, add_info, time, mask=mask, return_mask=True
            )
            z_pred = self.generator(z_body, y.long(), mask=body_mask)

        if self.mode in [
            "classifier",
            "regression",
            "segmentation",
            "pretrain",
            "pretrain_decoder",
            "pretrain_regression",
            "ftag",
            "encoder",
        ]:
            x_body, body_mask = self.body(
                x,
                cond,
                pid,
                add_info,
                torch.zeros_like(time),
                mask=mask,
                return_mask=True,
            )

            if self.mode in [
                "classifier",
                "pretrain",
                "regression",
                "pretrain_regression",
                "pretrain_decoder",
                "ftag",
            ]:
                y_pred = self.classifier(x_body, mask=body_mask)
            if self.mode in ["pretrain", "pretrain_decoder", "pretrain_regression"]:
                y_perturb = self.classifier(z_body, mask=body_mask)
            if self.mode in ["encoder", "pretrain_decoder"]:
                latent = self.encoder_norm(
                    self.encoder_mlp(x_body[:, : self.num_tokens])
                )
                # z = latent + 0.05 * torch.randn_like(latent)
                x_hat, mask_logits = self.decoder(latent)
            if self.mode == "ftag" or self.mode == "segmentation":
                z_pred = self.generator(x_body, y, mask=body_mask)

        return {
            "y_pred": y_pred,
            "y_perturb": y_perturb,
            "z_pred": z_pred,
            "v": v,
            "v_weight": v_weight,
            "x_body": x_body,
            "z_body": z_body,
            "alpha": alpha**2,
            "x_hat": x_hat,
            "x": x,
            "mask_logits": mask_logits,
            "latent": latent,
            "mask": mask,
            "body_mask": body_mask,
        }


class PET_decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        num_particles,
        num_features,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.base_dim = base_dim
        self.num_particles = num_particles
        self.num_features = num_features

        self.input_proj = nn.Linear(
            input_dim,
            base_dim,
        )

        self.particle_token = nn.Parameter(torch.zeros(1, num_particles, base_dim))

        self.in_blocks = nn.ModuleList(
            [
                TokenAttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=num_particles,
                    skip=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = MLP(
            base_dim,
            int(mlp_ratio * base_dim),
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.out = nn.Linear(base_dim, num_features)
        self.mask_out = nn.Linear(base_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)
        nn.init.normal_(self.particle_token, std=0.02)

    def forward(self, c):
        B = c.shape[0]
        context = self.input_proj(c)

        particles = self.particle_token.expand(B, -1, -1)

        # particle tokens first, because TokenAttBlock updates first num_tokens
        x = torch.cat([particles, context], dim=1)

        for blk in self.in_blocks:
            x = blk(x)

        x = x[:, : self.num_particles]
        x = self.fc(x)

        x_hat = self.out(x)
        mask_logits = self.mask_out(x).squeeze(-1)

        return x_hat, mask_logits


class PET_classifier(nn.Module):
    def __init__(
        self,
        base_dim,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_classes=2,
        use_attn=True,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.in_blocks = nn.ModuleList(
            [
                TokenAttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=self.num_tokens,
                    skip=False,
                    use_attn=use_attn,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = MLP(
            base_dim * self.num_tokens,
            int(mlp_ratio * self.num_tokens * base_dim),
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.out = nn.Linear(self.num_tokens * base_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, mask=None):
        B = x.shape[0]
        if mask is None:
            raise ValueError("Classifier requires the explicit body mask")
        mask = mask[:, self.num_tokens :]

        for blk in self.in_blocks:
            x = blk(x, mask=mask)

        x = self.fc(x[:, : self.num_tokens].reshape(B, -1))
        return self.out(x)


class PET_generator(nn.Module):
    def __init__(
        self,
        output_size,
        base_dim,
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
        skip_pid=False,
        use_attn=True,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_add = num_add
        self.num_classes = num_classes
        self.skip_pid = skip_pid

        if not self.skip_pid:
            self.pid_embed = nn.Sequential(
                nn.Embedding(num_classes, base_dim),
                MLP(
                    base_dim,
                    int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    act_layer=act_layer,
                    drop=mlp_drop,
                ),
            )
            self.num_add = self.num_add + 1

        self.in_blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    skip=False,
                    use_int=False,
                    use_attn=use_attn,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = nn.Sequential(
            MLP(
                base_dim,
                int(mlp_ratio * base_dim),
                act_layer=act_layer,
                drop=mlp_drop,
            ),
        )

        self.out = nn.Linear(base_dim, output_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, y=None, mask=None):
        if mask is None:
            raise ValueError("Generator requires the explicit body mask")

        if not self.skip_pid and y is not None:
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
            x = torch.cat([self.pid_embed(y).unsqueeze(1), x], 1) * mask

        for blk in self.in_blocks:
            x = blk(x, mask=mask)
        x = (
            self.fc(x[:, self.num_add + self.num_tokens :])
            * mask[:, self.num_add + self.num_tokens :]
        )

        return self.out(x) * mask[:, self.num_add + self.num_tokens :]


class PET_body(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
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
        K=10,
        use_int=True,
        local_int=True,
        int_type="lhc",
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        use_time=False,
        skip=False,
        num_coord=3,
        use_local=True,
        use_attn=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_int = use_int and use_attn
        self.use_time = use_time
        self.use_local = use_local
        self.activation_checkpointing = False
        self.K = K
        self.conditional = conditional
        self.pid = pid
        self.add_info = add_info
        self.skip = skip
        self.num_coord = num_coord
        self.embed = InputBlock(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        if self.use_int:
            self.interaction = InteractionBlock(
                hidden_features=base_dim,
                out_features=num_heads,
                # mlp_drop=mlp_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                int_type=int_type,
                feature_drop=feature_drop,
            )

        if self.use_local:
            self.local_physics = LocalEmbeddingBlock(
                in_features=input_dim,
                hidden_features=mlp_ratio * base_dim,
                out_features=base_dim,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                norm_layer=norm_layer,
                K=K,
                num_heads=num_heads,
                local_int=local_int,
                int_type=int_type,
                num_transformers=num_transf_local,
                feature_drop=feature_drop,
            )

        self.num_add = 0
        if self.conditional:
            self.cond_embed = nn.Sequential(
                MLP(
                    in_features=cond_dim,
                    hidden_features=base_dim,
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            self.num_add += 1

        if self.use_time:
            # Time embedding module for diffusion timesteps
            self.MPFourier = MPFourier(base_dim)
            self.time_embed = MLP(
                in_features=base_dim,
                hidden_features=int(mlp_ratio * base_dim),
                out_features=base_dim,
                norm_layer=norm_layer,
                act_layer=act_layer,
                bias=False,
            )
            self.num_add += 1

        if self.add_info:
            self.add_embed = nn.Sequential(
                MLP(
                    in_features=add_dim,
                    hidden_features=int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    bias=False,
                ),
                NoScaleDropout(feature_drop),
            )

        if self.pid:
            # Will assume PIDs are just a list of integers and use the embedding layer, notice that zero_pid_idx is used to map zero-padded entries
            self.pid_embed = nn.Sequential(
                nn.Embedding(pid_dim, base_dim, padding_idx=0),
                NoScaleDropout(feature_drop),
            )

        self.num_tokens = num_tokens
        self.token = nn.Parameter(1e-3 * torch.ones(1, self.num_tokens, base_dim))

        self.num_heads = num_heads

        if self.skip:
            self.in_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        skip=False,
                        use_int=use_int,
                        use_attn=use_attn,
                    )
                    for _ in range(num_transformers // 2)
                ]
            )

            self.out_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        skip=True,
                        use_int=use_int,
                        use_attn=use_attn,
                    )
                    for _ in range(num_transformers // 2)
                ]
            )

        else:
            self.in_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        skip=False,
                        use_int=use_int,
                        use_attn=use_attn,
                    )
                    for _ in range(num_transformers)
                ]
            )

        self.norm = norm_layer(base_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(
        self,
        x,
        cond=None,
        pid=None,
        add_info=None,
        time=None,
        mask=None,
        return_mask=False,
    ):
        B = x.shape[0]
        mask = x[:, :, 2:3] != 0 if mask is None else mask.bool()
        x = torch.where(mask, x, 0.0)
        if self.conditional and cond is None:
            raise ValueError("Conditional model requires global features")
        if self.use_time and time is None:
            time = x.new_zeros(B)
        token = self.token.expand(B, -1, -1)

        x_embed, x = self.embed(x, mask)

        # Move away zero-padded entries
        indices = None
        if self.use_local:
            local_features, indices = self.local_physics(
                x[:, :, : self.num_coord], x, mask
            )

        x_int = None
        if self.use_int:
            if self.interaction.int_type == "ot" and indices is None:
                indices, _ = knn(x[:, :, : self.num_coord], mask, self.K)
            x_int = self.interaction(x, mask, indices)

        x = x_embed

        if self.use_local:
            # Combine local + global info
            x = x + local_features

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

        prefix = self.num_tokens + self.num_add
        mask = torch.cat(
            (torch.ones((B, prefix, 1), device=x.device, dtype=torch.bool), mask), 1
        )
        attn_mask = None
        if self.use_int:
            attn_mask = x.new_zeros((B * self.num_heads, x.shape[1], x.shape[1]))
            attn_mask[:, prefix:, prefix:] = x_int
            attn_mask = attn_mask.view(B, self.num_heads, x.shape[1], x.shape[1])
            attn_mask = attn_mask.masked_fill(
                ~mask.squeeze(-1)[:, None, None, :], -torch.inf
            )
        dense_mask, dense_shape = mask, x.shape
        offsets = None
        if attn_mask is None:
            offsets = offsets_from_mask(mask)
            x = x[mask.squeeze(-1)]
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)

        def apply_block(block, value, skip=None):
            if self.activation_checkpointing and self.training:
                return checkpoint(
                    block, value, mask, attn_mask, skip, offsets, use_reentrant=False
                )
            return block(
                value, mask=mask, attn_mask=attn_mask, skip=skip, offsets=offsets
            )

        skips = []
        for blk in self.in_blocks:
            x = apply_block(blk, x)
            if self.skip:
                skips.append(x)
        if self.skip:
            for blk in self.out_blocks:
                x = apply_block(blk, x, skips.pop())
        x = self.norm(x) * mask
        if offsets is not None:
            dense = x.new_zeros(dense_shape)
            dense[dense_mask.squeeze(-1)] = x
            x = dense
        return (x, dense_mask) if return_mask else x


class MLPGEN(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        num_layers=3,
        mlp_drop=0.0,
        conditional=False,
        cond_dim=1,
    ):
        super().__init__()
        self.embed = MLP(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.cond = MLP(
            in_features=cond_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.in_blocks = nn.ModuleList(
            [
                MLP(
                    in_features=base_dim,
                    hidden_features=int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for _ in range(num_layers)
            ]
        )

        # Time embedding module for diffusion timesteps
        self.MPFourier = MPFourier(base_dim)
        self.time_embed = MLP(
            in_features=base_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            act_layer=act_layer,
        )

        self.out = nn.Linear(base_dim, input_dim)

    def forward(self, z, time, cond=None):
        z = self.embed(z)
        if cond is not None:
            z = z + self.cond(cond)
        z = torch.cat([self.time_embed(self.MPFourier(time)), z], 1)
        for blk in self.in_blocks:
            z = blk(z)
        z = self.out(z)
        return z
