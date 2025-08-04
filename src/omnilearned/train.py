import json
import numpy as np
import torch
import torch.nn as nn
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup

from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_param_groups,
    CLIPLoss,
    get_checkpoint_name,
    shadow_copy,
)
import time
import os
import torch.amp as amp

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False


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


def get_loss(
    outputs, y, class_cost, gen_cost, use_event_loss, use_clip, clip_loss, logs
):
    loss = 0.0
    if outputs["y_pred"] is not None:
        counts = torch.bincount(y, minlength=outputs["y_pred"].shape[-1]).float()
        class_weights = 1.0 / (counts + 1e-6)
        weights = class_weights[y]
        weights = weights / weights.mean()
        loss = loss + get_class_loss(
            weights, outputs["y_pred"], y, class_cost, use_event_loss, logs
        )

    if outputs["z_pred"] is not None:
        nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
        loss_gen = gen_cost(outputs["v"], outputs["z_pred"]).sum((1, 2)) / nonzero
        loss_gen = loss_gen.mean()
        loss = loss + loss_gen
        logs["loss_gen"] += loss_gen.detach()
    if outputs["y_perturb"] is not None:
        counts = torch.bincount(y, minlength=outputs["y_pred"].shape[-1]).float()
        class_weights = 1.0 / (counts + 1e-6)
        weights = class_weights[y]
        weights = outputs["alpha"].squeeze() * weights / weights.mean()
        loss = loss + get_class_loss(
            weights, outputs["y_perturb"], y, class_cost, use_event_loss, logs
        )

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


def train_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    optimizer,
    scheduler,
    epoch,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
    use_amp=False,
    gscaler=None,
    ema_model=None,
    ema_decay=0.999,
):
    model.train()

    logs_buff = torch.zeros((5), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_clip"] = logs_buff[3].view(-1)
    logs["loss_class_event"] = logs_buff[4].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()  # Zero the gradients
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        with amp.autocast(
            "cuda:{}".format(device) if torch.cuda.is_available() else "cpu",
            enabled=use_amp,
        ):
            outputs = model(X, y, **model_kwargs)
            loss = get_loss(
                outputs,
                y,
                class_cost,
                gen_cost,
                use_event_loss,
                use_clip,
                clip_loss,
                logs,
            )

        if use_amp and gscaler is not None:
            gscaler.scale(loss).backward()
            gscaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gscaler.step(optimizer)
            gscaler.update()
        else:
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()  # Update parameters
        scheduler.step()

        if ema_model is not None:
            with torch.no_grad():
                for ema_p, model_p in zip(
                    ema_model.parameters(), model.module.parameters()
                ):
                    ema_p.mul_(ema_decay).add_(model_p, alpha=1.0 - ema_decay)

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def val_step(
    model,
    dataloader,
    class_cost,
    gen_cost,
    epoch,
    device,
    clip_loss=CLIPLoss(),
    use_clip=False,
    use_event_loss=False,
    iterations_per_epoch=-1,
):
    model.eval()

    logs_buff = torch.zeros((5), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_clip"] = logs_buff[3].view(-1)
    logs["loss_class_event"] = logs_buff[4].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        with torch.no_grad():
            outputs = model(X, y, **model_kwargs)
            get_loss(
                outputs,
                y,
                class_cost,
                gen_cost,
                use_event_loss,
                use_clip,
                clip_loss,
                logs,
            )

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    lr_scheduler,
    num_epochs=1,
    device="cpu",
    patience=500,
    loss_class=nn.CrossEntropyLoss(),
    loss_gen=nn.MSELoss(),
    use_clip=False,
    use_event_loss=False,
    output_dir="",
    save_tag="",
    iterations_per_epoch=-1,
    epoch_init=0,
    loss_init=np.inf,
    use_amp=False,
    run=None,
    ema_model=None,
    ema_decay=0.999,
):
    checkpoint_name = get_checkpoint_name(save_tag)

    losses = {
        "train_loss": [],
        "val_loss": [],
    }

    tracker = {"bestValLoss": loss_init, "bestEpoch": epoch_init}
    if use_amp:
        gscaler = amp.GradScaler()
    else:
        gscaler = None
    for epoch in range(int(epoch_init), num_epochs):
        if isinstance(
            train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_logs = train_step(
            model,
            train_loader,
            loss_class,
            loss_gen,
            optimizer,
            lr_scheduler,
            epoch,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
            use_amp=use_amp,
            gscaler=gscaler,
            ema_model=ema_model,
            ema_decay=ema_decay,
        )
        val_logs = val_step(
            model,
            val_loader,
            loss_class,
            loss_gen,
            epoch,
            device,
            use_clip=use_clip,
            use_event_loss=use_event_loss,
            iterations_per_epoch=iterations_per_epoch,
        )

        losses["train_loss"].append(train_logs["loss"])
        losses["val_loss"].append(val_logs["loss"])

        if is_master_node():
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses['train_loss'][-1]:.4f}, Val Loss: {losses['val_loss'][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}"
            )
            print(
                f"Class Loss: {train_logs['loss_class']:.4f}, Class Val Loss: {val_logs['loss_class']:.4f}"
            )
            if use_event_loss:
                print(
                    f"Class Event Loss: {train_logs['loss_class_event']:.4f}, Class Event Val Loss: {val_logs['loss_class_event']:.4f}"
                )
            print(
                f"Gen Loss: {train_logs['loss_gen']:.4f}, Gen Val Loss: {val_logs['loss_gen']:.4f}"
            )
            if use_clip:
                print(
                    f"CLIP loss: {train_logs['loss_clip']:.4f}, CLIP Val Loss: {val_logs['loss_clip']:.4f}"
                )
            print(
                "Time taken for epoch {} is {} sec".format(epoch, time.time() - start)
            )

        if losses["val_loss"][-1] < tracker["bestValLoss"]:
            tracker["bestValLoss"] = losses["val_loss"][-1]
            tracker["bestEpoch"] = epoch

        if is_master_node():
            print("replacing best checkpoint ...")
            save_checkpoint(
                model,
                ema_model,
                epoch + 1,
                optimizer,
                losses["val_loss"][-1],
                lr_scheduler,
                output_dir,
                checkpoint_name,
            )

        if run is not None:
            for key in train_logs:
                run.log({f"train {key}": train_logs[key]})
            for key in val_logs:
                run.log({f"val {key}": val_logs[key]})

        if epoch - tracker["bestEpoch"] > patience:
            print(f"breaking on device: {device}")
            break

    if is_master_node():
        print(
            f"Training Complete, best loss: {tracker['bestValLoss']:.5f} at epoch {tracker['bestEpoch']}!"
        )
        # save losses
        json.dump(losses, open(f"{output_dir}/training_{save_tag}.json", "w"))


def save_checkpoint(
    model,
    ema_model,
    epoch,
    optimizer,
    loss,
    lr_scheduler,
    checkpoint_dir,
    checkpoint_name,
):
    save_dict = {
        "body": model.module.body.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "sched": lr_scheduler.state_dict(),
    }

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
    optimizer,
    lr_scheduler,
    checkpoint_dir,
    checkpoint_name,
    device,
    ema_model=None,
    is_main_node=False,
    fine_tune=False,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)

    if not fine_tune:
        base_model.body.load_state_dict(checkpoint["body"], strict=False)

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            base_model.classifier.load_state_dict(
                checkpoint["classifier_head"], strict=False
            )

        if base_model.generator is not None:
            base_model.generator.load_state_dict(
                checkpoint["generator_head"], strict=False
            )

        lr_scheduler.load_state_dict(checkpoint["sched"])
        startEpoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]
    else:
        if base_model.body is not None and "body" in checkpoint:
            body_state = checkpoint["body"]
            model_state = base_model.body.state_dict()
            filtered_state = {}
            for k, v in body_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    if is_main_node:
                        print(
                            f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                        )

            base_model.body.load_state_dict(filtered_state, strict=False)

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            classifier_state = checkpoint["classifier_head"]
            model_state = base_model.classifier.state_dict()
            filtered_state = {}
            for k, v in classifier_state.items():
                if "out." in k:
                    if is_main_node:
                        print(f"Skipping {k}: explicitly excluded from loading")
                    continue

                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    if is_main_node:
                        print(
                            f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                        )

            base_model.classifier.load_state_dict(filtered_state, strict=False)

        if base_model.generator is not None:
            generator_state = checkpoint["generator_head"]
            model_state = base_model.generator.state_dict()
            filtered_state = {}
            for k, v in generator_state.items():
                if "out." in k:
                    if is_main_node:
                        print(f"Skipping {k}: explicitly excluded from loading")
                    continue
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    if is_main_node:
                        print(
                            f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                        )

            base_model.generator.load_state_dict(filtered_state, strict=False)

        startEpoch = 0.0
        best_loss = np.inf

    if ema_model is not None:
        ema_model.load_state_dict(base_model.state_dict())

        # if "ema_body" in checkpoint:
        #     ema_model.body.load_state_dict(checkpoint["ema_body"],strict=False)

        #     if base_model.generator is not None:
        #         ema_model.generator.load_state_dict(base_model.generator.state_dict())

        # else:
        #     if is_main_node:
        #         print("No EMA body in checkpoint; starting fresh EMA")

        #     ema_model.load_state_dict(base_model.state_dict())

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception:
        if is_main_node:
            print("Optimizer cannot be loaded back, skipping...")

    return startEpoch, best_loss


def run(
    outdir: str = "",
    save_tag: str = "",
    pretrain_tag: str = "pretrain",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    wandb=False,
    fine_tune: bool = False,
    resuming: bool = False,
    num_feat: int = 4,
    conditional: bool = False,
    num_cond: bool = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    pid_dim: int = 9,
    use_add: bool = False,
    num_add: int = 4,
    zero_add: bool = False,
    use_clip: bool = False,
    use_event_loss: bool = False,
    num_classes: int = 2,
    mode: str = "classifier",
    batch: int = 64,
    iterations: int = -1,
    epoch: int = 15,
    warmup_epoch: int = 1,
    use_amp: bool = False,
    optim: str = "lion",
    b1: float = 0.95,
    b2: float = 0.98,
    lr: float = 5e-4,
    lr_factor: float = 10.0,
    wd: float = 0.3,
    num_transf: int = 6,
    num_transf_heads: int = 2,
    num_tokens: int = 4,
    num_head: int = 8,
    K: int = 15,
    base_dim: int = 64,
    mlp_ratio: int = 2,
    attn_drop: float = 0.1,
    mlp_drop: float = 0.1,
    feature_drop: float = 0.0,
    num_workers: int = 16,
    clip_inputs: bool = False,
):
    local_rank, rank, size = ddp_setup()
    # set up model
    model = PET2(
        input_dim=num_feat,
        hidden_size=base_dim,
        num_transformers=num_transf,
        num_transformers_head=num_transf_heads,
        num_heads=num_head,
        mlp_ratio=mlp_ratio,
        mlp_drop=mlp_drop,
        attn_drop=attn_drop,
        feature_drop=feature_drop,
        num_tokens=num_tokens,
        K=K,
        conditional=conditional,
        cond_dim=num_cond,
        pid=use_pid,
        pid_dim=pid_dim,
        add_info=use_add,
        add_dim=num_add,
        use_time=False if mode == "classifier" else True,
        mode=mode,
        num_classes=num_classes,
    )

    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Training on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    train_loader = load_data(
        dataset,
        dataset_type="train",
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        zero_add=zero_add,
        clip_inputs=clip_inputs,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(train_loader)}")
        print("************")

    val_loader = load_data(
        dataset,
        dataset_type="val",
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        zero_add=zero_add,
        clip_inputs=clip_inputs,
    )

    param_groups = get_param_groups(
        model, wd, lr, lr_factor=lr_factor, fine_tune=fine_tune
    )

    if optim not in ["adam", "lion"]:
        raise ValueError(
            f"Optimizer '{optim}' not supported. Choose from adam or lion."
        )

    if optim == "lion":
        optimizer = Lion(param_groups, betas=(b1, b2))
    if optim == "adam":
        optimizer = torch.optim.AdamW(param_groups)

    train_steps = len(train_loader) if iterations < 0 else iterations

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_steps * warmup_epoch,
        num_training_steps=(train_steps * epoch),
    )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )

    # Set up EMA model
    ema_model = shadow_copy(model.module)

    epoch_init = 0
    loss_init = np.inf

    if os.path.isfile(os.path.join(outdir, get_checkpoint_name(save_tag))) and resuming:
        if is_master_node():
            print(
                f"Continue training with checkpoint from {os.path.join(outdir, get_checkpoint_name(save_tag))}"
            )

        epoch_init, loss_init = restore_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            outdir,
            get_checkpoint_name(save_tag),
            local_rank,
            ema_model=ema_model,
            is_main_node=is_master_node(),
        )

    elif (
        os.path.isfile(os.path.join(outdir, get_checkpoint_name(pretrain_tag)))
        and fine_tune
    ):
        if is_master_node():
            print(
                f"Will fine-tune using checkpoint {os.path.join(outdir, get_checkpoint_name(pretrain_tag))}"
            )

        epoch_init, loss_init = restore_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            outdir,
            get_checkpoint_name(pretrain_tag),
            local_rank,
            ema_model=ema_model,
            is_main_node=is_master_node(),
            fine_tune=fine_tune,
        )

    if wandb:
        import wandb

        if is_master_node():
            mode_wandb = None
            wandb.login()
        else:
            mode_wandb = "disabled"

        run = wandb.init(
            # Set the project where this run will be logged
            project="OmniLearn",
            name=save_tag,
            mode=mode_wandb,
            # Track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "epochs": epoch,
                "batch size": batch,
                "mode": mode,
            },
        )
    else:
        run = None

    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        num_epochs=epoch,
        device=device,
        loss_class=nn.CrossEntropyLoss(reduction="none"),
        loss_gen=nn.MSELoss(reduction="none"),
        output_dir=outdir,
        save_tag=save_tag,
        use_clip=use_clip,
        use_event_loss=use_event_loss,
        iterations_per_epoch=iterations,
        epoch_init=epoch_init,
        loss_init=loss_init,
        use_amp=use_amp,
        run=run,
        ema_model=ema_model,
    )

    dist.destroy_process_group()
