import json
import numpy as np
import torch
import torch.nn as nn
from omnilearned.network import MLPGEN
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup

from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
    shadow_copy,
)

from omnilearned.diffusion import perturb_hl

import time
import os


def train_step(
    model,
    dataloader,
    gen_cost,
    optimizer,
    scheduler,
    epoch,
    device,
    iterations_per_epoch=-1,
    ema_model=None,
    ema_decay=0.9999,
):
    model.train()

    logs_buff = torch.zeros((1), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)

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
        X = batch["cond"].to(device, dtype=torch.float)
        c = X[:, 1:2]
        X = torch.cat((X[:, :1], X[:, 2:]), dim=1)

        time = torch.rand(size=(X.shape[0],)).to(X.device)
        z, v, _ = perturb_hl(X, time)
        z_pred = model(z, time, c)
        loss = gen_cost(v, z_pred).mean()
        logs["loss"] += loss.detach()

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
    gen_cost,
    epoch,
    device,
    iterations_per_epoch=-1,
):
    model.eval()

    logs_buff = torch.zeros((1), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)

    if iterations_per_epoch < 0:
        iterations_per_epoch = len(dataloader)

    data_iter = iter(dataloader)

    for batch_idx in range(iterations_per_epoch):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        X = batch["cond"].to(device, dtype=torch.float)
        c = X[:, 1:2]
        X = torch.cat((X[:, :1], X[:, 2:]), dim=1)

        time = torch.rand(size=(X.shape[0],)).to(X.device)
        z, v, _ = perturb_hl(X, time)

        with torch.no_grad():
            z_pred = model(z, time, c)
            loss = gen_cost(v, z_pred).mean()

        logs["loss"] += loss.detach()

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
    loss_gen=nn.MSELoss(),
    output_dir="",
    save_tag="",
    iterations_per_epoch=-1,
    epoch_init=0,
    loss_init=np.inf,
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
    for epoch in range(int(epoch_init), num_epochs):
        if isinstance(
            train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        start = time.time()
        train_logs = train_step(
            model,
            train_loader,
            loss_gen,
            optimizer,
            lr_scheduler,
            epoch,
            device,
            iterations_per_epoch=iterations_per_epoch,
            ema_model=ema_model,
            ema_decay=ema_decay,
        )
        val_logs = val_step(
            model,
            val_loader,
            loss_gen,
            epoch,
            device,
            iterations_per_epoch=iterations_per_epoch,
        )

        losses["train_loss"].append(train_logs["loss"])
        losses["val_loss"].append(val_logs["loss"])

        if is_master_node():
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] Loss: {losses['train_loss'][-1]:.4f}, Val Loss: {losses['val_loss'][-1]:.4f} , lr: {lr_scheduler.get_last_lr()[0]}"
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
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "sched": lr_scheduler.state_dict(),
    }

    if ema_model is not None:
        save_dict["ema_model"] = ema_model.state_dict()

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
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model
    base_model.to(device)

    base_model.load_state_dict(checkpoint["model"], strict=True)
    lr_scheduler.load_state_dict(checkpoint["sched"])
    startEpoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]

    if ema_model is not None:
        if "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"], strict=True)

    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
    except Exception:
        if is_main_node:
            print("Optimizer cannot be loaded back, skipping...")

    return startEpoch, best_loss


def run(
    outdir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    wandb=False,
    resuming: bool = False,
    num_feat: int = 3,
    conditional: bool = False,
    num_cond: bool = 1,
    batch: int = 64,
    iterations: int = -1,
    epoch: int = 15,
    warmup_epoch: int = 1,
    optim: str = "lion",
    b1: float = 0.95,
    b2: float = 0.98,
    lr: float = 5e-4,
    wd: float = 0.3,
    mlp_drop: float = 0.1,
    num_workers: int = 16,
):
    local_rank, rank, size = ddp_setup()

    # set up model
    model = MLPGEN(
        input_dim=num_feat,
        hidden_size=256,
        mlp_drop=mlp_drop,
        conditional=conditional,
        cond_dim=num_cond,
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
        use_cond=conditional,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(train_loader)}")
        print("************")

    val_loader = load_data(
        dataset,
        dataset_type="val",
        use_cond=conditional,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
    )

    param_groups = model.parameters()

    if optim not in ["adam", "lion"]:
        raise ValueError(
            f"Optimizer '{optim}' not supported. Choose from adam or lion."
        )

    if optim == "lion":
        optimizer = Lion(param_groups, lr=lr, betas=(b1, b2))
    if optim == "adam":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)

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
        loss_gen=nn.MSELoss(),
        output_dir=outdir,
        save_tag=save_tag,
        iterations_per_epoch=iterations,
        epoch_init=epoch_init,
        loss_init=loss_init,
        run=run,
        ema_model=ema_model,
    )

    dist.destroy_process_group()
