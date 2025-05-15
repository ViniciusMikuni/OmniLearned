import json
import numpy as np
import torch
import torch.nn as nn
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_optimizer import Lion
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_param_groups,
    CLIPLoss,
    get_checkpoint_name,
)
import time
import os
import torch.amp as amp

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False


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
):
    model.train()

    logs_buff = torch.zeros((7), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_perturb"] = logs_buff[3].view(-1)
    logs["loss_clip"] = logs_buff[4].view(-1)
    logs["loss_class_event"] = logs_buff[5].view(-1)
    logs["loss_event_perturb"] = logs_buff[6].view(-1)

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
            loss = 0

            if outputs["y_pred"] is not None:
                if use_event_loss:
                    event_mask = y >= 200
                    if event_mask.any():
                        loss_event = class_cost(
                            outputs["y_pred"][event_mask][:, 200:], y[event_mask] - 200
                        ).mean()
                        logs["loss_class_event"] += loss_event.detach()
                        loss = loss + loss_event
                    if (~event_mask).any():
                        loss_class = class_cost(
                            outputs["y_pred"][~event_mask][:, :200], y[~event_mask]
                        ).mean()
                        logs["loss_class"] += loss_class.detach()
                        loss = loss + loss_class
                else:
                    loss_class = class_cost(outputs["y_pred"], y).mean()
                    loss = loss + loss_class
                    logs["loss_class"] += loss_class.detach()
            if outputs["z_pred"] is not None:
                nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
                loss_gen = (
                    gen_cost(outputs["v"], outputs["z_pred"]).sum((1, 2)) / nonzero
                )
                loss_gen = loss_gen.mean()
                loss = loss + loss_gen
                logs["loss_gen"] += loss_gen.detach()
            if outputs["y_perturb"] is not None:
                if use_event_loss:
                    event_mask = y >= 200
                    if event_mask.any():
                        loss_event = torch.mean(
                            outputs["alpha"][event_mask].squeeze(1)
                            * class_cost(
                                outputs["y_perturb"][event_mask][:, 200:],
                                y[event_mask] - 200,
                            )
                        )
                        logs["loss_event_perturb"] += loss_event.detach()
                        loss = loss + loss_event

                    if (~event_mask).any():
                        loss_class = torch.mean(
                            outputs["alpha"][~event_mask].squeeze(1)
                            * class_cost(
                                outputs["y_perturb"][~event_mask][:, :200],
                                y[~event_mask],
                            )
                        )
                        logs["loss_perturb"] += loss_class.detach()
                        loss = loss + loss_class

                else:
                    loss_perturb = torch.mean(
                        outputs["alpha"].squeeze(1)
                        * class_cost(outputs["y_perturb"], y)
                    )
                    loss = loss + loss_perturb
                    logs["loss_perturb"] += loss_perturb.detach()

            if (
                use_clip
                and outputs["z_body"] is not None
                and outputs["x_body"] is not None
            ):
                loss_clip = clip_loss(
                    outputs["x_body"].view(X.shape[0], -1),
                    outputs["z_body"].view(X.shape[0], -1),
                    weight=outputs["alpha"],
                )
                loss = loss + loss_clip
                logs["loss_clip"] += loss_clip.detach()

        logs["loss"] += loss.detach()
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

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def test_step(
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

    logs_buff = torch.zeros((7), dtype=torch.float32, device=device)
    logs = {}
    logs["loss"] = logs_buff[0].view(-1)
    logs["loss_class"] = logs_buff[1].view(-1)
    logs["loss_gen"] = logs_buff[2].view(-1)
    logs["loss_perturb"] = logs_buff[3].view(-1)
    logs["loss_clip"] = logs_buff[4].view(-1)
    logs["loss_class_event"] = logs_buff[5].view(-1)
    logs["loss_event_perturb"] = logs_buff[6].view(-1)

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

        loss = 0

        if outputs["y_pred"] is not None:
            if use_event_loss:
                event_mask = y >= 200
                if event_mask.any():
                    loss_event = class_cost(
                        outputs["y_pred"][event_mask][:, 200:], y[event_mask] - 200
                    ).mean()
                    logs["loss_class_event"] += loss_event.detach()
                    loss = loss + loss_event
                if (~event_mask).any():
                    loss_class = class_cost(
                        outputs["y_pred"][~event_mask][:, :200], y[~event_mask]
                    ).mean()
                    logs["loss_class"] += loss_class.detach()
                    loss = loss + loss_class
            else:
                loss_class = class_cost(outputs["y_pred"], y).mean()
                loss = loss + loss_class
                logs["loss_class"] += loss_class.detach()
        if outputs["z_pred"] is not None:
            nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
            loss_gen = gen_cost(outputs["v"], outputs["z_pred"]).sum((1, 2)) / nonzero
            loss_gen = loss_gen.mean()
            loss = loss + loss_gen
            logs["loss_gen"] += loss_gen.detach()
        if outputs["y_perturb"] is not None:
            if use_event_loss:
                event_mask = y >= 200
                if event_mask.any():
                    loss_event = torch.mean(
                        outputs["alpha"][event_mask].squeeze(1)
                        * class_cost(
                            outputs["y_perturb"][event_mask][:, 200:],
                            y[event_mask] - 200,
                        )
                    )
                    logs["loss_event_perturb"] += loss_event.detach()
                    loss = loss + loss_event
                if (~event_mask).any():
                    loss_class = torch.mean(
                        outputs["alpha"][~event_mask].squeeze(1)
                        * class_cost(
                            outputs["y_perturb"][~event_mask][:, :200], y[~event_mask]
                        )
                    )
                    logs["loss_perturb"] += loss_class.detach()
                    loss = loss + loss_class

            else:
                loss_perturb = torch.mean(
                    outputs["alpha"].squeeze(1) * class_cost(outputs["y_perturb"], y)
                )
                loss = loss + loss_perturb
                logs["loss_perturb"] += loss_perturb.detach()

        if use_clip and outputs["z_body"] is not None and outputs["x_body"] is not None:
            loss_clip = clip_loss(
                outputs["x_body"].view(X.shape[0], -1),
                outputs["z_body"].view(X.shape[0], -1),
                weight=outputs["alpha"],
            )
            loss = loss + loss_clip
            logs["loss_clip"] += loss_clip.detach()

        logs["loss"] += loss.detach()

    if dist.is_initialized():
        for key in logs:
            dist.all_reduce(logs[key].detach())
            logs[key] = float(logs[key] / dist.get_world_size() / iterations_per_epoch)

    return logs


def train_model(
    model,
    train_loader,
    test_loader,
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
        )
        val_logs = test_step(
            model,
            test_loader,
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
            print(
                f"Class Event Loss: {train_logs['loss_class_event']:.4f}, Class Event Val Loss: {val_logs['loss_class_event']:.4f}"
            )
            print(
                f"Gen Loss: {train_logs['loss_gen']:.4f}, Gen Val Loss: {val_logs['loss_gen']:.4f}"
            )
            print(
                f"Class Perturb Loss: {train_logs['loss_perturb']:.4f}, Class Val Perturb Loss: {val_logs['loss_perturb']:.4f}"
            )
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
    model, epoch, optimizer, loss, lr_scheduler, checkpoint_dir, checkpoint_name
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
    base_model.body.load_state_dict(checkpoint["body"], strict=False)

    if not fine_tune:
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
        # for param in base_model.body.parameters():
        #     param.requires_grad = False

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            classifier_state = checkpoint["classifier_head"]
            model_state = base_model.classifier.state_dict()
            filtered_state = {}
            for k, v in classifier_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    filtered_state[k] = v
                else:
                    if is_main_node:
                        print(
                            f"Skipping {k}: shape mismatch (checkpoint: {v.shape}, model: {model_state[k].shape if k in model_state else 'missing'})"
                        )

            base_model.classifier.load_state_dict(filtered_state, strict=False)

        if base_model.generator is not None:
            classifier_state = checkpoint["generator_head"]
            model_state = base_model.generator.state_dict()
            filtered_state = {}
            for k, v in classifier_state.items():
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
    use_add: bool = False,
    num_add: int = 4,
    use_clip: bool = False,
    use_event_loss: bool = False,
    num_classes: int = 2,
    mode: str = "classifier",
    batch: int = 64,
    iterations: int = -1,
    epoch: int = 15,
    use_amp: bool = False,
    b1: float = 0.95,
    b2: float = 0.98,
    lr: float = 5e-4,
    lr_factor: float = 10.0,
    wd: float = 0.3,
    num_transf: int = 6,
    num_tokens: int = 4,
    num_head: int = 8,
    K: int = 15,
    base_dim: int = 64,
    mlp_ratio: int = 2,
    attn_drop: float = 0.1,
    mlp_drop: float = 0.1,
    feature_drop: float = 0.0,
    num_workers: int = 16,
):
    local_rank, rank, size = ddp_setup()
    # set up model
    model = PET2(
        input_dim=num_feat,
        hidden_size=base_dim,
        num_transformers=num_transf,
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
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(train_loader)}")
        print("************")

    test_loader = load_data(
        dataset,
        dataset_type="test",
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
    )

    param_groups = get_param_groups(
        model, wd, lr, lr_factor=lr_factor, fine_tune=fine_tune
    )
    optimizer = Lion(param_groups, betas=(b1, b2))
    # optimizer =  torch.optim.AdamW(param_groups)
    train_steps = len(train_loader) if iterations < 0 else iterations
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, (train_steps * epoch)
    )

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
        )

    if (
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
            is_main_node=is_master_node(),
            fine_tune=fine_tune,
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
        test_loader,
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
    )

    dist.destroy_process_group()
