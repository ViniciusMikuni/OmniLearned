"""Shared training engine for single-device and multi-node runs."""

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from omnilearned import checkpoint
from omnilearned.diffusion import perturb_hl
from omnilearned.packing import dense_batch
from omnilearned.runtime import (
    make_loader,
    make_model,
    metadata,
    seed_everything,
    to_device,
)
from omnilearned.utils import (
    ddp_setup,
    get_checkpoint_name,
    get_loss,
    get_param_groups,
    shadow_copy,
)

torch.set_float32_matmul_precision("high")
torch._dynamo.config.verbose = False


def batch_loss(model, batch, c, high_level, logs):
    if high_level:
        global_features = batch["cond"]
        cond = global_features[:, 1 : 1 + c["num_cond"]]
        x = torch.cat(
            (global_features[:, :1], global_features[:, 1 + c["num_cond"] :]), -1
        )
        if x.shape[-1] != c["num_feat"]:
            raise ValueError(
                "num_feat must equal the global feature width minus num_cond"
            )
        time = torch.sigmoid(torch.randn(len(x), device=x.device))
        z, target, _ = perturb_hl(x, time)
        loss = (model(z, time, cond).float() - target).square().mean()
        logs.update(loss=loss.detach(), loss_gen=loss.detach())
        return loss
    outputs = model(
        batch["X"],
        batch["y"],
        **{
            k: batch[k]
            for k in ("cond", "pid", "add_info", "offsets")
            if batch.get(k) is not None
        },
    )
    return get_loss(
        outputs,
        batch["y"],
        c["mode"],
        use_event_loss=c.get("use_event_loss", False),
        use_clip=c.get("use_clip", False),
        logs=logs,
        data_pid=(
            dense_batch(batch)["data_pid"]
            if batch.get("data_pid") is not None
            else None
        ),
    )


def pass_epoch(
    model,
    loader,
    c,
    device,
    high_level=False,
    optimizer=None,
    scheduler=None,
    scaler=None,
    ema=None,
):
    training = optimizer is not None
    model.train(training)
    limit = c.get("iterations" if training else "val_iterations", -1)
    steps = limit if limit >= 0 else len(loader)
    if len(loader) == 0 or steps == 0:
        raise ValueError("No batches available for this pass")
    accum = c.get("grad_accumulation", 1) if training else 1
    totals, n_events, updates = {}, 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    precision = c.get("precision", "fp32")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    iterator = iter(loader)
    for index in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = to_device(batch, device)
        logs = {}
        end = (index + 1) % accum == 0 or index + 1 == steps
        sync = (
            model.no_sync()
            if training and not end and isinstance(model, DDP)
            else nullcontext()
        )
        with (
            sync,
            torch.set_grad_enabled(training),
            torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32"),
        ):
            loss = batch_loss(model, batch, c, high_level, logs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {index}")
            if training:
                group = min(accum, steps - (index // accum) * accum)
                scaler.scale(loss / group).backward()
        if training and end:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                c.get("grad_clip", 1.0),
                error_if_nonfinite=not scaler.is_enabled(),
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                scheduler.step()
                updates += 1
                if ema is not None:
                    with torch.no_grad():
                        source = checkpoint.unwrap(model)
                        for a, b in zip(
                            ema.parameters(), source.parameters(), strict=False
                        ):
                            a.lerp_(b, 1 - c.get("ema_decay", 0.999))
                        for a, b in zip(ema.buffers(), source.buffers(), strict=False):
                            a.copy_(b)
            optimizer.zero_grad(set_to_none=True)
        count = len(batch["y"])
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + float(v) * count
        n_events += count
    keys = ["loss", "loss_class", "loss_perturbed", "loss_gen", "loss_clip", "loss_ae", "loss_emd", "loss_ot"]
    values = torch.tensor(
        [totals.get(k, 0.0) for k in keys] + [n_events],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(values)
    if values[-1] == 0:
        raise ValueError("No events available for this pass")
    return {k: float(values[i] / values[-1]) for i, k in enumerate(keys)}, updates


def run(high_level=False, **c):
    local, rank, size = ddp_setup()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_everything(c.get("seed", 42) + rank)
    if c.get("use_amp"):
        c["precision"] = (
            "bf16"
            if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else "fp16"
        )
    if c.get("precision", "fp32") not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16 or fp16")
    if c.get("attention_backend", "auto") not in ("auto", "sdpa", "varlen"):
        raise ValueError("attention_backend must be auto, sdpa or varlen")
    if (
        c.get("grad_accumulation", 1) < 1
        or c["epoch"] < 1
        or not 0 <= c.get("ema_decay", 0.999) < 1
    ):
        raise ValueError("Invalid epoch, accumulation or EMA decay")
    model = make_model(c, high_level).to(device)
    train_loader, val_loader = [
        make_loader(c, split, rank, size, high_level) for split in ("train", "val")
    ]
    outdir = Path(c["outdir"] or ".")
    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / get_checkpoint_name(c["save_tag"])
    latest_path = best_path.with_name(best_path.stem + "_latest.pth")
    resume = None
    if c.get("resuming") and c.get("fine_tune"):
        raise ValueError("resuming and fine_tune are mutually exclusive")
    if c.get("resuming") or c.get("fine_tune"):
        path = (
            latest_path
            if c.get("resuming")
            else Path(c["pretrain_path"])
            if c.get("pretrain_path")
            else outdir / get_checkpoint_name(c["pretrain_tag"])
        )
        if c.get("fine_tune"):
            if c.get("pretrain_path") and not path.is_file():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
            checkpoint.fetch_pretrained(path)
        if not path.exists() and c.get("resuming"):
            path = best_path
        resume = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint.load_weights(
            model,
            resume,
            fine_tune=c.get("fine_tune", False),
            checkpoint_path=path,
        )
    groups = get_param_groups(
        model,
        c["wd"],
        c["lr"],
        c.get("lr_factor", 1.0),
        c.get("fine_tune", False),
        c.get("freeze", False),
    )
    steps = c["iterations"] if c["iterations"] >= 0 else len(train_loader)
    if steps < 1:
        raise ValueError("No training batches; increase data or reduce ranks")
    updates = math.ceil(steps / c.get("grad_accumulation", 1))
    total = updates * c["epoch"]
    if c["optim"] == "adam":
        optimizer = torch.optim.AdamW(groups)
    elif c["optim"] == "lion":
        from pytorch_optimizer import Lion

        optimizer = Lion(groups, betas=(c["b1"], c["b2"]))
    elif c["optim"] == "ranger":
        from pytorch_optimizer import Ranger21

        optimizer = Ranger21(groups, num_iterations=total)
    else:
        raise ValueError("optim must be adam, lion or ranger")
    if c.get("sched", "cosine") == "cosine":
        warmup = c.get("warmup_epoch", 0) * updates

        def rate(step):
            return (
                step / max(1, warmup)
                if step < warmup
                else 0.5
                * (
                    1
                    + math.cos(
                        math.pi * min(1.0, (step - warmup) / max(1, total - warmup))
                    )
                )
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    elif c["sched"] == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=[g["lr"] for g in groups], total_steps=total
        )
    else:
        raise ValueError("sched must be cosine or onecycle")
    scaler = torch.amp.GradScaler(device.type, enabled=c.get("precision") == "fp16")
    ema = shadow_copy(model) if c.get("ema", True) else None
    start, best, step = 0, math.inf, 0
    if resume is not None and c.get("resuming"):
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["sched"])
        if resume.get("scaler"):
            scaler.load_state_dict(resume["scaler"])
        start, best, step = (
            resume["epoch"],
            resume["loss"],
            resume.get("global_step", 0),
        )
        if ema is not None:
            checkpoint.load_weights(ema, resume, ema=True)
        if "rng_by_rank" in resume:
            if len(resume["rng_by_rank"]) != size:
                raise ValueError("Exact resume requires the same world size")
            checkpoint.restore_rng(resume["rng_by_rank"][rank])
    # All ranks participate; sampler sizes can differ for validation.
    counts = torch.tensor(
        [len(train_loader.rank_sampler), len(val_loader.rank_sampler)],
        device=device,
        dtype=torch.int64,
    )
    if dist.is_initialized():
        dist.all_reduce(counts)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Model: {c.get('size', 'high-level')} | parameters: {total_params:,} | trainable: {trainable:,}",
            flush=True,
        )
        for i, (label, loader) in enumerate(
            (("Train", train_loader), ("Validation", val_loader))
        ):
            limit = c.get("iterations" if i == 0 else "val_iterations", -1)
            batches = limit if limit >= 0 else len(loader)
            print(
                f"{label}: {len(loader.dataset):,} indexed events, {len(loader.dataset.file_paths):,} files | "
                f"selected across {size} ranks: {int(counts[i]):,} | rank 0: {len(loader.rank_sampler):,} events, "
                f"{len(loader):,} batches | batches/pass: {batches:,}",
                flush=True,
            )
    info = metadata(c, model, train_loader, size, device)
    if rank == 0:
        (outdir / f"resolved_{c['save_tag']}.json").write_text(
            json.dumps(info, indent=2)
        )
    tracker = None
    if c.get("wandb") and rank == 0:
        import wandb

        tracker = wandb.init(project="OmniLearned", name=c["save_tag"], config=info)
    if c.get("compile_model"):
        model = torch.compile(model)
    if size > 1:
        model = DDP(model, device_ids=[local] if device.type == "cuda" else None)
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(start, c["epoch"]):
        start_time = time.time()
        train_loader.rank_sampler.set_epoch(epoch)
        train_logs, n = pass_epoch(
            model,
            train_loader,
            c,
            device,
            high_level,
            optimizer,
            scheduler,
            scaler,
            ema,
        )
        val_logs, _ = pass_epoch(
            checkpoint.unwrap(model), val_loader, c, device, high_level
        )
        step += n
        improved = val_logs["loss"] < best
        best = min(best, val_logs["loss"])
        states = [None] * size
        if size > 1:
            dist.all_gather_object(states, checkpoint.rng_state())
        else:
            states[0] = checkpoint.rng_state()
        if rank == 0:
            for path in [latest_path] + ([best_path] if improved else []):
                checkpoint.save(
                    path,
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    best,
                    c,
                    states,
                    step,
                )
            history["train_loss"].append(train_logs["loss"])
            history["val_loss"].append(val_logs["loss"])
            print(
                f"Epoch [{epoch + 1}/{c['epoch']}] "
                f"Loss: {train_logs['loss']:.4f}, "
                f"Val Loss: {val_logs['loss']:.4f}, "
                f"lr: {scheduler.get_last_lr()[0]:.6g}",
                flush=True,
            )
            print(
                f"Class Loss: {train_logs['loss_class']:.4f}, "
                f"Class Val Loss: {val_logs['loss_class']:.4f}"
            )
            if c.get("mode") in ("pretrain", "pretrain_regression","pretrain_decoder"):
                print(
                    f"Perturbed Loss: {train_logs['loss_perturbed']:.4f}, "
                    f"Perturbed Val Loss: {val_logs['loss_perturbed']:.4f}"
                )
            if c.get("mode") in ("pretrain", "pretrain_regression","pretrain_decoder","generator"):
                print(
                    f"Gen Loss: {train_logs['loss_gen']:.4f}, "
                    f"Gen Val Loss: {val_logs['loss_gen']:.4f}"
                )
            if c.get("use_clip", False):
                print(
                    f"CLIP Loss: {train_logs['loss_clip']:.4f}, "
                    f"CLIP Val Loss: {val_logs['loss_clip']:.4f}"
                )
            if c.get("mode") in ["encoder","pretrain_decoder"]:
                print(
                    f"AE Loss: {train_logs['loss_ae']:.4f}, "
                    f"AE Val Loss: {val_logs['loss_ae']:.4f} | "
                    f"EMD: {train_logs['loss_emd']:.4f} / {val_logs['loss_emd']:.4f} | "
                    f"OT: {train_logs['loss_ot']:.4f} / {val_logs['loss_ot']:.4f}"
                )
            print(
                f"Time taken for epoch {epoch + 1}: {time.time() - start_time:.2f} sec",
                flush=True,
            )
            if tracker:
                from omnilearned.attention import Attention

                tracker.log(
                    {
                        **{f"train/{k}": v for k, v in train_logs.items()},
                        **{f"val/{k}": v for k, v in val_logs.items()},
                        "epoch": epoch + 1,
                        "global_step": step,
                        "best_loss": best,
                        **{
                            f"lr/group_{i}": g["lr"]
                            for i, g in enumerate(optimizer.param_groups)
                        },
                        "attention_backends": sorted(
                            {
                                m.last_backend
                                for m in checkpoint.unwrap(model).modules()
                                if isinstance(m, Attention)
                            }
                        ),
                    },
                    step=step,
                )
    if rank == 0:
        file = outdir / f"training_{c['save_tag']}.json"
        if c.get("resuming") and file.exists():
            old = json.loads(file.read_text())
            history = {k: old[k][:start] + v for k, v in history.items()}
        file.write_text(json.dumps(history, indent=2))
        if tracker:
            tracker.finish()
        print(f"Training Complete, best loss: {best:.5f}")
    train_loader.dataset.close()
    val_loader.dataset.close()
    if dist.is_initialized():
        dist.destroy_process_group()
