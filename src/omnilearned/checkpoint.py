"""Complete atomic checkpoints and legacy component migration."""

import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def unwrap(model):
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = model.module if hasattr(model, "module") else model._orig_mod
    return model


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save(
    path, model, ema, optimizer, scheduler, scaler, epoch, best, config, states, step
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 2,
        "model": unwrap(model).state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "loss": best,
        "config": config,
        "rng_by_rank": states,
        "global_step": step,
    }
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def load_weights(model, checkpoint, fine_tune=False, ema=False, checkpoint_path=None):
    model = unwrap(model)
    state = checkpoint.get("ema") if ema else None
    if state is None:
        state = checkpoint.get("model")
    if state is None:
        state = {}
        for target, source in [
            ("body", "ema_body" if ema else "body"),
            ("classifier", "classifier_head"),
            ("generator", "ema_generator" if ema else "generator_head"),
            ("decoder", "decoder"),
            ("encoder_norm", "encoder_norm"),
        ]:
            fallback = {"ema_body": "body", "ema_generator": "generator_head"}.get(
                source, source
            )
            state.update(
                {
                    f"{target}.{key}": value
                    for key, value in checkpoint.get(
                        source, checkpoint.get(fallback, {})
                    ).items()
                }
            )
        if not state:
            state = checkpoint.get("state_dict", checkpoint)
        if (
            not fine_tune
            and getattr(model, "mode", "") == "encoder"
            and not any(k.startswith("decoder.") for k in state)
        ):
            raise ValueError(
                "Legacy encoder checkpoint lacks decoder weights; exact reconstruction cannot be restored"
            )
    own = model.state_dict()
    if fine_tune:
        selected = {
            k: v
            for k, v in state.items()
            if k in own
            and v.shape == own[k].shape
            and not k.startswith(("classifier.out.", "generator.out.", "decoder.out."))
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Will fine-tune using checkpoint {checkpoint_path}", flush=True)

            checkpoint_components = {k.split(".")[0] for k in state}
            components = [
                name
                for name in dict.fromkeys(k.split(".")[0] for k in own)
                if name in checkpoint_components
            ]
            for component in components:
                prefix = component + "."
                target = {k: v for k, v in own.items() if k.startswith(prefix)}
                source = {k: v for k, v in state.items() if k.startswith(prefix)}

                print(f"{component}:", flush=True)
                for k, v in source.items():
                    if k in selected:
                        continue
                    if k.startswith(
                        ("classifier.out.", "generator.out.", "decoder.out.")
                    ):
                        reason = "explicitly excluded"
                    elif k not in own:
                        reason = "not present in new model"
                    else:
                        reason = f"{v.shape} -> {own[k].shape}"
                    print(f"Skipping {k}: {reason}", flush=True)

                for k in target:
                    if k not in state:
                        print(f"Random {k}: not present in checkpoint", flush=True)

                total = sum(v.numel() for v in target.values())
                loaded = sum(v.numel() for k, v in target.items() if k in selected)
                if total:
                    print(
                        f"Loaded: {loaded:,}/{total:,} ({100 * loaded / total:.2f}%)",
                        flush=True,
                    )
                    print(
                        f"Random: {total - loaded:,}/{total:,} "
                        f"({100 * (total - loaded) / total:.2f}%)",
                        flush=True,
                    )

        return model.load_state_dict(selected, strict=False)
    selected = {k: v for k, v in state.items() if k in own}
    missing = set(own) - set(selected)
    if missing:
        raise ValueError(f"Checkpoint lacks required weights: {sorted(missing)[:8]}")
    return model.load_state_dict(selected)


def fetch_pretrained(path):
    import requests
    import torch.distributed as dist

    path, error = Path(path), None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".download")
                with requests.get(
                    f"https://portal.nersc.gov/cfs/m4567/checkpoints/{path.name}",
                    stream=True,
                    timeout=60,
                ) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            output.write(chunk)
                temporary.replace(path)
        except (requests.RequestException, OSError) as exc:
            error = str(exc)
    if dist.is_initialized():
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error:
        raise RuntimeError(f"Cannot retrieve pretrained checkpoint {path}: {error}")
