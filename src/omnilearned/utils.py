from sklearn import metrics
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from copy import deepcopy
import torch.distributed as dist
from torch.distributed import init_process_group, get_rank
import torch.nn.functional as F
import requests


def get_model_parameters(model_size):
    model_dict = {}
    if model_size == "small":
        model_dict["num_transformers"] = 8
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 8
        model_dict["K"] = 10
        model_dict["base_dim"] = 128
        model_dict["mlp_ratio"] = 2

    elif model_size == "medium":
        model_dict["num_transformers"] = 12
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 16
        model_dict["K"] = 10
        model_dict["base_dim"] = 512
        model_dict["mlp_ratio"] = 2

    elif model_size == "large":
        model_dict["num_transformers"] = 28
        model_dict["num_transformers_head"] = 4
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 32
        model_dict["K"] = 10
        model_dict["base_dim"] = 1024
        model_dict["mlp_ratio"] = 2
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    return model_dict


def print_metrics(y_preds_np, y_np, thresholds=[0.3, 0.5], background_class=0):
    # Compute multiclass AUC
    auc_ovo = metrics.roc_auc_score(
        y_np,
        y_preds_np if y_preds_np.shape[-1] > 2 else y_preds_np[:, -1],
        multi_class="ovo",
    )
    print(f"AUC: {auc_ovo:.4f}\n")

    accuracy = metrics.accuracy_score(y_np, np.argmax(y_preds_np, axis=1))

    print(f"ACC: {accuracy:.4f}\n")

    num_classes = y_preds_np.shape[1]

    for signal_class in range(num_classes):
        if signal_class == background_class:
            continue

        # Create binary labels: 1 for signal_class, 0 for background_class, ignore others
        mask = (y_np == signal_class) | (y_np == background_class)
        y_bin = (y_np[mask] == signal_class).astype(int)
        scores_bin = y_preds_np[mask, signal_class] / (
            y_preds_np[mask, signal_class] + y_preds_np[mask, background_class]
        )

        # Compute ROC
        fpr, tpr, _ = metrics.roc_curve(y_bin, scores_bin)

        print(f"Signal class {signal_class} vs Background class {background_class}:")

        for threshold in thresholds:
            bineff = np.argmax(tpr > threshold)
            print(
                "Class {} effS at {} 1.0/effB = {}".format(
                    signal_class, tpr[bineff], 1.0 / fpr[bineff]
                )
            )


class CLIPLoss(nn.Module):
    # From AstroCLIP: https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/models/astroclip.py#L117
    def get_logits(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        clean_features = F.normalize(clean_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        perturbed_features = F.normalize(perturbed_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features

        logits_per_clean = logit_scale * clean_features @ perturbed_features.T
        return logits_per_clean, logits_per_clean.T

    def forward(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        weight=None,
        logit_scale: float = 2.74,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the clean and perturbed features
        logits_per_clean, logits_per_perturbed = self.get_logits(
            clean_features, perturbed_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_clean.shape[0], device=clean_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_clean, labels, reduction="none")
            + F.cross_entropy(logits_per_perturbed, labels, reduction="none")
        ) / 2
        if weight is not None:
            total_loss = torch.mean(weight * total_loss)
        else:
            total_loss = total_loss.mean()
        return {"contrastive_loss": total_loss} if output_dict else total_loss


def sum_reduce(num, device):
    r"""Sum the tensor across the devices."""
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def pad_array(tensor_list, M: int = 150) -> torch.Tensor:
    """
    Given a list of torch tensors, each of shape (B, N_i, F),
    pads or truncates each along dimension N to length M,
    and returns a single tensor of shape (I, M, F), where
      H = sum over list of B,
      M = target length,
      F = feature dimension.
    """
    # Determine total number of samples and feature dim
    H = sum(t.shape[0] for t in tensor_list)
    _, _, F = tensor_list[0].shape

    # Use the dtype/device of the first tensor
    device = tensor_list[0].device
    dtype = tensor_list[0].dtype

    # Allocate output buffer
    out = torch.zeros((H, M, F), dtype=dtype, device=device)

    idx = 0
    for t in tensor_list:
        B, N, F_ = t.shape
        assert F_ == F, "All tensors must have the same feature dimension F"

        if N < M:
            # create a (B, M, F) zero tensor and copy `t` into its first N slots
            padded = torch.zeros((B, M, F), dtype=dtype, device=device)
            padded[:, :N, :] = t
        else:
            # truncate to the first M points
            padded = t[:, :M, :]

        out[idx : idx + B] = padded
        idx += B

    return out


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
    outputs,
    y,
    class_cost,
    gen_cost,
    use_event_loss,
    use_clip,
    clip_loss,
    logs,
    data_pid=None,
):
    loss = 0.0
    if outputs["y_pred"] is not None:
        counts = torch.bincount(y, minlength=outputs["y_pred"].shape[-1]).float()
        class_weights = 1.0 / (counts + 1e-6)
        weights = class_weights[y]
        weights = weights / weights.mean()

        # weights = torch.ones_like(y).float()
        loss_class = get_class_loss(
            weights, outputs["y_pred"], y, class_cost, use_event_loss, logs
        )

        loss = loss + loss_class

    if outputs["z_pred"] is not None:
        if data_pid is not None:
            data_pid = data_pid.reshape((-1))
            mask = data_pid != -1  # -1 is associated to zero-pdded entries
            loss_gen = torch.mean(
                gen_cost(
                    outputs["z_pred"].reshape((-1, outputs["z_pred"].shape[-1]))[mask],
                    data_pid[mask],
                )
            )
        else:
            nonzero = (outputs["v"][:, :, 0] != 0).sum(1)
            loss_gen = outputs["v_weight"] * gen_cost(outputs["v"], outputs["z_pred"])
            loss_gen = loss_gen.sum((1, 2)) / nonzero
            loss_gen = loss_gen.mean()

        logs["loss_gen"] += loss_gen.detach()
        loss = loss + loss_gen

    if outputs["y_perturb"] is not None and data_pid is None:
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
    checkpoint_dir,
    checkpoint_name,
    device,
    is_main_node=False,
    restore_ema_model=False,
    ema_model=None,
    fine_tune=False,
    optimizer=None,
    lr_scheduler=None,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"

    if fine_tune and not os.path.exists(os.path.join(checkpoint_dir, checkpoint_name)):
        print(f"Fetching pretrained checkpoint {checkpoint_name}")
        file_url = f"https://portal.nersc.gov/cfs/m4567/checkpoints/{checkpoint_name}"
        file_path = os.path.join(checkpoint_dir, checkpoint_name)
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {checkpoint_name}")

    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)

    if restore_ema_model:
        body_name = "ema_body"
        generator_name = "ema_generator"
    else:
        body_name = "body"
        generator_name = "generator_head"

    if not fine_tune:
        base_model.body.load_state_dict(checkpoint[body_name], strict=False)

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            base_model.classifier.load_state_dict(
                checkpoint["classifier_head"], strict=True
            )

        if base_model.generator is not None:
            base_model.generator.load_state_dict(
                checkpoint[generator_name], strict=True
            )

        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["sched"])
        startEpoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["loss"]

    else:

        def filter_partial_model(state, model_state, is_main_node=False):
            filtered_state = {}
            for k, v in state.items():
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
            return filtered_state

        if base_model.body is not None and "body" in checkpoint:
            filtered_state = filter_partial_model(
                checkpoint["body"], base_model.body.state_dict(), is_main_node
            )
            base_model.body.load_state_dict(filtered_state, strict=False)

        if base_model.classifier is not None and "classifier_head" in checkpoint:
            filtered_state = filter_partial_model(
                checkpoint["classifier_head"],
                base_model.classifier.state_dict(),
                is_main_node,
            )
            base_model.classifier.load_state_dict(filtered_state, strict=False)

        if base_model.generator is not None:
            filtered_state = filter_partial_model(
                checkpoint["generator_head"],
                base_model.generator.state_dict(),
                is_main_node,
            )
            base_model.generator.load_state_dict(filtered_state, strict=False)

        startEpoch = 0.0
        best_loss = np.inf

    if ema_model is not None:
        if fine_tune:
            ema_model.load_state_dict(base_model.state_dict())
        elif "ema_body" in checkpoint:
            ema_model.body.load_state_dict(checkpoint["ema_body"], strict=True)

            if base_model.generator is not None:
                ema_model.generator.load_state_dict(
                    checkpoint["ema_generator"], strict=True
                )

    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception:
            if is_main_node:
                print("Optimizer cannot be loaded back, skipping...")

    return startEpoch, best_loss


def shadow_copy(model):
    ema_model = deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def gather_tensors(x):
    """
    If running under DDP, all_gather x from every rank, concat, then return as numpy.
    Otherwise just .cpu().numpy().
    """
    if dist.is_initialized():
        ws = dist.get_world_size()
        # pre‐allocate one buffer per rank
        buf = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(buf, x)
        x = torch.cat(buf, dim=0)
    return x.cpu()


def get_param_groups(model, wd, lr, lr_factor=1.0, fine_tune=False, freeze=False):
    no_decay, decay = [], []
    new_layer_no_decay, new_layer_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_new_layer = name.startswith(
            (
                "classifier.out",
                "generator",
                "body.cond",
                "body.add_embed",
            )
        )

        if any(keyword in name for keyword in model.no_weight_decay()):
            if is_new_layer:
                new_layer_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if is_new_layer:
                new_layer_decay.append(param)
            else:
                decay.append(param)

    # Base learning rate groups

    param_groups = [
        {"params": decay, "weight_decay": wd, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]

    # Adjust learning rate for new layer if fine-tuning
    new_layer_lr = lr * lr_factor if fine_tune else lr

    if new_layer_decay:
        param_groups.append(
            {"params": new_layer_decay, "weight_decay": wd, "lr": new_layer_lr}
        )
    if new_layer_no_decay:
        param_groups.append(
            {"params": new_layer_no_decay, "weight_decay": 0.0, "lr": new_layer_lr}
        )

    if fine_tune and freeze:
        # Freeze body parts but input embeddings
        for name, param in model.body.named_parameters():
            if (
                name.startswith("embed.")
                or name.startswith("local_physics.")
                or name.startswith("cond.")
                or name.startswith("add_embed.")
            ):
                continue
            else:
                param.requires_grad = False

    return param_groups


def get_checkpoint_name(tag):
    return f"best_model_{tag}.pt"


def is_master_node():
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    else:
        return True


def ddp_setup():
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2900"
        os.environ["RANK"] = "0"
        init_process_group(rank=0, world_size=1)
        rank = local_rank = 0
    else:
        init_process_group(init_method="env://")
        # overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    return local_rank, rank, dist.get_world_size()
