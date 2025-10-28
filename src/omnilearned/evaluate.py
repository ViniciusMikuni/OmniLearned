import torch
from omnilearned.network import PET2
from omnilearned.dataloader import load_data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
    restore_checkpoint,
    pad_array,
    get_model_parameters
)
from omnilearned.diffusion import generate
import os
import numpy as np
import h5py
from tqdm.auto import tqdm

# seed_value = 30  # You can choose any integer as your seed
# torch.manual_seed(seed_value)


def eval_model(
    model,
    test_loader,
    dataset,
    mode,
    use_event_loss,
    device="cpu",
    outdir="",
    save_tag="pretrain",
    rank=0,
):

    prediction, cond, labels = test_step(model, test_loader, mode, device)
    if mode == "classifier":
        if use_event_loss:
            np.savez(
                os.path.join(outdir,f"outputs_{save_tag}_{dataset}_{rank}.npz"),
                prediction=prediction[:, :200].softmax(-1).cpu().numpy(),
                event_prediction=prediction[:, 200:].softmax(-1).cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy() if cond is not None else [],
            )
        else:
            np.savez(
                os.path.join(outdir,f"outputs_{save_tag}_{dataset}_{rank}.npz"),
                prediction=prediction.softmax(-1).cpu().numpy(),
                pid=labels.cpu().numpy(),
                cond=cond.cpu().numpy() if cond is not None else [],
            )
    else:
        with h5py.File(
                os.path.join(outdir,f"generated_{save_tag}_{dataset}_{rank}.h5"),
            "w",
        ) as fh5:
            fh5.create_dataset("data", data=prediction.cpu().numpy())
            fh5.create_dataset("global", data=cond.cpu().numpy())
            fh5.create_dataset("pid", data=labels.cpu().numpy() + 1)



def test_step(
    model,
    dataloader,
    mode,
    device,
):
    model.eval()

    preds = []
    labels = []
    conds = []

    for ib, batch in enumerate(
        tqdm(dataloader, desc="Iterating", total=len(dataloader))
        if is_master_node()
        else dataloader
    ):
        X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        npart = X.shape[1]
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }

        with torch.no_grad():
            if mode == "classifier":
                outputs = model(X, y, **model_kwargs)
                preds.append(outputs["y_pred"])
            elif mode == "generator":
                assert "cond" in model_kwargs, (
                    "ERROR, conditioning variables not passed to model"
                )
                preds.append(generate(model,y, X.shape, **model_kwargs))
        labels.append(y)
        conds.append(batch["cond"])
        if mode == "generator":
            if batch["pid"] is not None:
                preds[-1] = torch.cat(
                    [preds[-1], model_kwargs["pid"].unsqueeze(-1).float()], -1
                )
            if batch["add_info"] is not None:
                preds[-1] = torch.cat([preds[-1], model_kwargs["add_info"]], -1)

    if mode == "generator":
        preds = pad_array(preds, npart)
    else:
        preds = torch.cat(preds).to(device)
    return (
        preds,
        torch.cat(conds).to(device) if conds[0] is not None else None,
        torch.cat(labels).to(device),
    )


def run(
    indir: str = "",
    outdir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    num_feat: int = 4,
    model_size: str = "small",
    interaction: bool = False,
    conditional: bool = False,
    num_cond: int = 3,
    use_pid: bool = False,
    pid_idx: int = -1,
    use_add: bool = False,
    num_add: int = 4,
    use_event_loss: bool = False,
    num_classes: int = 2,
    mode: str = "classifier",
    batch: int = 64,
    num_workers: int = 16,
    clip_inputs: bool = False,
):
    local_rank, rank, size = ddp_setup()

    model_params = get_model_parameters(model_size)
    
    # set up model
    model = PET2(
        input_dim=num_feat,
        use_int=interaction,
        conditional=conditional,
        cond_dim=num_cond,
        pid=use_pid,
        add_info=use_add,
        add_dim=num_add,
        mode=mode,
        num_classes=num_classes,
        **model_params
    )
    
    
    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    test_loader = load_data(
        dataset,
        dataset_type="test",
        use_cond=True,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        path=path,
        batch=batch,
        num_workers=num_workers,
        rank=rank,
        size=size,
        clip_inputs=clip_inputs,
        shuffle=False,
    )
    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(test_loader)}")
        print("************")
        
    if os.path.isfile(os.path.join(indir, get_checkpoint_name(save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(indir, get_checkpoint_name(save_tag))}"
            )

        restore_checkpoint(
            model,
            indir,
            get_checkpoint_name(save_tag),
            local_rank,
            is_main_node=is_master_node(),
            restore_ema_model = mode == 'generator',
        )

    else:
        raise ValueError(
            f"Error loading checkpoint: {os.path.join(indir, get_checkpoint_name(save_tag))}"
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

    eval_model(
        model,
        test_loader,
        dataset,
        mode=mode,
        use_event_loss=use_event_loss,
        device=device,
        rank=rank,
        outdir=outdir,
        save_tag=save_tag,
    )
    dist.barrier()
    dist.destroy_process_group()
