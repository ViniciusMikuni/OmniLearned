import torch
from omnilearned.network import MLPGEN
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omnilearned.utils import (
    is_master_node,
    ddp_setup,
    get_checkpoint_name,
)
from omnilearned.diffusion import generate_hl
import os
import numpy as np
import h5py
from tqdm.auto import tqdm

# seed_value = 30  # You can choose any integer as your seed
# torch.manual_seed(seed_value)



class HLDataset(Dataset):
    def __init__(
        self,
        file_path,
        file_name = 'outputs_mass_top.npz'
    ):

        self.file = np.load(os.path.join(file_path, file_name))['cond']

    def __len__(self):
        return self.file.shape[0]

    def __getitem__(self, idx):
        return self.file[idx]


def eval_model(
    model,
    test_loader,
    dataset,
    device="cpu",
    rank=0,
    save_tag = "",
    max_part = 100,
):

    prediction, cond= test_step(model, test_loader, device)

    #Fix distributions
    prediction[:,0] = torch.clip(prediction[:,0],np.log(500),None)
    prediction[:,-1] = torch.clip(torch.round(100*prediction[:,-1]).int(),1,max_part)
    prediction[:,-1] = prediction[:,-1].float()/100.

    prediction = torch.cat((prediction[:, :1], cond[:,None], prediction[:, 1:]), dim=1)
    
    with h5py.File(
        f"/pscratch/sd/v/vmikuni/Omnilearned/generated_{save_tag}_{dataset}_{rank}.h5",
        "w",
    ) as fh5:
        fh5.create_dataset("data", data=torch.zeros((prediction.shape[0],max_part,4)).cpu().numpy())
        fh5.create_dataset("global", data=prediction.cpu().numpy())
        fh5.create_dataset("pid", data=torch.zeros(prediction.shape[0],dtype=torch.int64).cpu().numpy())

def test_step(
    model,
    dataloader,
    device,
):
    model.eval()

    preds = []
    conds = []

    for ib, batch in enumerate(
        tqdm(dataloader, desc="Iterating", total=len(dataloader))            
        if is_master_node()
        else dataloader
    ):
        #if ib > 100: break
        X = batch.to(device, dtype=torch.float)
        with torch.no_grad():
            preds.append(generate_hl(model, (X.shape[0],3),X[:,None]))
            
        conds.append(batch)
    return (
        torch.cat(preds).to(device),
        torch.cat(conds).to(device)
    )


def restore_checkpoint(
    model,
    checkpoint_dir,
    checkpoint_name,
    device,
    is_main_node=False,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)

    if "ema_model" in checkpoint:
        model_name = "ema_model"
        if is_main_node:
            print("Will load EMA models for evaluation")
    else:
        model_name = "model"
    base_model.load_state_dict(checkpoint[model_name], strict=True)



def run(
    indir: str = "",
    save_tag: str = "",
    dataset: str = "top",
    path: str = "/pscratch/sd/v/vmikuni/datasets",
    num_feat: int = 3,
    conditional: bool = False,
    num_cond: int = 1,
    batch: int = 64,
    num_workers: int = 16,
):
    local_rank, rank, size = ddp_setup()

    # set up model
    model = MLPGEN(
        input_dim=num_feat,
        hidden_size=256,
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
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")



    data = HLDataset(path)
    test_loader = DataLoader(
        data,
        batch_size=batch,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        sampler=DistributedSampler(data),
        num_workers=num_workers,
        drop_last=False,
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
            rank == 0,
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
        device=device,
        rank=rank,
        save_tag=save_tag,
    )
    dist.barrier()
    dist.destroy_process_group()
