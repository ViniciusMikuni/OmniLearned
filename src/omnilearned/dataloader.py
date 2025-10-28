import torch
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path


def collate_point_cloud(batch, max_part=150):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]

    # Stack once to avoid repeated slicing
    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)

    # Use validity mask based on feature index 2
    valid_mask = point_clouds[:, :, 2] != 0
    max_particles = min(valid_mask.sum(dim=1).max().item(), max_part)
    # max_particles = point_clouds.shape[1]

    # Truncate point clouds
    truncated_X = point_clouds[:, :max_particles, :]  # (B, M, F)
    result = {"X": truncated_X, "y": labels}

    # Handle optional fields in a loop to reduce code duplication
    optional_fields = ["cond", "pid", "add_info", "data_pid", "vertex_pid"]
    for field in optional_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            # Truncate if it's sequence-like (i.e., has 2 or more dims)
            if stacked.dim() >= 2 and stacked.shape[1] >= max_particles:
                stacked = stacked[:, :max_particles]
            result[field] = stacked
        else:
            result[field] = None

    return result


def get_url(dataset_name, dataset_type, base_url="https://portal.nersc.gov/cfs/m4567/"):
    url = f"{base_url}/{dataset_name}/{dataset_type}/"
    try:
        requests.head(url, allow_redirects=True, timeout=5)
        return url
    except requests.RequestException:
        print(
            "ERROR: Request timed out, visit https://www.nersc.gov/users/status for status on  portal.nersc.gov"
        )
        return None


def download_h5_files(base_url, destination_folder):
    """
    Downloads all .h5 files from the specified directory URL.

    Args:
        base_url (str): The base URL of the directory containing the .h5 files.
        destination_folder (str): The local folder to save the downloaded files.
    """

    response = requests.get(base_url)
    if response.status_code != 200:
        print(f"Failed to access {base_url}")
        return

    file_links = re.findall(r'href="([^"]+\.h5)"', response.text)

    for file_name in file_links:
        file_url = urljoin(base_url, file_name)
        file_path = os.path.join(destination_folder, file_name)

        print(f"Downloading {file_url} to {file_path}")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {file_name}")


class HEPDataset(Dataset):
    def __init__(
        self,
        file_paths,
        file_indices=None,
        use_cond=False,
        use_pid=False,
        pid_idx=4,
        use_add=False,
        num_add=4,
        label_shift=0,
        clip_inputs=False,
        ftag=False,
    ):
        """
        Args:
            file_paths (list): List of file paths.
            use_pid (bool): Flag to select if PID information is used during training
            use_add (bool): Flags to select if additional information besides kinematics are used
        """
        self.use_cond = use_cond
        self.use_pid = use_pid
        self.use_add = use_add
        self.pid_idx = pid_idx
        self.num_add = num_add
        self.label_shift = label_shift

        self.file_paths = file_paths
        self._file_cache = {}  # lazy cache for open h5py.File handles
        self.file_indices = file_indices
        self.clip_inputs = clip_inputs
        self.ftag = ftag

        # random.shuffle(self.file_indices)  # Shuffle data entries globally

    def __len__(self):
        return len(self.file_indices)

    def _get_file(self, file_idx):
        # Get the file handle from cache; open it if it’s not already open.
        if file_idx not in self._file_cache:
            file_path = self.file_paths[file_idx]
            self._file_cache[file_idx] = h5py.File(file_path, "r")
        return self._file_cache[file_idx]

    def __getitem__(self, idx):
        file_idx, sample_idx = self.file_indices[idx]
        f = self._get_file(file_idx)

        sample = {}

        sample["X"] = torch.tensor(f["data"][sample_idx], dtype=torch.float32)
        if self.clip_inputs:
            # Enforce particles to be inside R=0.8 and pT > 0.5 MeV
            mask_part = (torch.hypot(sample["X"][:, 0], sample["X"][:, 1]) < 0.8) & (
                sample["X"][:, 2] > 0.0
            )
            sample["X"][:, 3] = np.clip(
                sample["X"][:, 3], a_min=sample["X"][:, 2], a_max=None
            )
            sample["X"] = sample["X"] * mask_part.unsqueeze(-1).float()

        label = f["pid"][sample_idx]
        sample["y"] = torch.tensor(label - self.label_shift, dtype=torch.int64)
        if "global" in f and self.use_cond:
            sample["cond"] = torch.tensor(f["global"][sample_idx], dtype=torch.float32)

        if self.use_pid:
            sample["pid"] = sample["X"][:, self.pid_idx].int()
            sample["X"] = torch.cat(
                (sample["X"][:, : self.pid_idx], sample["X"][:, self.pid_idx + 1 :]),
                dim=1,
            )
        if self.use_add:
            # Assume any additional info appears last
            sample["add_info"] = sample["X"][:, -self.num_add :]
            sample["X"] = sample["X"][:, : -self.num_add]
        if self.ftag:
            sample["data_pid"] = torch.tensor(
                f["data_pid"][sample_idx], dtype=torch.int64
            )

        return sample

    def __del__(self):
        # Clean up: close all cached file handles.
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception as e:
                print(f"Error closing file: {e}")


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    distributed=True,
    use_cond=False,
    use_pid=False,
    pid_idx=4,
    use_add=False,
    num_add=4,
    num_workers=16,
    rank=0,
    size=1,
    clip_inputs=False,
    ftag=False,  # special flag for atlas ftag
    shuffle=True,
):
    supported_datasets = [
        "top",
        "qg",
        "pretrain",
        "atlas",
        "aspen",
        "jetclass",
        "jetclass2",
        "h1",
        "toy",
        "cms_qcd",
        "cms_bsm",
        "cms_top",
        "aspen_bsm",
        "aspen_top_ad_sb",
        "aspen_top_ad_sr",
        "aspen_top_ad_sr_hl",
        "jetnet150",
        "jetnet30",
        "dctr",
        "atlas_flav",
        "custom",
    ]
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}."
        )

    if dataset_name == "pretrain":
        names = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
        types = [dataset_type]
    else:
        names = [dataset_name]
        types = [dataset_type]

    dataset_paths = [os.path.join(path, name, type) for name in names for type in types]

    file_list = []
    file_indices = []
    index_shift = 0
    for iname, dataset_path in enumerate(dataset_paths):
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        if not any(dataset_path.iterdir()):
            print(f"Fetching download url for dataset {names[iname]}")
            url = get_url(names[iname], dataset_type)
            if url is None:
                raise ValueError(f"No download URL found for dataset '{dataset_name}'.")
            download_h5_files(url, dataset_path)

        h5_files = list(dataset_path.glob("*.h5"))
        file_list.extend(map(str, h5_files))  # Convert to string paths

        index_file = dataset_path / "file_index.npy"
        if index_file.is_file():
            if shuffle:
                indices = np.load(index_file, mmap_mode="r")[rank::size]
            else:
                indices = np.load(index_file, mmap_mode="r")[
                    len(np.load(index_file, mmap_mode="r")) * rank // size : len(
                        np.load(index_file, mmap_mode="r")
                    )
                    * (rank + 1)
                    // size
                ]
            file_indices.extend(
                (file_idx + index_shift, sample_idx) for file_idx, sample_idx in indices
            )
            index_shift += len(h5_files)

        else:
            print(f"Creating index list for dataset {names[iname]}")
            file_indices = []
            # Precompute indices for efficient access
            for file_idx, path in enumerate(h5_files):
                try:
                    with h5py.File(path, "r") as f:
                        num_samples = len(f["data"])
                        file_indices.extend([(file_idx, i) for i in range(num_samples)])
                except Exception as e:
                    print(f"ERROR: File {path} is likely corrupted: {e}")
            np.save(index_file, np.array(file_indices, dtype=np.int32))
            print(f"Number of events: {len(file_indices)}")

    # Shift labels if they are not used for pretrain
    label_shift = {
        "jetclass": 2,
        "jetclass2": 12,
        "aspen": 200,
        "cms_qcd": 201,
        "cms_bsm": 202,
    }

    data = HEPDataset(
        file_list,
        file_indices,
        use_cond=use_cond,
        use_pid=use_pid,
        pid_idx=pid_idx,
        use_add=use_add,
        num_add=num_add,
        label_shift=label_shift.get(dataset_name, 0),
        clip_inputs=clip_inputs,
        ftag=ftag,
    )

    loader = DataLoader(
        data,
        batch_size=batch,
        pin_memory=torch.cuda.is_available(),
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_point_cloud,
    )
    return loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="top",
        help="Dataset name to download",
    )
    parser.add_argument(
        "-f",
        "--folder",
        default="./",
        help="Folder to save the dataset",
    )
    args = parser.parse_args()

    for tag in ["train", "test", "val"]:
        load_data(args.dataset, args.folder, dataset_type=tag, distributed=False)
