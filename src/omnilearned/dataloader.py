"""Rank-zero preparation, memory-mapped indices and packed HDF5 batches."""

import fcntl
import itertools
import json
import os
from collections import OrderedDict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import h5py
import numpy as np
import requests
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from omnilearned.data_spec import LABEL_OFFSETS, PRETRAIN_DATASETS, FeatureLayout

BASE_URL = "https://portal.nersc.gov/cfs/dasrepo/omnilearned/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.files = []

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href", "")
        if tag == "a" and href.endswith((".h5", ".hdf5")) and Path(href).name == href:
            self.files.append(href)


def remote_files(url):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code in (404, 410):
            return None
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not check remote dataset at {url}: {error}"
        ) from error
    parser = Links()
    parser.feed(response.text)
    return sorted(set(parser.files))


def download_h5_files(base_url, destination_folder, filenames=None):
    folder = Path(destination_folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in filenames if filenames is not None else remote_files(base_url) or []:
        target = folder / name
        if target.exists():
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        with requests.get(urljoin(base_url, name), stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
        with h5py.File(temporary) as source:
            if "data" not in source or "pid" not in source:
                raise ValueError(f"Downloaded file {name} lacks data/pid")
        temporary.replace(target)


def get_url(dataset_name, dataset_type, base_url=BASE_URL):
    return f"{base_url.rstrip('/')}/{dataset_name}/{dataset_type}/"


def _prepare(folder, name, split, base_url, check_remote):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".file_index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _prepare_locked(folder, name, split, base_url, check_remote)


def _prepare_locked(folder, name, split, base_url, check_remote):
    folder.mkdir(parents=True, exist_ok=True)
    local = sorted([*folder.glob("*.h5"), *folder.glob("*.hdf5")])
    url = get_url(name, split, base_url)
    if check_remote or not local:
        try:
            remote = remote_files(url)
        except RuntimeError:
            if not local:
                raise
            print(f"Remote check failed for {url}; using available local files.")
            remote = None
        if remote:
            download_h5_files(url, folder, remote)
        elif not local:
            raise FileNotFoundError(
                f"Dataset '{name}/{split}' is unavailable: no HDF5 files in {folder} and no dataset files at {url}"
            )
    files = sorted([*folder.glob("*.h5"), *folder.glob("*.hdf5")])
    manifest, index = folder / "file_manifest.json", folder / "file_index.npy"
    if index.exists():
        # The index and manifest are a snapshot. Never rewrite an existing index.
        if not manifest.exists():
            raise ValueError(
                f"{index} exists without {manifest}; restore its manifest or remove the index explicitly"
            )
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, list) or not all(
            isinstance(e, dict)
            and isinstance(e.get("name"), str)
            and Path(e["name"]).name == e["name"]
            and isinstance(e.get("events"), int)
            and e["events"] >= 0
            for e in entries
        ):
            raise ValueError(f"Invalid manifest {manifest}; index was not changed")
        array = np.load(index, mmap_mode="r", allow_pickle=False)
        if (
            array.shape != (sum(e["events"] for e in entries), 2)
            or array.dtype != np.int64
        ):
            raise ValueError(
                f"Invalid index {index}; restore it or remove it explicitly"
            )
        missing = [e["name"] for e in entries if not (folder / e["name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Indexed files missing in {folder}: {missing}")
        added = sorted({f.name for f in files} - {e["name"] for e in entries})
        if added:
            print(
                f"Reusing {index}; {len(added)} new files are not indexed. Remove file_index.npy explicitly to include them.",
                flush=True,
            )
        return
    entries = []
    for file in files:
        with h5py.File(file) as source:
            if "data" not in source or "pid" not in source:
                raise ValueError(f"Missing data/pid in {file}")
            if (
                source["data"].ndim != 3
                or source["data"].shape[-1] < 3
                or len(source["pid"]) != len(source["data"])
            ):
                raise ValueError(f"Invalid particle / label dimensions in {file}")
            entries.append(
                {
                    "name": file.name,
                    "events": len(source["data"]),
                    "size": file.stat().st_size,
                    "mtime_ns": file.stat().st_mtime_ns,
                }
            )
    temporary = folder / f"file_index.{os.getpid()}.tmp.npy"
    array = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.int64,
        shape=(sum(e["events"] for e in entries), 2),
    )
    start = 0
    for i, entry in enumerate(entries):
        stop = start + entry["events"]
        array[start:stop, 0] = i
        array[start:stop, 1] = np.arange(entry["events"])
        start = stop
    del array
    temp = manifest.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(entries, indent=2))
    temp.replace(manifest)
    temporary.replace(index)
    print(f"Created {index}: {start:,} events in {len(entries):,} files", flush=True)


def prepare_dataset(
    dataset_name, path, dataset_type="train", base_url=BASE_URL, check_remote=True
):
    error = None
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            for name in (
                PRETRAIN_DATASETS if dataset_name == "pretrain" else [dataset_name]
            ):
                _prepare(
                    Path(path) / name / dataset_type,
                    name,
                    dataset_type,
                    base_url,
                    check_remote,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        status = [error]
        dist.broadcast_object_list(status, src=0)
        error = status[0]
    if error:
        raise RuntimeError(error)


class IndexTable:
    def __init__(self, paths):
        self.paths = list(map(str, paths))
        self.maps = [np.load(p, mmap_mode="r") for p in paths]
        self.ends = np.cumsum([len(a) for a in self.maps])
        self.file_offsets = np.cumsum(
            [0]
            + [
                len(json.loads(Path(p).with_name("file_manifest.json").read_text()))
                for p in paths[:-1]
            ]
        )

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        group = int(np.searchsorted(self.ends, index, side="right"))
        start = self.ends[group - 1] if group else 0
        fid, row = self.maps[group][int(index - start)]
        return int(fid + self.file_offsets[group]), int(row)

    def __getstate__(self):
        return {"paths": self.paths}

    def __setstate__(self, state):
        self.__init__(state["paths"])


class HEPDataset(Dataset):
    def __init__(
        self,
        file_paths,
        file_indices,
        use_cond=False,
        use_pid=False,
        pid_idx=4,
        use_add=False,
        num_add=4,
        clip_inputs=False,
        mode="",
        nevts=-1,
        global_only=False,
        zero_add=False,
        max_open_files=8,
        dataset_name="",
    ):
        self.file_paths, self.file_indices = list(map(str, file_paths)), file_indices
        self.use_cond, self.use_pid, self.use_add = use_cond, use_pid, use_add
        self.pid_idx, self.num_add = pid_idx, num_add
        self.clip_inputs, self.mode = clip_inputs, mode
        self.nevts, self.global_only, self.zero_add = nevts, global_only, zero_add
        self.max_open_files = max_open_files
        self.label_shift = LABEL_OFFSETS.get(dataset_name, 0)
        self._file_cache, self._pid = OrderedDict(), os.getpid()

    def __len__(self):
        return (
            len(self.file_indices)
            if self.nevts < 0
            else min(self.nevts, len(self.file_indices))
        )

    def close(self):
        for f in self._file_cache.values():
            f.close()
        self._file_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_cache"] = OrderedDict()
        return state

    def _get_file(self, i):
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if i not in self._file_cache:
            while len(self._file_cache) >= self.max_open_files:
                _, old = self._file_cache.popitem(last=False)
                old.close()
            self._file_cache[i] = h5py.File(self.file_paths[i], "r")
        self._file_cache.move_to_end(i)
        return self._file_cache[i]

    def _sample(self, fid, row, arrays):
        # Match upstream: pretrain preserves global IDs; standalone subtracts.
        label = np.asarray(arrays["pid"], dtype=np.float32) - self.label_shift
        out = {
            "y": torch.as_tensor(np.asarray(label), dtype=torch.float32),
            "file_id": fid,
            "row_id": row,
        }
        if "global" in arrays:
            out["cond"] = torch.as_tensor(arrays["global"], dtype=torch.float32)
        if self.global_only:
            return out
        x = torch.as_tensor(arrays["data"], dtype=torch.float32)
        out["stored_shape"] = tuple(x.shape)
        mask = x[:, 2] != 0
        if self.clip_inputs:
            mask &= (torch.hypot(x[:, 0], x[:, 1]) < 0.8) & (x[:, 2] > -0.7)
            x[:, 3] = torch.maximum(x[:, 3], x[:, 2])
        positions = mask.nonzero().flatten()
        x = x[positions]
        pid_idx = self.pid_idx % x.shape[-1] if self.use_pid else None
        layout = FeatureLayout(
            x.shape[-1], pid_idx, self.num_add if self.use_add else 0
        )
        out["X"], out["positions"] = x[:, layout.physical], positions
        if self.use_pid:
            out["pid"] = x[:, pid_idx].long()
        if self.use_add:
            out["add_info"] = x[:, layout.additional]
            if self.zero_add:
                out["add_info"] = torch.zeros_like(out["add_info"])
        if "data_pid" in arrays:
            out["data_pid"] = torch.as_tensor(
                arrays["data_pid"],
                dtype=torch.float32 if self.mode == "segmentation" else torch.long,
            )[positions]
        return out

    def __getitems__(self, indices):
        groups, result = {}, [None] * len(indices)
        for order, index in enumerate(indices):
            fid, row = self.file_indices[index]
            groups.setdefault(fid, []).append((order, row))
        for fid, requests_ in groups.items():
            source = self._get_file(fid)
            rows, inverse = np.unique([r for _, r in requests_], return_inverse=True)
            keys = ["pid"] + ([] if self.global_only else ["data"])
            if self.use_cond:
                keys.append("global")
            if self.mode in ("ftag", "segmentation"):
                keys.append("data_pid")
            selection = (
                slice(int(rows[0]), int(rows[-1]) + 1)
                if len(rows) == rows[-1] - rows[0] + 1
                else rows
            )
            arrays = {k: source[k][selection] for k in keys}
            for (order, row), j in zip(requests_, inverse, strict=False):
                result[order] = self._sample(
                    fid, row, {k: v[j] for k, v in arrays.items()}
                )
        return result

    def __getitem__(self, index):
        return self.__getitems__([index])[0]


def collate_point_cloud(batch, max_part=None, k=None):
    out = {
        "y": torch.stack([b["y"] for b in batch]),
        "file_id": torch.tensor([b["file_id"] for b in batch]),
        "row_id": torch.tensor([b["row_id"] for b in batch]),
    }
    out["cond"] = (
        torch.stack([b["cond"] for b in batch])
        if all("cond" in b for b in batch)
        else None
    )
    if "X" not in batch[0]:
        return out
    lengths = torch.tensor([len(b["X"]) for b in batch], dtype=torch.long)
    out.update(
        lengths=lengths,
        offsets=torch.cat((lengths.new_zeros(1), lengths.cumsum(0))),
        stored_shape=torch.tensor([b["stored_shape"] for b in batch]),
    )
    for key in ("X", "pid", "add_info", "data_pid", "positions"):
        out[key] = (
            torch.cat([b[key] for b in batch]) if all(key in b for b in batch) else None
        )
    return out


class RankSampler(Sampler):
    """Fixed per-dataset rank shards; shuffle only this rank's events."""

    def __init__(
        self,
        size,
        rank=0,
        world_size=1,
        shuffle=True,
        seed=42,
        limit=-1,
        dataset_ends=None,
        equalize=False,
    ):
        if not 0 <= rank < world_size:
            raise ValueError("Expected 0 <= rank < world_size")
        self.size, self.rank, self.world_size = size, rank, world_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.limit = limit
        self.dataset_ends = np.asarray(
            [size] if dataset_ends is None else dataset_ends, dtype=np.int64
        )
        self.starts = np.concatenate(([0], self.dataset_ends[:-1]))
        counts = self.dataset_ends - self.starts
        self.local_counts = np.maximum(
            0, (counts + world_size - 1 - rank) // world_size
        )
        self.local_ends = np.cumsum(self.local_counts)
        self.local_size = int(self.local_counts.sum())
        self.equal_size = (
            int((counts // world_size).sum()) if equalize else self.local_size
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self.shuffle:
            n = self.equal_size
        else:
            n = (
                self.size * (self.rank + 1) // self.world_size
                - self.size * self.rank // self.world_size
            )
        return min(n, self.limit) if self.limit >= 0 else n

    def __iter__(self):
        if not self.shuffle:
            start = self.size * self.rank // self.world_size
            yield from range(start, start + len(self))
            return
        if len(self) == 0:
            return
        # One compact rank-local permutation, not a global event permutation.
        dtype = np.int32 if self.local_size <= np.iinfo(np.int32).max else np.int64
        order = np.arange(self.local_size, dtype=dtype)
        np.random.default_rng(self.seed + self.epoch).shuffle(order)
        local_starts = np.concatenate(([0], self.local_ends[:-1]))
        # Translate in bounded chunks; never build a full list of event tuples.
        for start in range(0, len(self), 65536):
            selected = order[start : min(start + 65536, len(self))]
            group = np.searchsorted(self.local_ends, selected, side="right")
            indices = (
                self.starts[group]
                + self.rank
                + (selected - local_starts[group]) * self.world_size
            )
            yield from map(int, indices)


class BucketBatchSampler(Sampler):
    def __init__(self, sampler, dataset, batch_size, window):
        self.sampler, self.dataset, self.batch_size = sampler, dataset, batch_size
        self.window = max(batch_size, window // batch_size * batch_size)

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        iterator = iter(self.sampler)
        while window := list(itertools.islice(iterator, self.window)):
            lengths = []
            for index in window:
                fid, row = self.dataset.file_indices[index]
                lengths.append(
                    np.count_nonzero(self.dataset._get_file(fid)["data"][row, :, 2])
                )
            ordered = [window[i] for i in np.argsort(lengths, kind="stable")]
            for start in range(0, len(ordered), self.batch_size):
                yield ordered[start : start + self.batch_size]


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
    num_workers=0,
    rank=0,
    size=1,
    clip_inputs=False,
    mode="",
    shuffle=True,
    nevts=-1,
    seed=42,
    base_url=BASE_URL,
    check_remote=True,
    persistent_workers=False,
    prefetch_factor=1,
    max_open_files=8,
    global_only=False,
    zero_add=False,
    bucket_size=0,
):
    if min(batch, prefetch_factor, max_open_files) < 1 or num_workers < 0:
        raise ValueError("Invalid batch, cache, prefetch or worker setting")
    prepare_dataset(dataset_name, path, dataset_type, base_url, check_remote)
    folders = (
        [
            Path(path) / name / dataset_type
            for name in PRETRAIN_DATASETS
            if dataset_name == "pretrain"
        ]
        if dataset_name == "pretrain"
        else [Path(path) / dataset_name / dataset_type]
    )
    entries = [
        (folder, e)
        for folder in folders
        for e in json.loads((folder / "file_manifest.json").read_text())
    ]
    files = [folder / e["name"] for folder, e in entries]
    indices = IndexTable([folder / "file_index.npy" for folder in folders])
    data = HEPDataset(
        files,
        indices,
        use_cond,
        use_pid,
        pid_idx,
        use_add,
        num_add,
        clip_inputs,
        mode,
        global_only=global_only,
        zero_add=zero_add,
        max_open_files=max_open_files,
        dataset_name=dataset_name,
    )
    sampler = RankSampler(
        len(data),
        rank if distributed else 0,
        size if distributed else 1,
        shuffle,
        seed,
        nevts,
        dataset_ends=indices.ends,
        equalize=dataset_type == "train",
    )
    options = (
        {"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor}
        if num_workers
        else {}
    )
    if bucket_size and shuffle:
        options["batch_sampler"] = BucketBatchSampler(sampler, data, batch, bucket_size)
    else:
        options.update(batch_size=batch, sampler=sampler)
    loader = DataLoader(
        data,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_point_cloud,
        **options,
    )
    loader.rank_sampler = sampler
    return loader
