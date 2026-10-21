"""Central dataset conventions."""

from dataclasses import dataclass

LABEL_OFFSETS = {
    "jetclass": 2,
    "jetclass2": 12,
    "aspen": 200,
    "cms_qcd": 201,
    "cms_bsm": 202,
}
PRETRAIN_DATASETS = (
    "atlas",
    "aspen",
    "jetclass",
    "jetclass2",
    "h1",
    "cms_qcd",
    "cms_bsm",
)


@dataclass(frozen=True)
class FeatureLayout:
    width: int
    pid_idx: int | None = None
    num_add: int = 0

    @property
    def physical(self):
        c = [i for i in range(self.width) if i != self.pid_idx]
        return c[: -self.num_add] if self.num_add else c

    @property
    def additional(self):
        c = [i for i in range(self.width) if i != self.pid_idx]
        return c[-self.num_add :] if self.num_add else []


def noise_parameters(x, dataset):
    mean, std = [0.0] * x.shape[-1], [1.0] * x.shape[-1]
    if x.shape[-1] == 4:
        if "jetnet" in dataset:
            mean, std = (
                [0.0, 0.0, 2.92904309, 3.14843288],
                [0.11580968, 0.11667726, 0.95442127, 0.98245158],
            )
        elif dataset == "lhco":
            mean, std = (
                [0.0, 0.0, 1.11398, 1.43384],
                [0.270949, 0.27426, 1.30273, 1.33559],
            )
    return x.new_tensor(mean), x.new_tensor(std)


def multiplicity_counts(multiplicity, maximum):
    counts = (multiplicity * 100).round().long().reshape(-1)
    if ((counts < 0) | (counts > maximum)).any():
        raise ValueError(f"Multiplicity must represent 0..{maximum} particles")
    return counts
