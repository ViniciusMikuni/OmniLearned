import math

import torch

from omnilearned.dataloader import RankSampler
from omnilearned.diffusion import generate
from omnilearned.network import PET2


def test_rank_shards():
    assert sum(
        [list(RankSampler(17, r, 3, shuffle=False)) for r in range(3)], []
    ) == list(range(17))
    samplers = [RankSampler(17, r, 3, seed=9, equalize=True) for r in range(3)]
    first = [list(s) for s in samplers]
    assert all(len(s) == 5 for s in first) and len(set(sum(first, []))) == 15
    for s in samplers:
        s.set_epoch(1)
    assert first != [list(s) for s in samplers]
    assert list(RankSampler(1, 0, 3, shuffle=False)) == []


def test_generation_fixed_counts():
    model = PET2(
        4,
        mode="generator",
        num_classes=1,
        use_int=False,
        base_dim=16,
        num_heads=2,
        num_transformers=1,
        num_transformers_head=1,
        K=4,
    ).eval()
    observed = []

    def hook(module, args, kwargs):
        observed.append(kwargs["mask"].sum(1).flatten().tolist())

    h = model.body.register_forward_pre_hook(hook, with_kwargs=True)
    out = generate(
        model,
        torch.zeros(3, dtype=torch.long),
        (3, 7, 4),
        multiplicity=torch.tensor([0.0, 0.01, 0.05]),
        nsteps=3,
        device="cpu",
    )
    h.remove()
    assert out.shape == (3, 7, 4) and torch.isfinite(out).all()
    assert observed and all(c == [0, 1, 5] for c in observed)
    assert (
        not out[0].count_nonzero()
        and not out[1, 1:].count_nonzero()
        and not out[2, 5:].count_nonzero()
    )


def test_sbi_base_density(monkeypatch):
    import omnilearned.diffusion as diffusion
    from omnilearned.data_spec import noise_parameters

    def velocity(model, z, condition, pid, add_info, y, time, mask=None):
        return z * 0 + condition[:, :1, None]

    monkeypatch.setattr(diffusion, "network_wrapper", velocity)
    x = torch.zeros(1, 1, 4)
    mean, std = noise_parameters(x, "lhco")
    x[:] = mean - 0.4
    r = diffusion.estimate_max_logp(
        torch.nn.Identity(),
        x,
        torch.zeros(1),
        num_param=1,
        cond=torch.tensor([[0.0, 0.01]]),
        multiplicity=torch.tensor([0.01]),
        scan_ranges=[(0.0, 0.8)],
        scan_points=5,
        nsteps=2,
        coarse_steps=2,
        refinements=1,
        probes=1,
        candidate_batch=2,
        seed=4,
        device="cpu",
        dataset="lhco",
    )
    torch.testing.assert_close(r["theta"], torch.tensor([[0.4]]), atol=1e-5, rtol=1e-5)
    assert (
        abs(r["max_logp"] - (-std.log() - 0.5 * math.log(2 * math.pi)).sum().item())
        < 1e-4
    )


def test_base_distributions_match_training_and_generation(monkeypatch):
    import omnilearned.diffusion as diffusion
    from omnilearned.data_spec import noise_parameters

    def zero_velocity(model, z, *args, **kwargs):
        return torch.zeros_like(z)

    monkeypatch.setattr(diffusion, "network_wrapper", zero_velocity)
    for dataset, features in [
        ("jetnet150", 4),
        ("lhco", 4),
        ("lhco_ad", 4),
        ("pretrain", 4),
        ("gaussian_example", 4),
        ("jetnet30", 3),
    ]:
        x = torch.ones(3, 5, features)
        mean, std = noise_parameters(x, dataset)
        torch.manual_seed(7)
        expected = torch.randn_like(x) * std + mean
        torch.manual_seed(7)
        z, target, _ = diffusion.perturb(
            x, torch.ones(3), dataset, mask=torch.ones(3, 5, 1)
        )
        torch.testing.assert_close(z, expected)
        torch.testing.assert_close(target, expected - x)
        torch.manual_seed(7)
        generated = diffusion.generate(
            torch.nn.Identity(),
            torch.zeros(3, dtype=torch.long),
            x.shape,
            dataset=dataset,
            nsteps=2,
            device="cpu",
        )
        torch.testing.assert_close(generated, expected)
