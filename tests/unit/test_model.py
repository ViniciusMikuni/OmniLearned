import pytest
import torch

from omnilearned.attention import Attention
from omnilearned.layers import NoScaleDropout
from omnilearned.neighbors import knn
from omnilearned.network import PET2
from omnilearned.packing import offsets_from_mask
from omnilearned.utils import CLIPLoss, get_loss, get_model_parameters


def small(**kwargs):
    return PET2(
        4,
        base_dim=16,
        num_heads=2,
        num_transformers=1,
        num_transformers_head=1,
        K=4,
        **kwargs,
    )


def test_sdpa_matches_mha():
    torch.manual_seed(2)
    old = torch.nn.MultiheadAttention(16, 2, batch_first=True, bias=False)
    new = Attention(16, 2)
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[False, False, True, True, True], [False] * 5])
    expected = old(x, x, x, key_padding_mask=mask, need_weights=False)[0]
    actual = new(x, x, x, key_padding_mask=mask)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), x)[0],
        torch.autograd.grad(expected.square().sum(), x)[0],
    )


@pytest.mark.parametrize("interaction", [False, True])
def test_padding_and_packing(interaction):
    torch.manual_seed(2)
    model = small(use_int=interaction, local_int=True).eval()
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])[..., None]
    x *= mask
    y = torch.zeros(2)
    with torch.no_grad():
        a = model(x, y)["y_pred"]
        b = model(torch.nn.functional.pad(x, (0, 0, 0, 5)), y)["y_pred"]
        c = model(x[mask.squeeze(-1)], y, offsets=offsets_from_mask(mask))["y_pred"]
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, c)


def test_knn_edges():
    points = torch.tensor(
        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [999.0, 999.0]], [[0.0, 0.0]] * 4]
    )
    mask = torch.tensor([[True, True, True, False], [False] * 4])[..., None]
    idx, live = knn(points, mask, 6, chunk_size=1)
    idx2, live2 = knn(points, mask, 6)
    assert torch.equal(idx, idx2) and torch.equal(live, live2)
    idx = idx.reshape(2, 4, 6)
    for center in range(3):
        assert set(idx[0, center][live[0, center, :, 0]].tolist()) == set(range(3)) - {
            center
        }
    assert not live[1].any() and not live[0, 3].any()


def test_dropout_per_event():
    offsets = torch.tensor([0, 2, 5, 9])
    out = NoScaleDropout(0.5).train()(torch.ones(9, 4), offsets)
    for start, stop in zip(offsets[:-1], offsets[1:], strict=False):
        assert torch.all(out[start:stop] == out[start])
    assert set(out.unique().tolist()) <= {0.0, 1.0}


def test_qk_normalization():
    assert all(
        m.qk_scale is None
        for m in small(use_int=False).modules()
        if isinstance(m, Attention)
    )
    model = small(use_int=False, qk_norm=True)
    model(torch.randn(2, 3, 4), torch.zeros(2))["y_pred"].square().sum().backward()
    for m in model.modules():
        if isinstance(m, Attention):
            assert m.qk_scale is not None and torch.isfinite(m.qk_scale.grad).all()


def test_clip_weight_shape():
    x, z = torch.randn(4, 8), torch.randn(4, 8)
    w = torch.tensor([1.0, 0.2, 0.5, 0.7])
    loss = CLIPLoss()
    torch.testing.assert_close(loss(x, z, w), loss(x, z, w[:, None, None]))


def test_pretrain_regression_both_heads():
    out = {
        "x": torch.ones(2, 1, 4),
        "mask": torch.ones(2, 1, 1, dtype=torch.bool),
        "y_pred": torch.tensor([[1.0], [3.0]]),
        "y_perturb": torch.tensor([[2.0], [4.0]]),
    }
    assert get_loss(out, torch.tensor([0.0, 2.0]), "pretrain_regression") == 5


def test_diffusion_counts_not_velocity():
    out = {
        "x": torch.zeros(1, 2, 4),
        "mask": torch.tensor([[[True], [False]]]),
        "v": torch.zeros(1, 2, 4),
        "v_weight": torch.ones(1, 2, 4),
        "z_pred": torch.ones(1, 2, 4),
    }
    assert get_loss(out, torch.zeros(1), "generator") == 4


def test_decoder_condition():
    model = small(use_int=False, mode="encoder", num_particles=5).eval()
    z = torch.randn(1, 10)
    a, _ = model.decoder(z, torch.tensor([0]))
    b, _ = model.decoder(z, torch.tensor([1]))
    assert not torch.allclose(a, b)


def test_presets():
    for size, expected in [
        ("small", (8, 2, 128, 8)),
        ("medium", (12, 2, 512, 16)),
        ("large", (28, 4, 1024, 32)),
    ]:
        p = get_model_parameters(size)
        assert (
            tuple(
                p[k]
                for k in (
                    "num_transformers",
                    "num_transformers_head",
                    "base_dim",
                    "num_heads",
                )
            )
            == expected
        )
    model = PET2(4, use_int=False, **get_model_parameters("dense"))
    assert not any(isinstance(m, Attention) for m in model.modules())


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_native_varlen_cuda():
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Native varlen needs Ampere or newer")
    offsets = torch.tensor([0, 2, 5], device="cuda")
    model = Attention(32, 2, backend="varlen").cuda().bfloat16()
    x = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    actual = model(x, x, x, offsets=offsets)[0]
    model.backend = "sdpa"
    expected = model(x, x, x, offsets=offsets)[0]
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    actual.float().square().sum().backward()
    assert torch.isfinite(x.grad).all()


def test_ot_without_local_and_padding():
    torch.manual_seed(2)
    model = small(use_int=True, int_type="ot", use_local=False).eval()
    x = torch.randn(2, 5, 4)
    x[0, 2:] = 0
    y = torch.zeros(2)
    with torch.no_grad():
        a = model(x, y)["y_pred"]
        b = model(torch.nn.functional.pad(x, (0, 0, 0, 3)), y)["y_pred"]
    torch.testing.assert_close(a, b)


def test_reconstruction_dispatch(monkeypatch):
    import omnilearned.utils as utils

    calls = []

    def fn(name):
        def loss(pred, x, mask, logits):
            calls.append((name, mask.sum(1).flatten().tolist()))
            return pred.sum() * 0

        return loss

    monkeypatch.setattr(utils, "emd_loss", fn("emd"))
    monkeypatch.setattr(utils, "wd_loss", fn("wd"))
    mask = torch.arange(250)[None, :, None] < torch.tensor([3, 201])[:, None, None]
    out = {
        "x": torch.zeros(2, 250, 4),
        "mask": mask,
        "x_hat": torch.zeros(2, 240, 4),
        "mask_logits": torch.zeros(2, 240),
    }
    assert torch.isfinite(get_loss(out, torch.zeros(2), "encoder"))
    assert calls == [("emd", [3]), ("wd", [201])]


def test_checkpoint_roundtrip():
    from omnilearned.checkpoint import load_weights

    a = small(use_int=False, mode="encoder", num_particles=4).eval()
    b = small(use_int=False, mode="encoder", num_particles=4).eval()
    load_weights(b, {"model": a.state_dict()})
    latent, y = torch.randn(2, 10), torch.tensor([0, 1])
    for x, z in zip(a.decoder(latent, y), b.decoder(latent, y), strict=False):
        torch.testing.assert_close(x, z)
    a, b = small(use_int=False).eval(), small(use_int=False).eval()
    load_weights(
        b, {"body": a.body.state_dict(), "classifier_head": a.classifier.state_dict()}
    )
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(a(x, y)["y_pred"], b(x, y)["y_pred"])


def test_activation_checkpointing():
    a, b = small(use_int=False), small(use_int=False, activation_checkpointing=True)
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(2, 4, 4), torch.zeros(2)
    a(x, y)["y_pred"].sum().backward()
    b(x, y)["y_pred"].sum().backward()
    for p, q in zip(a.parameters(), b.parameters(), strict=False):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


def test_fine_tune_encoder_from_legacy_pretraining():
    from omnilearned.checkpoint import load_weights

    pretrained = small(use_int=False, mode="pretrain")
    encoder = small(use_int=False, mode="encoder", num_particles=4)
    initial_decoder = {k: v.clone() for k, v in encoder.decoder.state_dict().items()}
    load_weights(
        encoder,
        {
            "body": pretrained.body.state_dict(),
            "classifier_head": pretrained.classifier.state_dict(),
        },
        fine_tune=True,
    )
    torch.testing.assert_close(
        encoder.body.embed.mlp.fc1.weight, pretrained.body.embed.mlp.fc1.weight
    )
    for key, value in initial_decoder.items():
        torch.testing.assert_close(encoder.decoder.state_dict()[key], value)
