"""Jet observables computed from constituent features (eta, phi, log pT)."""

import numpy as np

NAMES = {
    "jet_pt": "jet pT [GeV]",
    "width": "pT-weighted <dR> to jet axis",
    "soft8_fraction": "pT fraction of the 8 softest constituents",
    "outer_fraction": "pT fraction at dR > 0.15",
    "ptd": "pT_D",
    "lead_fraction": "leading constituent pT fraction",
    "jet_eta": "jet eta",
}


def jet_observables(x):
    """x: [N, P, 3] physical features of fully contained jets."""
    x = np.asarray(x, dtype=np.float64)
    eta, phi, pt = x[..., 0], x[..., 1], np.exp(x[..., 2])
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    jpx, jpy, jpz = px.sum(1), py.sum(1), pz.sum(1)
    jpt = np.hypot(jpx, jpy)
    jeta = np.arcsinh(jpz / jpt)
    jphi = np.arctan2(jpy, jpx)
    dphi = np.angle(np.exp(1j * (phi - jphi[:, None])))
    dr = np.hypot(eta - jeta[:, None], dphi)
    spt = pt.sum(1)
    srt = np.sort(pt, 1)
    return {
        "jet_pt": jpt,
        "jet_eta": jeta,
        "width": (pt * dr).sum(1) / spt,
        "soft8_fraction": srt[:, :8].sum(1) / spt,
        "outer_fraction": (pt * (dr > 0.15)).sum(1) / spt,
        "ptd": np.sqrt((pt ** 2).sum(1)) / spt,
        "lead_fraction": srt[:, -1] / spt,
    }


def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    d = np.abs(ca - cb).max()
    n = a.size * b.size / (a.size + b.size)
    # Asymptotic Kolmogorov p-value.
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d
    k = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)), 0, 1))
    return float(d), p
