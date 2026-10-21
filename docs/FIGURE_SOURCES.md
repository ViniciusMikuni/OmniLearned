# Figure sources

Figures are reproduced from the original TeX source submissions on arXiv. PDF figures were rasterized for browser display without changing plotted values, labels, or legends. Original figure PDFs are included alongside the PNG previews where available. Rights remain with the respective authors.

| Highlight | Paper | Source assets |
| --- | --- | --- |
| OmniLearned collider results | https://arxiv.org/abs/2510.24066 | `top_tagging_rej30.pdf`, `btag.pdf`, `ctag.pdf`, `ad_cathode_m.pdf`, `fine_tune_ad_cathode_m.pdf` |
| OmniCosmos / CAMELS-SAM | https://arxiv.org/abs/2512.24422v2 | `figures/camels_0.pdf`, `figures/camels_1.pdf` (Figure 2) |
| OmniMol speed / accuracy | https://arxiv.org/abs/2601.10791v2 | `images/OmniMolSpeed.png` |
| Neutrino energy regression | https://arxiv.org/abs/2604.12364v2 | `figures/scaling/regression/iqr_E_resolution_plot_1A.pdf` |
| Trigger distillation | https://arxiv.org/abs/2609.26899v1 | `roc_deepsets.pdf` |
| Molecule time evolution | User-supplied `viz4.mp4` | Top panel only: crop 1080 × 568 at (0, 6); original timing retained, silent H.264 MP4 |

Source archives: https://arxiv.org/src/ followed by the respective paper identifier. Retrieved 21 October 2026.

## Wording and comparisons

- CAMELS-SAM: the less-than-half-data highlight refers specifically to the sigma8 comparison in Figure 2: 100 versus 600 training examples for the plotted reference benchmarks. The Omega_m comparison uses 300 examples. These are comparisons of central R² values, not claims of a statistically significant advantage over every fully trained model.
- OmniMol: approximately 3× faster is the paper's OmniMol-m comparison with eSEN-md-d and AllScAIP-md on one A100-40GB GPU and systems of order 100 atoms. AllScAIP-md has lower energy error; the plot retains that trade-off.
- Neutrinos: the figure shows available hadronic-energy residual resolution and bias in MINERvA playlist 1A.
- Trigger: the paper demonstrates distillation and 8-bit quantization-aware training toward FPGA deployment. It explicitly states that synthesis, resource and latency optimization are still in progress. No measured FPGA latency is claimed on the page.
- Hero particles are decorative, not a measured or simulated physics event.

Architecture: `plots/PET2.png` uses the original repository’s `assets/PET2.png`, the compact version without panel letters.
