"""toy-quenching-v1: invented jet-quenching simulator for Route A closure tests.

Ported unchanged from the omni-sbi-mc application (a/src/jet_quenching_inference)
for the self-contained ``dev/nle-perlmutter`` development. See
PLAN_NLE_PERLMUTTER.md. Modules:

* ``simulator``: the generator (``sample_toy``, ``ToyConstants``, ``Selection``).
* ``reco``: anti-kt reconstruction with FastJet through hepyy/cppyy.
* ``selection``: the same selection applied to generated clouds, and the
  surrogate selection-acceptance table.
* ``observables``: jet observables and a KS statistic.
* ``design``: event-id streams, seeded generators, reserved settings.
* ``records``: HDF5 record writers.
* ``config``: load ``configs/toy-quenching-v1.toml``; ``sample_setting`` and
  ``array_sha256`` for reproducible draws.
"""

from omnilearned.sbi.toy.config import array_sha256, load_toy_config, sample_setting  # noqa: F401
