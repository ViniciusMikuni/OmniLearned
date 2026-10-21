# Validation of the YAML workflow package

Python 3.12, CPU PyTorch 2.11.0. **42 tests passed, 4 deselected** using `not remote and not cuda and not distributed`. See `pytest.log`.

- The complete expected matrix of 69 training and 69 evaluation configs is checked. Every file resolves against the actual command defaults, with matching dataset, size, checkpoint directory and save tag. Fine-tuning references the pretraining tag of the same size. Correct classifier/generator/ftag modes are checked.
- Both shell launchers execute with mocked Slurm commands. Config filenames, relative paths, absolute paths with spaces, missing paths, missing arguments, module loading and rank forwarding are exercised. No real job is launched by these tests.
- Local tests also cover model modes, training/checkpoint/resume/evaluation, metrics, index reuse, Slurm dry-run/mock submission and config snapshotting.
- Ruff lint and formatting checks passed. A wheel built successfully; see `build.log`.
- Three example notebooks and the scientific model/runtime code remain unchanged from the previously delivered version. Notebook execution was not repeated for this workflow-only change.

No actual NERSC/GPU job, full benchmark training or multi-node execution was performed. Batch sizes are editable starting points, not measured capacity guarantees. Remote, CUDA and distributed tests were excluded; the previous cleanup run encountered Gloo socket restrictions.
