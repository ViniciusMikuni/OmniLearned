# OmniLearned SBI: branch dev/nle-perlmutter

Read `PLAN_NLE_PERLMUTTER.md` first. It is the complete brief for this branch: goal, rules,
what exists, lessons learned, milestones and acceptance criteria. Keep the running log
`FINDINGS_NLE.md` at the repository root.

On this branch the toy simulator `toy-quenching-v1` lives in the repository
(`src/omnilearned/sbi/toy/`, config `configs/toy-quenching-v1.toml`). Changing its
`[toy]` or `[selection]` sections defines a new backend id.

- Do not create or modify environments; the user provides an `henv` environment.
- Do not switch branches while training or fitting runs import this checkout.
- Record source revision, dirty status and actual device with every result.
- Do not commit data, checkpoints, run trees, figures or binaries; never force-push.
- Add tests (`tests/sbi`) for every new generator, record format, driver and numerical method.
