# Generic OmniLearned SBI framework

This source worktree belongs to development a on dev/agent-a.
Its paired physics application is ../jest-a.
Read ../jest-a/AGENTS.md and ../jest-a/configs/local.toml for
the assigned environment, framework path, branch and application handoff.
Read the paired application's docs/sbi-handoff/CODING_AGENT_PLAN.md for
the reusable library requirements and acceptance tests.

Implement reusable likelihood evaluation, generic SBI scans, model/loader
interfaces, MPS support and generic numerical tests here. Keep the toy
generator, JEWEL production, physics studies, data and reports in the paired
application. Do not create another environment config here. Do not switch
branches during training, reinstall packages, move environments or edit
another development. Record source revision, dirty status and actual device.
