# Notebook migration validation — 2026-10-08

- Scope: explicitly requested migration from `experiments/` to `experiment_OPS/`,
  recursive ordinary-file sync, calendar hierarchy and agent guidance. No model
  experiment, dependency installation, sparse server checkout or LFS was added.
- Source revision and exact file mappings/hashes: [manifest](migration-manifest.json).
  Original dated notes were moved byte-for-byte. Test/uncertain material remains
  under legacy. No original Markdown links required relocation.
- The old engine/configuration and notes were backed up on persistent storage
  at `/sfs/markdown-sync/backups/20261008T083148Z/`; credentials were not copied.
- Shared validation: 58 tests passed (16 launcher + 42 recursive sync/calendar/
  migration). The 42 shared integration tests also passed through each project's
  documented interpreter. Shell syntax checks passed. No model suite was needed
  because research implementation was not changed.
- Tested nested binary/text files, direct children, dotfiles, executable regular
  files without execution, ignore overrides, secrets and oversized files,
  symlinks/path escapes, nested repos/submodules, deletion/conflict/divergence,
  concurrent edits and commits, staged-work preservation, migration collisions,
  original-byte preservation and Kolkata Sunday/month/year boundaries.
- Ignore configuration audit found no configured global exclusion file or
  checkout filters. Root exceptions now include all OPS descendants; existing
  protections elsewhere remain. The engine also discovers ignored descendants
  directly so future nested/local/global ignores cannot silently omit them.
- Publication is a separately reviewed manual commit covering notebook moves,
  scaffold and `.gitignore` only. Existing unrelated edits stay local. Normal
  automatic publication still forbids all outside paths and old-path deletions.
- Both services remain PAUSED, with a persistent Mac-update hold. The morning
  command does not bypass it. Mac full checkout, equivalent recursive engine,
  and post-update cross-device round-trip remain unverified. See
  [handoff](../mac-handoff.md) before explicit resume.
- Original model experiment prompts/metrics that were not recorded remain
  unknown; the migration does not infer them from notebook transport tests.
