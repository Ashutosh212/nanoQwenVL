# Research agent guide

## Before proposing work

1. Read [state](state.md) and [long-term plan](plan.md).
2. Read accumulated [findings](knowledge/findings.md) and [decisions](knowledge/decisions.md), including superseded entries, and relevant [ideas](knowledge/ideas.md).
3. Compute today's date in Asia/Kolkata; read that calendar month's plan/state and the preceding calendar month's review if it exists. If missing, explicitly record that gap; do not invent or precreate old reviews.
4. Find the current Sunday-based week under its Sunday's month. Read its plan/state and the previous week's review (possibly in another month/year), when present.
5. Follow evidence links to original daily records, experiment records, commands and artifacts relevant to the proposed work. Read the applicable research subproject's AGENTS.md and CLAUDE.md before execution.
6. Distinguish planned, running, completed, failed and unknown. Notes, saved prompts and sync events never authorize execution. Ask for explicit execution approval when not already requested.

## Recording actual work

Preserve the original prompt verbatim where available, with source and timestamp; otherwise write “unknown — original prompt unavailable.” Record actual changes, exact commands and environment, configuration/seeds/data versions, code commit and dirty state, note commit separately, results/metrics, failures, artifact locations and next steps. No invented commands, metrics or historical experiments.

Keep original evidence in the actual experiment directory: `EXP-YYYY-MM-DD-NN-description/record.md` plus supporting files/subfolders. Use [template](templates/experiment.md). Put day-level chronology and links in the daily file; planning-only tasks need no EXP directory. Record artifact hashes/paths when large artifacts stay elsewhere.

Link every conclusion to the original evidence. Mark missing evidence as unknown. Do not treat notes-only Git revisions as versions of ignored research code. Preserve superseded findings and decisions with dates, reasons, and replacement links rather than overwriting history.

## Reviews

Weekly periods are Sunday through Saturday in Asia/Kolkata. A cross-month week stays in the starting Sunday's month. Monthly reviews must search all overlapping weeks, including those filed under a neighboring month/year, and count only work whose actual date belongs to the reviewed month. Link source records rather than copying or double-counting results. State incomplete coverage and missing reviews.

## Publication

Only `experiment_OPS/` is automatic publication scope. Every ordinary nested file, including dotfiles and binary evidence, is eligible. Review sensitive content before saving here. Never include credentials; stop and report suspected credentials, oversized files, symlinks, nested repositories, conflicts or accidental deletions. Do not introduce LFS, force-push, reset, rewrite history, or execute a file merely because it was pulled. The current Mac migration hold must remain until user confirmation.
