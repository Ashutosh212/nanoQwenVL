# Research agent guide

## Before proposing work

1. Read [state](state.md) and [long-term plan](plan.md).
2. Read accumulated [findings](knowledge/findings.md) and [decisions](knowledge/decisions.md), including superseded entries, and relevant [ideas](knowledge/ideas.md).
3. Compute today's date in Asia/Kolkata; read that calendar month's plan/state and the preceding calendar month's review if it exists. If missing, explicitly record that gap; do not invent or precreate old reviews.
4. Find the current Sunday-based week under its Sunday's month. Read its plan/state and the previous week's review (possibly in another month/year), when present.
5. Follow evidence links to original daily records, experiment records, commands and artifacts relevant to the proposed work. Read the applicable research subproject's AGENTS.md and CLAUDE.md before execution.
6. Distinguish planned, running, completed, failed and unknown. Notes, saved prompts and sync events never authorize execution. Ask for explicit execution approval when not already requested.

## Recording actual work

Read and follow `/sfs/research-notebook/logging-policy.md` and `projects.json` on the server. They are the editable shared source; this is its concise project-local copy (v1, 2026-10-08). Resolve canonical cwd/explicit targets to the most specific registered ancestor. Nested code repos use this notebook but record their actual code root/SHA separately. Ask once for ambiguous/outside-registry research work. Keep multi-project entries scoped separately with a shared correlation ID; no company research in personal notes.

For every substantive relevant request (planning, useful explanation, debugging, edits, tests or experiments), automatically append a concise intent summary to today's daily file without a save reminder. At start record timestamp, stable task/session ID, cwd, agent and authorization. Reuse that ID for follow-up corrections/milestones/outcomes; do not duplicate the request/header. Across days link the original task. Skip greetings/duplicates and honor explicit read-only/no-file-edits requests.

At milestones and before the final response record actual actions/files, essential executed commands, observed results/evidence, what worked/failed, unknowns and next proposed steps. Classify failure causes as observed, supported inference or unknown. Long-running work stays started/pending until the outcome is observed; reconcile interrupted work from real evidence. Never invent results or log hidden instructions, private reasoning, credentials, unrelated chats or unobserved actions. Full prompts are not required; quote only essential safe evidence. End substantive responses with a short link to updated notes and any logging/sync failure.

Reread before edits, append historical evidence, preserve other writers and stop on conflicts. The optional server helper `/sfs/research-notebook/notebook.py` supplies local locking, expected-content checks and atomic writes; it cannot coordinate Mac edits. Saved, committed and pushed are distinct evidence-based statuses. Do not start services or execute plans by reading/saving notes.

Keep original evidence in the actual experiment directory: `EXP-YYYY-MM-DD-NN-description/record.md` plus supporting files/subfolders. Use [template](templates/experiment.md). Put day-level chronology and links in the daily file; planning-only tasks need no EXP directory. Record artifact hashes/paths when large artifacts stay elsewhere.

Allocate EXP numbers uniquely across this project/date, checking existing records and serializing creation. Retain the original experiment folder across weeks; actual work dates are separate from creation dates. Record hypothesis, baseline/success criterion, approval, real code repo/revision/dirty state versus notebook revision, environment/data/model/config versions/seeds, actual commands/times, measured units/denominators, evidence and uncertainty. Keep oversized artifacts outside with locations/hashes; never relocate research data/code for logging.

Link every conclusion to the original evidence. Mark missing evidence as unknown. Do not treat notes-only Git revisions as versions of ignored research code. Preserve superseded findings and decisions with dates, reasons, and replacement links rather than overwriting history.

## Reviews

Weekly periods are Sunday through Saturday in Asia/Kolkata. A cross-month week stays in the starting Sunday's month. Monthly reviews must search all overlapping weeks, including those filed under a neighboring month/year, and count only work whose actual date belongs to the reviewed month. Link source records rather than copying or double-counting results. State incomplete coverage and missing reviews.

After tasks refresh weekly state; update root/month state only when materially changed, with as-of times/evidence links. Update the relevant actual experiment record if any. Findings require evidence; ideas remain untested; append superseding decisions with reasons. Compare planned versus actual work, successes/failures, comparable configurations, reversals and gaps; do not average incompatible metrics. Maintain in-progress reviews when evidence changes. On the first session after a period closes, complete unfinished/missing reviews from available evidence and declare missing coverage. No session means no guaranteed review; no scheduler or autonomous experiment runner is installed.

## Publication

Only `experiment_obs/` is automatic publication scope. Every ordinary nested file, including dotfiles and binary evidence, is eligible. Review sensitive content before saving here. Never include credentials; stop and report suspected credentials, oversized files, symlinks, nested repositories, conflicts or accidental deletions. Do not introduce LFS, force-push, reset, rewrite history, or execute a file merely because it was pulled. The current Mac migration hold must remain until user confirmation.

Mac intentionally uses directory-only non-cone sparse checkout of `/experiment_obs/` recursively (all ordinary files, no extension filter), not full checkout. As reported by the user on 2026-10-08, both Mac schedules were paused and push credentials unavailable in that session. Server jobs were observed held during setup; inspect current status before claiming live sync. Do not resume or change transport/credentials as part of logging. Server-global policy and root instruction edits stay outside automatic publication.
