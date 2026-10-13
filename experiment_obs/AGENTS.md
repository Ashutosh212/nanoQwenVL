# Notebook instructions — inherited by every subdirectory

Read `/sfs/research-notebook/logging-policy.md`, its registry, and this
notebook's `agent-guide.md`; preserve this project's root instructions.

## Human and AI authorship — effective 2026-10-14

This rule applies recursively to every directory under each registered `experiment_obs/`,
including root/knowledge notes, month/week notes, daily entries, experiment records,
reports, attachments, templates and future subdirectories. It supersedes earlier
unsuffixed AI write destinations in this guide and linked instructions.

- **Human lane:** unsuffixed files such as `plan.md`, `state.md`, `review.md`,
  `YYYY-MM-DD.md`, and `record.md` are researcher/Obsidian-editable source notes.
  Preserve their contents; never append AI commentary or results to them.
- **AI lane:** write AI-authored text, results, analysis, summaries, plans, state,
  decisions, reviews and new generated notebook evidence to a same-directory,
  same-base-name counterpart with lowercase **`_ai` immediately before the extension**:
  `plan_ai.md`, `state_ai.md`, `review_ai.md`, `YYYY-MM-DD_ai.md`, `record_ai.md`,
  `metrics_ai.json`, `plot_ai.png`. With no extension, end the name in `_ai`.
  Use exactly one lowercase suffix, not `_AI` or `_ai_ai`.
- Create the pair when actually writing a note: keep an existing human file intact;
  if absent, create only an **empty human slot**, not fabricated researcher text.
  Start the AI note with an authorship marker and a link to its human counterpart.
  Empty slots are not evidence of activity or experiment completion.
- Read the human note, relevant legacy evidence and the `_ai` counterpart before work.
  Reference the source note/previous experiment, record timestamps and stable task IDs,
  append observed outcomes and uncertainty, and distinguish proposals from authorization.
  AI notes summarize or respond to the human plan; they do not replace it.
- Keep daily AI results under the actual Asia/Kolkata date in the Sunday-based week:
  `YYYY/YYYY-MM/week-YYYY-MM-DD/daily/YYYY-MM-DD_ai.md`.
  Keep actual experiment setup/results/analysis in that experiment's `record_ai.md`
  and AI-suffixed supporting files; link them from the daily AI note.
  Planning alone still does not create an experiment or authorize running it.
- Obsidian/server synchronization retains these filenames and roles. A human may
  review/edit an `_ai` file, but that does not silently turn its content into a
  researcher-authored unsuffixed file. Copy/adopt material only on explicit request.
- Existing unsuffixed and mixed-author history stays at its current path and remains
  readable evidence. Do not bulk rename it, claim it was wholly written by the human,
  or break historical links. Apply the new split to every future substantive AI write.
- **Fixed-name control exceptions:** `AGENTS.md`, `CLAUDE.md`, `AGENTS.override.md`,
  and `agent-guide.md` retain loader-required names. Their policy edits are allowed
  when authorized; they are instructions, not experiment results or human notes.
  Existing machine-consumed input/protocol/config files keep their required names;
  this rule alone does not authorize editing them. New AI-generated notebook copies
  and reports still use `_ai`. Code, weights and large outputs outside the notebook
  keep their project's normal naming and publication boundaries.

Use `templates/*_ai.md` for AI records. Registries and indexes created/updated by AI
inside the notebook use corresponding `_ai` files; read both lanes when allocating
IDs or collecting evidence so historical records are not missed or double-counted.
No sync watcher or autonomous research execution is added by this rule.
