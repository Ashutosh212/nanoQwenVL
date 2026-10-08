# Project state

As of notebook preparation on 2026-10-08, Asia/Kolkata:

- Existing notes migrated without inventing research results; [manifest](legacy/migration-manifest.json).
- Notebook root is now `experiment_obs/`; server engine supports ordinary files recursively.
- Both server services are paused with a persistent Mac-update hold. Publication/revision details belong in the handoff and Git history; this file does not claim a completed Mac update.
- Mac intentionally keeps directory-only non-cone sparse checkout of `/experiment_obs/`
  recursively. Latest user report: Mac schedules paused and push credentials unavailable;
  live end-to-end verification remains pending. Do not disable sparse checkout.
- Active research jobs, latest model metrics and external artifact availability: unknown; not inspected by this infrastructure task.
- As of 2026-10-08 16:12 IST: automatic summary-first agent instructions and
  notebook templates are installed; 22 helper tests passed. Fresh Codex logging
  verification is blocked by sandbox startup, Claude by expired OAuth. See
  [setup evidence](2026/2026-10/week-2026-10-04/daily/2026-10-08.md).
- Historical CPU validation is recorded in the October 6 daily note; training and GPU results were not established there.
