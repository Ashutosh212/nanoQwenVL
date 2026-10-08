# Research notebook — Ashutosh212/nanoQwenVL

This is the current notebook root: `experiment_obs/`. Start with [agent guide](agent-guide.md), [plan](plan.md), [state](state.md), and [findings](knowledge/findings.md).

- GitHub: https://github.com/Ashutosh212/nanoQwenVL; branch: `main`.
- Commit identity: Ashutosh212 <98666503+Ashutosh212@users.noreply.github.com>. Authentication is separate for each project.
- Current [monthly plan](2026/2026-10/plan.md), [weekly plan](2026/2026-10/week-2026-10-04/plan.md), and [daily record](2026/2026-10/week-2026-10-04/daily/2026-10-08.md).
- All ordinary files recursively in this root are eligible for automatic sync, including binary attachments and intentional dotfiles. There is no Markdown, UTF-8, extension, or 1 MiB restriction.
- Nothing outside this root may be automatically committed/pushed. Legacy names are not permanent publication exceptions.
- GitHub blocks regular-Git files above 100 MiB and warns above 50 MiB. Oversized files stop publication; no LFS or storage is installed automatically. [GitHub limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
- Credentials, symlinks, special files, nested Git repositories, conflicting/deleted/staged notebook files and unrelated/divergent commits stop synchronization for review. Credential recognition is not a comprehensive secret detector (especially encrypted/compressed material). Inspect attachments before publication.
- Saved files do not authorize commands, agents, training, evaluation or scheduled LLM work.

## Calendar and hierarchy

Dates use Asia/Kolkata. A week is Sunday 00:00 through Saturday 23:59:59, named for the starting Sunday and kept under that Sunday's month, even across year boundaries. Monthly reviews attribute work by the actual work date and link overlapping weeks in neighboring months; never duplicate the underlying daily or experiment record.

```text
experiment_obs/
  README.md, agent-guide.md, plan.md, state.md
  knowledge/{findings,decisions,ideas}.md
  templates/{experiment,daily,weekly-review,monthly-review}.md
  YYYY/YYYY-MM/
    plan.md, state.md, review.md
    week-YYYY-MM-DD/
      plan.md, state.md, review.md
      daily/YYYY-MM-DD.md
      experiments/EXP-YYYY-MM-DD-NN-description/record.md
  legacy/
```

Create only the current/needed periods. Create an EXP directory only for an actual experiment, never to imply an unrun experiment occurred. Supporting files can be any ordinary type and may have subfolders. Use relative Markdown links so they work in GitHub and Obsidian; images: `![caption](relative/path.png)`.

## Migration and sync state

Server synchronization is deliberately PAUSED until the Mac has a full checkout and the new recursive OPS engine. Old notes are preserved with [source paths, hashes and revisions](legacy/migration-manifest.json); uncertain notes stay in [legacy](legacy/README.md). Historical commands/paths in old notes describe that past setup, not current instructions.

Server launcher: `bash /sfs/markdown-sync/sync.sh status`. The morning start command does not bypass the Mac-migration hold. See [Mac handoff](mac-handoff.md) for the required coordinated update. No provider boot hook or autonomous experiment runner is installed.
