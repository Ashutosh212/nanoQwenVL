# Mac compatibility handoff

Repository: `Ashutosh212/nanoQwenVL`, branch `main`; identity: Ashutosh212 <98666503+Ashutosh212@users.noreply.github.com>.

1. Stop this repo's old Mac sync job before fetching the migration. Preserve all local/staged changes; if a conflicting destination or unrelated outgoing commit exists, stop for manual review. Do not reset, force-push, or recreate the old folders.
2. Keep the Mac's intentional directory-only non-cone sparse checkout of `/experiment_obs/` recursively, with all ordinary nested files and dotfiles included. Do not disable sparse checkout, use an extension pattern, or materialize research code in Obsidian. Preserve local work, fetch and review incoming changes, then `git pull --ff-only origin main` only when fast-forward and local-change checks permit. The earlier manually reviewed migration renamed old notes; do not grant permanent old-path publication exceptions.
3. Update the Mac engine to recurse under `experiment_obs/` only, including ordinary binary files and intentional dotfiles. Remove extension, UTF-8 and 1 MiB filters. Never silently skip intended files due to global/local/nested ignore rules. Retain secret/100 MiB checks, deletion/conflict/history safeguards, exact-reviewed-commit pushes and personal/company credential separation. No LFS, execution scheduling or blanket repository staging.
4. Review Git attributes: do not execute imported files or install filters. The server preserves file bytes; make equivalent preservation choices on the Mac. Test nested binary/dotfiles and outside-scope/staged-work preservation using isolated repositories first.
5. Open the sparse checkout's `experiment_obs` folder in Obsidian. Keep directory-only patterns and existing working links; no Mac settings are changed by the server logging setup.
6. Latest user report (2026-10-08): both Mac schedules paused and push credentials unavailable in that session. Server services were observed held during logging setup. Neither saving notes nor fetching this guide verifies a round trip. Resolve authentication and separately verify compatibility before explicitly authorizing resume. Only after that confirmation, run as appropriate:

```bash
bash /sfs/qwen2.5/scripts/experiment_sync_service.sh resume --mac-updated
bash /sfs/end_to_end/scripts/experiment_sync_service.sh resume --mac-updated
bash /sfs/markdown-sync/sync.sh status
```

Normal subsequent server starts: `bash /sfs/markdown-sync/sync.sh`. Both services retain the 60-second wait after each cycle with no editing-idle delay. Running these resume commands is the explicit acknowledgement that the Mac side has been updated; do not run them early. A genuine conflict still requires manual resolution.
