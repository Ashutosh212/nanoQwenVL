# Mac compatibility handoff

Repository: `Ashutosh212/nanoQwenVL`, branch `main`; identity: Ashutosh212 <98666503+Ashutosh212@users.noreply.github.com>.

1. Stop this repo's old Mac sync job before fetching the migration. Preserve all local/staged changes; if a conflicting destination or unrelated outgoing commit exists, stop for manual review. Do not reset, force-push, or recreate the old folders.
2. Replace sparse checkout with a normal full checkout. After reviewing/saving local work, use `git sparse-checkout disable`, fetch and review the migration, then `git pull --ff-only origin main`. This manually reviewed migration legitimately renames/deletes old notes and adjusts root ignore rules; do not grant permanent exceptions for those old paths in the new automatic engine.
3. Update the Mac engine to recurse under `experiment_OPS/` only, including ordinary binary files and intentional dotfiles. Remove extension, UTF-8 and 1 MiB filters. Never silently skip intended files due to global/local/nested ignore rules. Retain secret/100 MiB checks, deletion/conflict/history safeguards, exact-reviewed-commit pushes and personal/company credential separation. No LFS, execution scheduling or blanket repository staging.
4. Review Git attributes: do not execute imported files or install filters. The server preserves file bytes; make equivalent preservation choices on the Mac. Test nested binary/dotfiles and outside-scope/staged-work preservation using isolated repositories first.
5. Open the normal checkout's `experiment_OPS` folder in Obsidian. Remove obsolete sparse patterns and update existing links/automation paths. The personal full checkout also contains the code already tracked in its repository; only OPS is automatic publication scope.
6. Tell the server agent when both Mac jobs are updated. Server services remain held until explicit confirmation. Then run, as appropriate:

```bash
bash /sfs/qwen2.5/scripts/experiment_sync_service.sh resume --mac-updated
bash /sfs/end_to_end/scripts/experiment_sync_service.sh resume --mac-updated
bash /sfs/markdown-sync/sync.sh status
```

Normal subsequent server starts: `bash /sfs/markdown-sync/sync.sh`. Both services retain the 60-second wait after each cycle with no editing-idle delay. Running these resume commands is the explicit acknowledgement that the Mac side has been updated; do not run them early. A genuine conflict still requires manual resolution.
