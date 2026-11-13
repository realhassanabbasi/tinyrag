"""Refresh the library — re-run the enhancer over every saved prompt.

`check` guards the library: it re-scores each prompt and flags regressions.
`refresh` is the other half — it FIXES them. For every saved name it takes
the keeper text (the enhanced version when saved with --improve, else the
prompt), runs the improve pipeline on it, and saves the rebuild as a new
version ONLY when the rebuilt score beats the recorded best. A zero or
negative margin saves nothing: a rewrite that only looks better but scores
the same is a wasted rewrite (the same rule the exported skill teaches in
its step 3 — demand a positive margin before shipping).

`promptsmith refresh`              -> refresh every saved prompt
`promptsmith refresh NAME`         -> refresh one saved prompt
`promptsmith refresh --tags a,b`   -> refresh only names carrying a tag
`promptsmith refresh --dry-run`    -> show what would improve, save nothing
`--json`                           -> machine-readable results (keeps script
                                      output pure when piped into CI)
"""

from .check import keeper_text
from .improve import improve_prompt
from .library import entries_for, filter_by_tags, load_library, save_entry

VERDICTS = ("improved", "unchanged")


def refresh_name(name, records=None, cwd=None, dry_run=False):
    """Re-improve one saved prompt's keeper text and save the rebuild.

    Returns a result dict, or None when nothing is saved under `name`.
    A new version is saved only when the rebuilt score beats the recorded
    best — otherwise the verdict is "unchanged" and nothing is written.
    The enhancer is idempotent on its own output, so a second refresh
    right after a successful one always reports unchanged: refresh
    converges instead of stacking versions.
    """
    versions = entries_for(name, records, cwd)
    if not versions:
        return None
    keeper = keeper_text(versions[-1])
    if not keeper.strip():
        return None
    best = max(v.get("score", 0) for v in versions)
    result = improve_prompt(keeper)
    after = result["after"]["score"]
    delta = after - best
    tags = []
    for v in versions:
        for t in v.get("tags") or []:
            if t not in tags:
                tags.append(t)
    improved = delta > 0
    saved_version = None
    if improved and not dry_run:
        record = save_entry(name, result["enhanced"], tags=tags, cwd=cwd)
        saved_version = record["version"]
    return {
        "name": name,
        "versions": len(versions),
        "before": best,
        "after": after,
        "delta": delta,
        "verdict": "improved" if improved else "unchanged",
        "saved_version": saved_version,
        "tags": tags,
    }


def refresh_library(records=None, cwd=None, tags=None, dry_run=False):
    """Refresh every saved prompt, biggest gains first. Empty library -> [].

    `tags` filters the table to names carrying at least one of the tags —
    the same any-match, case-insensitive rule as `library --tags` and
    `check --tags` (filter_by_tags is the single source of truth). The
    filter applies AFTER improving, so a name's recorded best still counts
    every version, not just the tagged ones.
    """
    if records is None:
        records = load_library(cwd)
    names = []
    for r in records:
        n = r.get("name", "")
        if n and n not in names:
            names.append(n)
    results = [refresh_name(n, records, cwd, dry_run=dry_run)
               for n in names]
    results = [r for r in results if r is not None]
    results = filter_by_tags(results, tags)
    results.sort(key=lambda r: (-r["delta"], -r["before"]))
    return results


def refresh_payload(results):
    """Machine-readable envelope: per-prompt results plus totals."""
    counts = {v: 0 for v in VERDICTS}
    for r in results:
        counts[r["verdict"]] += 1
    return {
        "results": results,
        "refreshed": len(results),
        **counts,
    }
