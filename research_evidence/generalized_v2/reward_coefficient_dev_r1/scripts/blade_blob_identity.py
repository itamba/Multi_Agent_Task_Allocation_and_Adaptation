"""Git-blob identity of the IMPORTED BLADE engine against the measured tree (standard library).

``prelaunch_verify.py``'s byte-level tree digest compares working-copy bytes, which differ
between two checkouts only by line endings under ``core.autocrlf=true`` (the same condition
the PR #79 pre-launch recorded). This check compares what Git would store: ``git hash-object``
(which applies the repository's clean filter) of every file of the imported copy and of the
measured worktree, against ``git ls-tree -r <measured sha>``; it also confirms each byte
difference is EOL-only.

Usage: python blade_blob_identity.py <repo> <measured sha> <imported blade dir> <worktree blade dir> <out>
"""

import json
import subprocess
import sys
from pathlib import Path

REL = "src/match_aou/integrations/panopticon-main/gym/blade"


def main() -> int:
    repo, sha, imported, worktree, out = sys.argv[1:6]
    tree = subprocess.run(["git", "ls-tree", "-r", sha, REL + "/"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.splitlines()
    blobs = {line.split("\t", 1)[1][len(REL) + 1:]: line.split()[2] for line in tree}

    def hashes(root):
        root = Path(root)
        files = sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        out = {}
        for p in files:
            rel = p.relative_to(root).as_posix()
            out[rel] = subprocess.run(["git", "hash-object", "--path", REL + "/" + rel, str(p)],
                                      cwd=repo, capture_output=True, text=True,
                                      check=True).stdout.strip()
        return out

    hi, hw = hashes(imported), hashes(worktree)
    eol_only = []
    for rel in sorted(set(hi) & set(hw)):
        a = (Path(imported) / rel).read_bytes()
        b = (Path(worktree) / rel).read_bytes()
        if a != b:
            eol_only.append({"file": rel,
                             "eol_only": a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")})
    rec = {
        "check": "blade_engine_git_blob_identity", "measured_sha": sha,
        "n_tree_files": len(blobs), "imported_copy": str(imported), "worktree_copy": str(worktree),
        "imported_matches_measured_blobs": hi == blobs,
        "worktree_matches_measured_blobs": hw == blobs,
        "mismatches_imported": sorted(k for k in set(hi) | set(blobs) if hi.get(k) != blobs.get(k)),
        "byte_differences_between_copies": len(eol_only),
        "all_byte_differences_eol_only": all(d["eol_only"] for d in eol_only),
    }
    rec["pass"] = (rec["imported_matches_measured_blobs"] and rec["worktree_matches_measured_blobs"]
                   and rec["all_byte_differences_eol_only"])
    Path(out).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("PASS" if rec["pass"] else "FAIL", json.dumps({k: rec[k] for k in rec if k != "mismatches_imported"}))
    return 0 if rec["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
