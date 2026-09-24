"""Ledger of the committed evidence files, checked against Git blob ids (standard library).

For every file of the package that is STAGED in the index (``git ls-files -s``), records its
SHA-256 and byte size, the index blob id, and whether that blob is the file's exact bytes
(``git hash-object --no-filters``, "raw") or its autocrlf-filtered form ("filtered", files staged
through the repository filter before launch); anything else is a mismatch.
For ``run_artifacts/`` copies it also checks byte identity against the external original.
``committed_files.txt`` does not list itself. Exits non-zero on any mismatch.

Usage: python committed_ledger.py <repo root> <package rel path> <run dir> <launcher dir>
"""

import hashlib
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo, rel, run_dir, launcher = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    staged = subprocess.run(["git", "ls-files", "-s", "--", rel], cwd=repo, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    rows, bad = [], []
    for line in staged:
        meta, path = line.split("\t", 1)
        blob = meta.split()[1]
        if path.endswith("/committed_files.txt"):
            continue
        p = repo / path
        data = p.read_bytes()
        raw = subprocess.run(["git", "hash-object", "--no-filters", str(p)], cwd=repo,
                             capture_output=True, text=True, check=True).stdout.strip()
        filtered = subprocess.run(["git", "hash-object", "--path", path, str(p)], cwd=repo,
                                  capture_output=True, text=True, check=True).stdout.strip()
        # "raw": the blob IS the file's bytes; "filtered": the file was staged earlier through
        # the repository's autocrlf clean filter (CRLF working copy of an LF blob)
        mode = "raw" if raw == blob else "filtered" if filtered == blob else "MISMATCH"
        ok = mode != "MISMATCH"
        sub = path[len(rel) + 1:]
        orig_ok = "-"
        if sub.startswith("run_artifacts/"):
            name = sub[len("run_artifacts/"):]
            src = (launcher / name[len("launcher/"):]) if name.startswith("launcher/") else run_dir / name
            orig_ok = "identical" if src.exists() and src.read_bytes() == data else "DIFFERENT"
            ok = ok and orig_ok == "identical"
        rows.append("%s  %d  %s  %s  %s  %s" % (hashlib.sha256(data).hexdigest(), len(data),
                                               blob, mode, orig_ok, sub))
        if not ok:
            bad.append(sub)
    out = repo / rel / "committed_files.txt"
    out.write_text("# sha256  bytes  git_blob  blob_identity  vs_external_original  path (relative"
                   " to the package)\n"
                   "# blob_identity raw = the blob is the file's exact bytes; filtered = the blob is"
                   " the file after the repository's autocrlf clean filter\n"
                   + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print("files", len(rows), "mismatches", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
