#!/usr/bin/env bash
# Verify that a Thunder snapshot really contains this project.
#
#   On the ORIGINAL instance, right before `tnr snapshot create`:
#       scripts/snapshot_check.sh make     -> writes SNAPSHOT_MANIFEST.txt (in the project, so it is snapshotted too)
#   On a NEW instance restored from that snapshot:
#       scripts/snapshot_check.sh verify   -> compares every file against the manifest, prints OK / MISSING / CHANGED
#
# Only delete the original instance after `verify` prints "ALL OK" on the restored instance.
# Model weights / venv are checked by size only (hashing 60 GB is slow); code, data and results by sha256.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAN=SNAPSHOT_MANIFEST.txt

list() {
  find . -type f ! -name "$MAN" ! -path './.venv/*' ! -path '*/__pycache__/*' ! -path './models/*/.cache/*' -printf '%P\n' | sort |
  while read -r f; do
    sz=$(stat -c %s "$f")
    case "$f" in
      models/*) echo "size:$sz  $f" ;;
      *)        echo "$(sha256sum "$f" | cut -c1-64)  $f" ;;
    esac
  done
  echo "venv_files:$(find .venv -type f | wc -l)  .venv"
}

case "${1:-}" in
  make)
    sync
    { echo "# created $(date -u +%FT%TZ) on $(hostname)"; list; } > "$MAN"
    sync
    echo "wrote $MAN: $(grep -vc '^#' "$MAN") entries, project size $(du -sh . | cut -f1)"
    echo "now run on your laptop:  tnr snapshot create   then wait for  tnr snapshot list  -> READY" ;;
  verify)
    [ -f "$MAN" ] || { echo "FAIL: $MAN not found -> snapshot does not contain the project"; exit 1; }
    diff <(grep -v '^#' "$MAN") <(list) > /tmp/snapshot_diff.txt && { head -1 "$MAN"; echo "ALL OK ($(grep -vc '^#' "$MAN") entries match)"; } || {
      echo "MISMATCH (< = in manifest, > = on this disk):"; cat /tmp/snapshot_diff.txt | head -50; exit 1; } ;;
  *) echo "usage: $0 make|verify"; exit 1 ;;
esac
