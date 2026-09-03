#!/usr/bin/env bash
# Run a command inside a kgfm container and hand its output back to the host user.
#
# Containers run as root, so every file they write to a bind mount lands owned
# by root on the host. The next step outside the container -- git add, an
# editor, scripts/make_report.py -- then fails on files it cannot touch. The
# trap below runs on every exit path, a SIGTERM from `docker stop` included, so
# an interrupted run still hands its partial output back.
#
# Only the directories a run writes are chowned. A recursive chown of the whole
# work tree would walk .git as well, which is slow and needs to happen never.
#
#   usage: in_container.sh <command> [args...]
set -uo pipefail

hand_back() {
  status=$?
  if [ -n "${HOST_UID:-}" ] && [ -n "${HOST_GID:-}" ]; then
    for d in /kgfm-src/ranks /kgfm-src/results /kgfm-src/output \
             /kgfm-src/data/roots /kgfm-src/data/raw /kgfm/output /kgfm/ranks; do
      [ -d "$d" ] && chown -R "$HOST_UID:$HOST_GID" "$d" 2>/dev/null
    done
  fi
  exit $status
}
trap hand_back EXIT INT TERM

# A container killed mid-build (docker stop) leaves torch's extension build
# lock behind, and the next container waits on it forever: one hour lost on
# 2026-09-03. A lock older than ten minutes cannot belong to a live build.
find /kgfm-src/output -maxdepth 3 -path '*torch_extensions*' -name lock \
     -mmin +10 -delete 2>/dev/null

# Deliberately not exec: exec replaces this shell and the trap never fires.
"$@"
