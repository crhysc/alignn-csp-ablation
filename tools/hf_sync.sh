#!/usr/bin/env bash
# Move the DVC remote between a machine and Hugging Face.
#
#   bash tools/hf_sync.sh pull   # new machine: fetch the remote, point DVC at it, `dvc pull`
#   bash tools/hf_sync.sh push   # after `dvc add` + `dvc push`: mirror the remote dir to the Hub
#
# Needs HF_NAMESPACE (and HF_TOKEN for push, or for pull if the repo is private).
#
# Why this exists: DVC has no native Hugging Face remote (checked 2026-09-06,
# dvc 3.67 -- `hf://` is "Unsupported URL type", and its http remote is
# pull-only against a content-addressed layout).  So the DVC remote is a plain
# directory, and this script mirrors that directory to/from a Hub dataset repo.
# Everything DVC-shaped stays DVC-shaped; the Hub is only transport.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${HF_NAMESPACE:?set HF_NAMESPACE}"
HF="${HF_BIN:-$(command -v hf || echo /data/ccamp104/envs/repo-tools-x86/bin/hf)}"
DVC="${DVC_BIN:-$(command -v dvc || echo /data/ccamp104/envs/repo-tools-x86/bin/dvc)}"
REPO="$HF_NAMESPACE/alignn-csp-ablation-dvc"
case "${1:-}" in
  pull)
    DIR="${DVC_REMOTE_DIR:-$PWD/.dvc-remote}"
    "$HF" download "$REPO" --repo-type dataset --local-dir "$DIR"
    "$DVC" remote modify hfmirror url "$DIR"
    "$DVC" pull
    "$DVC" status -c || true
    echo "artifacts restored; see PROJECT_STATE.md for what they are"
    ;;
  push)
    DIR=$("$DVC" remote list | awk '$1=="hfmirror"{print $2}')
    "$DVC" push
    "$HF" upload "$REPO" "$DIR" . --repo-type dataset --commit-message "dvc remote mirror $(date -Is) ($(git rev-parse --short HEAD))"
    ;;
  *) echo "usage: $0 {pull|push}" >&2; exit 2;;
esac
