#!/usr/bin/env bash
# Publish this project's artifacts to Hugging Face.
#
#   HF_NAMESPACE=<user-or-org> HF_TOKEN=<write token> bash tools/hf_publish.sh [--dry-run] [--private]
#
# Three repos are created (idempotently) under $HF_NAMESPACE:
#
#   <ns>/alignn-csp-ablation-dvc        dataset  the DVC remote, byte-for-byte:
#                                                 what `dvc pull` reads.  Opaque
#                                                 content-addressed layout.
#   <ns>/alignn-csp-angular-ablations   model    the trained checkpoints, one
#                                                 folder per (experiment set,
#                                                 dataset, cell), with config,
#                                                 history, metrics and the
#                                                 ABLATION.yaml beside each --
#                                                 the browsable mirror.
#   <ns>/alignn-csp-ablation-datasets   dataset  the prepared splits and the
#                                                 raw Alexandria inputs -- the
#                                                 browsable mirror.
#
# The DVC remote is the one that matters for reproducibility (a clone runs
# `bash tools/hf_sync.sh pull`).  The other two exist so a person, or a model
# reading the Hub, can see what is there without DVC.  hf/model and hf/datasets
# are staged by tools/write_metadata.py (hardlinks into the DVC-tracked tiers).
#
# Requires: dvc + huggingface_hub (the tooling env /data/ccamp104/envs/repo-tools-x86
# on atomgptlab, or `pip install dvc huggingface_hub`).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${HF_NAMESPACE:?set HF_NAMESPACE (Hugging Face user or org)}"
: "${HF_TOKEN:?set HF_TOKEN (a write token)}"
HF="${HF_BIN:-$(command -v hf || echo /data/ccamp104/envs/repo-tools-x86/bin/hf)}"
DVC="${DVC_BIN:-$(command -v dvc || echo /data/ccamp104/envs/repo-tools-x86/bin/dvc)}"
DRY=0; PRIV=""
for a in "$@"; do case "$a" in --dry-run) DRY=1;; --private) PRIV="--private";; esac; done
run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
REV=$(git rev-parse --short HEAD)

REMOTE_DIR=$("$DVC" remote list | awk '$1=="hfmirror"{print $2}')
[ -d "$REMOTE_DIR" ] || { echo "DVC remote dir not found: $REMOTE_DIR (run: dvc push)" >&2; exit 1; }
[ -d hf/model ] && [ -d hf/datasets ] || { echo "hf/ staging missing: run tools/write_metadata.py first" >&2; exit 1; }

# 1. the DVC remote -----------------------------------------------------------
run "$HF" repo create "$HF_NAMESPACE/alignn-csp-ablation-dvc" --repo-type dataset $PRIV --exist-ok
run "$HF" upload "$HF_NAMESPACE/alignn-csp-ablation-dvc" "$REMOTE_DIR" . --repo-type dataset \
    --commit-message "dvc remote mirror $(date -Is) ($REV)"
run "$HF" upload "$HF_NAMESPACE/alignn-csp-ablation-dvc" hf/cards/dvc-README.md README.md --repo-type dataset

# 2. browsable weights --------------------------------------------------------
run "$HF" repo create "$HF_NAMESPACE/alignn-csp-angular-ablations" --repo-type model $PRIV --exist-ok
run "$HF" upload "$HF_NAMESPACE/alignn-csp-angular-ablations" hf/model . --repo-type model \
    --commit-message "checkpoints + per-cell metadata $(date -Is) ($REV)"

# 3. browsable datasets -------------------------------------------------------
run "$HF" repo create "$HF_NAMESPACE/alignn-csp-ablation-datasets" --repo-type dataset $PRIV --exist-ok
run "$HF" upload "$HF_NAMESPACE/alignn-csp-ablation-datasets" hf/datasets . --repo-type dataset \
    --commit-message "prepared splits + raw inputs $(date -Is) ($REV)"

echo
echo "published under https://huggingface.co/$HF_NAMESPACE"
echo "next: sed -i 's/<HF_NAMESPACE>/$HF_NAMESPACE/g' PROJECT_STATE.md .dvc/config hf/cards/*.md && git commit -am 'record HF namespace'"
