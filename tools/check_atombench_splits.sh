#!/usr/bin/env bash
# Rebuild AtomBench's splits by running AtomBench's own preprocessing code, and
# check with tools/check_atombench_splits.py that ALIGNN-CSP trains and scores
# on exactly those splits.
#
#   bash tools/check_atombench_splits.sh
#
# Writes a <check>.json report and <check>.log transcript per comparison, and
# summary.txt, to atombench_split_check/ (git-tracked).  The bulky inputs --
# the AtomBench checkout, jarvis-tools, dft_3d, the Alexandria tables, the
# regenerated splits -- stay in WORK (default .atombench-split-check/,
# git-ignored), so a rerun reuses them.  Exits 1 if any comparison fails.
#
# Needs datasets/ pulled (tools/hf_sync.sh pull), network access, and a python
# with pymatgen, pandas and tqdm: PY, or SCORE_ENV_PATH from site.env.
#
# Every input is pinned:
#   ATOMBENCH_COMMIT  324ed9d, AtomBench HEAD.  Its split code
#                     (tc_supercon/scripts/data_preprocess.py and
#                     alexandria/scripts/alexandria_preprocess.py) last changed
#                     in b046069 (2026-02-09); with aa1e878 (2026-01-19) that
#                     added the JARVIS hygiene: drop JVASP-19919 before the
#                     shuffle, and the leaked JVASP-20425/16080 from test.
#   JARVIS_TOOLS      jarvis-tools versions to build the JARVIS split with.
#                     2026.1.10 is the one AtomBench's docs/reproducing.md pins
#                     (dft_3d = jdft_3d-12-12-2022.json); 2026.6.12 is the one in
#                     this project's environments (jdft_3d-9-24-2025.json).
#                     Alexandria is built with the first; there jarvis-tools
#                     only writes POSCARs.
#
# AtomBench's run_*_data.sh are not executed as they are: they move their
# outputs into models/, which is an uninitialised submodule here.  The
# preprocessing command each one contains is run instead, with the same
# arguments; only --output differs, and it does not enter the split.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO/site.env" ]; then
    set +u
    # shellcheck disable=SC1091
    HARNESS_ID=none . "$REPO/site.env" 2>/dev/null
    set -u
fi
PY="${PY:-${SCORE_ENV_PATH:+$SCORE_ENV_PATH/bin/python}}"
: "${PY:?set PY to a python with pymatgen, pandas and tqdm (or SCORE_ENV_PATH in site.env)}"

WORK="${WORK:-$REPO/.atombench-split-check}"
OUT="${OUT:-$REPO/atombench_split_check}"
ATOMBENCH_URL="${ATOMBENCH_URL:-https://github.com/atomgptlab/atombench.git}"
ATOMBENCH_COMMIT="${ATOMBENCH_COMMIT:-324ed9d}"
JARVIS_TOOLS="${JARVIS_TOOLS:-2026.1.10 2026.6.12}"
ALEX_JT="${JARVIS_TOOLS%% *}"
ALEX_RAW_URL="https://github.com/hyllios/utils/raw/refs/heads/main/models/supercond_modnet_model"

JARVIS_SPLIT="$REPO/datasets/jarvis_supercon3d"
ALEX_SPLIT="$REPO/datasets/alexandria_dsab"
for d in "$JARVIS_SPLIT" "$ALEX_SPLIT"; do
    [ -f "$d/test.json" ] || { echo "missing $d/test.json -- pull datasets/ first (tools/hf_sync.sh pull)" >&2; exit 1; }
done

mkdir -p "$WORK/splits" "$OUT"
log() { printf '\n== %s\n' "$*"; }

# ---- AtomBench at ATOMBENCH_COMMIT --------------------------------------------
[ -d "$WORK/atombench/.git" ] || git clone -q "$ATOMBENCH_URL" "$WORK/atombench"
TREE="$WORK/atombench-$ATOMBENCH_COMMIT"
[ -d "$TREE" ] || git -C "$WORK/atombench" worktree add -q --detach "$TREE" "$ATOMBENCH_COMMIT"

# ---- jarvis-tools, unpacked here ----------------------------------------------
# Only the jarvis package goes on PYTHONPATH, so dft_3d is the snapshot that
# version fetches.  Older versions download it into the package directory, i.e.
# here; newer ones into $ATOMGPTLAB_CACHE, which is pointed here too.  Neither
# touches $PY's env or ~/.cache.
jt_dir() { echo "$WORK/jarvis-tools-$1"; }
ensure_jt() {
    local d; d="$(jt_dir "$1")"
    [ -d "$d/jarvis" ] && return 0
    mkdir -p "$d"
    "$PY" -m pip download -q --no-deps --only-binary :all: "jarvis-tools==$1" -d "$d"
    unzip -q -o "$d"/jarvis_tools-"$1"-*.whl 'jarvis/*' -d "$d"
}

# prep <log> <dir> <jarvis-tools version> <args...>: one AtomBench preprocessing run
prep() {
    local logf="$1" dir="$2" jt="$3"; shift 3
    (cd "$dir" && env PYTHONPATH="$(jt_dir "$jt")" ATOMGPTLAB_CACHE="$WORK/atomgptlab-cache" \
        "$PY" "$@" > "$logf" 2>&1) || { tail -20 "$logf" >&2; return 1; }
}

# ---- JARVIS Supercon-3D: tc_supercon/scripts/run_{atomgpt,cdvae,flowmm}_data.sh
jarvis_art() { echo "$WORK/splits/jarvis-$ATOMBENCH_COMMIT-jt$1"; }
build_jarvis() {
    local art; art="$(jarvis_art "$1")"
    [ -d "$art" ] && return 0
    ensure_jt "$1"
    rm -rf "$art.partial"
    for m in atomgpt cdvae flowmm; do
        log "AtomBench $ATOMBENCH_COMMIT, jarvis-tools $1: data_preprocess.py $m"
        prep "$art.$m.log" "$TREE/tc_supercon" "$1" scripts/data_preprocess.py "$m" \
            --dataset dft_3d --output "$art.partial/$m" --target Tc_supercon --seed 123 --max-size 1058
    done
    mv "$art.partial" "$art"  # only a complete build is ever reused
}

# ---- Alexandria DS-A/B: alexandria/scripts/get_csv.sh, then run_*_data.sh ----
alex_art() { echo "$WORK/splits/alex-$ATOMBENCH_COMMIT-jt$1"; }
build_alex() {
    local art src; art="$(alex_art "$1")"
    [ -d "$art" ] && return 0
    ensure_jt "$1"
    rm -rf "$art.partial"
    src="$TREE/alexandria"
    if [ ! -f "$src/dataset2.csv" ]; then
        log "Alexandria: get_csv.sh"
        wget -q "$ALEX_RAW_URL/DS-A.pk.bz2" -O "$src/DS-A.pk.bz2"
        wget -q "$ALEX_RAW_URL/DS-B.pk.bz2" -O "$src/DS-B.pk.bz2"
        # get_csv.sh's pd.read_pickle().to_csv(), plus the shim modern pandas
        # needs for pickles written by pandas 1.x
        (cd "$src" && "$PY" -c '
import sys, types
import pandas as pd
shim = types.ModuleType("pandas.core.indexes.numeric")
for name in ("Int64Index", "Float64Index", "UInt64Index", "NumericIndex"):
    setattr(shim, name, pd.Index)
sys.modules[shim.__name__] = shim
pd.read_pickle("DS-A.pk.bz2", compression="bz2").to_csv("dataset1.csv")
pd.read_pickle("DS-B.pk.bz2", compression="bz2").to_csv("dataset2.csv")
')
    fi
    for m in atomgpt cdvae flowmm; do
        log "AtomBench $ATOMBENCH_COMMIT, jarvis-tools $1: alexandria_preprocess.py $m"
        prep "$art.$m.log" "$src" "$1" scripts/alexandria_preprocess.py "$m" \
            --csv-files dataset1.csv dataset2.csv --output "$art.partial/$m" --seed 123 --max-size 8253
    done
    mv "$art.partial" "$art"  # only a complete build is ever reused
}

# Every benchmark CSV ALIGNN-CSP was scored from, in every harness of this repo.
bench_args() {
    local f
    for f in "$REPO"/*/results/"$1"/*/10_runs/*/seed*/bench_*.csv; do
        [ -f "$f" ] && printf -- '--bench-csv\n%s\n' "$f"
    done
    return 0
}
mapfile -t BENCH_JARVIS < <(bench_args jarvis)
mapfile -t BENCH_ALEX < <(bench_args alex)

# ---- Build --------------------------------------------------------------------
for v in $JARVIS_TOOLS; do
    build_jarvis "$v"
done
build_alex "$ALEX_JT"

# ---- Check --------------------------------------------------------------------
{
    echo "# tools/check_atombench_splits.sh, $(date -Iseconds) on $(hostname)"
    echo "# alignn-csp-ablation $(git -C "$REPO" rev-parse --short HEAD); atombench $(git -C "$TREE" rev-parse HEAD)"
    echo "# jarvis-tools: $JARVIS_TOOLS (Alexandria: $ALEX_JT)"
} > "$OUT/summary.txt"
status=0
check() {  # check <label> <checker args...>
    local label="$1"; shift
    log "check: $label"
    if "$PY" "$REPO/tools/check_atombench_splits.py" --label "$label" --strict-order \
            --report "$OUT/$label.json" "$@" 2>&1 | sed "s#$WORK/#<WORK>/#g; s#$REPO/#<REPO>/#g" \
            | tee "$OUT/$label.log"; then
        echo "PASS  $label" >> "$OUT/summary.txt"
    else
        echo "FAIL  $label" >> "$OUT/summary.txt"
        status=1
    fi
    [ -f "$OUT/$label.json" ] && sed -i "s#$WORK/#<WORK>/#g; s#$REPO/#<REPO>/#g" "$OUT/$label.json"
    return 0
}
artifact_args() {  # artifact_args <dir>: AtomBench's AtomGPT, CDVAE and FlowMM splits
    printf -- '--atomgpt-dir\n%s\n--cdvae-dir\n%s\n--flowmm-dir\n%s\n' "$1/atomgpt" "$1/cdvae" "$1/flowmm"
}

for v in $JARVIS_TOOLS; do
    mapfile -t ART < <(artifact_args "$(jarvis_art "$v")")
    check "jarvis__atombench-${ATOMBENCH_COMMIT}__jarvis-tools-${v}" \
        --alignn-csp-dir "$JARVIS_SPLIT" "${ART[@]}" "${BENCH_JARVIS[@]}"
done

mapfile -t ART < <(artifact_args "$(alex_art "$ALEX_JT")")
check "alex__atombench-${ATOMBENCH_COMMIT}" \
    --alignn-csp-dir "$ALEX_SPLIT" "${ART[@]}" "${BENCH_ALEX[@]}"

log "summary"
cat "$OUT/summary.txt"
exit "$status"
