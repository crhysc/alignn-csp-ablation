#!/usr/bin/env bash
# Price the run before committing to it.
#
#   bash 20_pilot.sh train    # train ONE cell at full settings -> s/epoch
#   bash 20_pilot.sh price    # time generation on a few targets -> extrapolate
#   bash 20_pilot.sh report   # read both back, print the PHASE_TIME to set
#
# Why this exists.  Training on this benchmark is measured and cheap: 4.26-4.38
# s/epoch on GB10 for the Alexandria split, about 1.2 h for 1000 epochs.
# GENERATION is neither measured nor cheap, and it is what actually decides the
# schedule: N_TEST targets x 32 candidates x 1000 denoising steps, then an
# ALIGNN-FF single-point prescreen and 4 relaxations per target.  On the
# Alexandria split that is 825 targets against JARVIS's 103.  Nobody has ever
# run it to completion -- the previous attempt was cancelled during training --
# so every walltime in this harness is a placeholder until `price` replaces it.
#
# `price` deliberately changes NOTHING that affects cost per target: full 32
# candidates, full 1000 denoising steps, the real relax settings.  It only
# reduces --limit, so the per-target number extrapolates honestly.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

MODE="${1:-price}"
LIMIT="${LIMIT:-12}"          # targets to time; cost is linear in this
PILOT_CELL="${PILOT_CELL:-both}"   # the most expensive cell: 3 ALIGNN layers
JOBLOG="$RESULTS/00_provenance/slurm_jobids.txt"
mkdir -p "$RESULTS"/{00_provenance,50_costs}

cell_config() {   # matrix label -> config dir name
    python3 - "$1" <<'PY'
import sys
sys.path.insert(0, "../alignn")
try:
    from alignn.inverse.ablations import MATRIX
except Exception:
    MATRIX = {"neither": {"config": "nolg"}, "line graph": {"config": "A0"},
              "angle diffusion": {"config": "nolg_ad"}, "both": {"config": "A3"}}
print(MATRIX[sys.argv[1]]["config"])
PY
}
CFG="$(cell_config "$PILOT_CELL")"
FIRST_SEED="$(cut -d, -f1 <<<"$SEEDS")"
RUN="$CSP_RUNS/train/${SPLIT}_${CFG}/seed$FIRST_SEED"
# Which array index this cell is.  Resolved here, on the login node, and
# interpolated as a literal: working it out inside the sbatch would need a
# nested quoted python -c inside an unquoted heredoc, which is exactly the
# kind of quoting that breaks silently and submits the wrong unit.
UNIT_INDEX="$(python3 - "$PILOT_CELL" <<'IDX'
import sys
sys.path.insert(0, "../alignn")
from alignn.inverse.ablations import MATRIX
print(list(MATRIX).index(sys.argv[1]))
IDX
)"

case "$MODE" in

train)
    echo "=== pilot training: cell '$PILOT_CELL' (${SPLIT}_${CFG}), full settings"
    echo "    -> $RUN"
    SB="$RESULTS/00_provenance/pilot_train.sbatch"
    cat > "$SB" <<TRAINSB
#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-pilot-train
#SBATCH --output=$RESULTS/00_provenance/pilot_train-%j.out
#SBATCH --error=$RESULTS/00_provenance/pilot_train-%j.out
#SBATCH --time=$PILOT_TIME
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
$(sbatch_account_line)
#SBATCH --partition=$PART_PILOT
set -uo pipefail
cd "$HARNESS"; source ./env.sh
# --only-stages train: the whole point is to isolate the training cost from
# the generation cost, which \`price\` measures separately.
run_task $MATRIX_TASK --seeds "$FIRST_SEED" \\
    --only-stages train --unit $UNIT_INDEX
TRAINSB
    if [ "${DRY:-0}" = "1" ]; then echo "would submit: sbatch $SB"; exit 0; fi
    JID=$(sbatch --parsable --export=ALL "$SB")
    printf 'pilot-train\tjob\t%s\n' "$JID" >> "$JOBLOG"
    echo "submitted $JID   watch: squeue -j $JID"
    echo "then:  bash 20_pilot.sh price"
    ;;

price)
    [ -f "$RUN/best_model.pt" ] || {
        echo "no checkpoint at $RUN/best_model.pt" >&2
        echo "run 'bash 20_pilot.sh train' first, or point PILOT_CELL at a cell that has one" >&2
        exit 1; }
    echo "=== pricing generation: $LIMIT of $N_TEST targets, real settings"
    SB="$RESULTS/00_provenance/pilot_price.sbatch"
    cat > "$SB" <<PRICESB
#!/usr/bin/env bash
#SBATCH --job-name=csp-lgm-price
#SBATCH --output=$RESULTS/00_provenance/pilot_price-%j.out
#SBATCH --error=$RESULTS/00_provenance/pilot_price-%j.out
#SBATCH --time=$PILOT_TIME
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=$CPUS_PER_TASK
#SBATCH --mem=$MEM_PER_TASK
#SBATCH --gres=$CSP_GPU_GRES
$(sbatch_account_line)
#SBATCH --partition=$PART_PILOT
set -uo pipefail
cd "$ALIGNN_REPO"
source task_runners/common.sh

OUT="$RESULTS/50_costs"
mkdir -p "\$OUT" "$CSP_RUNS/pilot_price"

# Every knob that scales cost per target is the REAL one -- 32 candidates,
# 1000 denoising steps, cell relaxation, 4 relaxed survivors.  Only the number
# of targets is reduced, so seconds/target extrapolates without a correction.
t0=\$SECONDS
OMP_NUM_THREADS=1 python -u scripts/atombench/generate_benchmark.py \\
    --checkpoint "$RUN/best_model.pt" \\
    --data-dir "$CSP_RUNS/data/$SPLIT" --split test \\
    --output-csv "$CSP_RUNS/pilot_price/pred.csv" \\
    --num-candidates 32 --guidance 2.0 --relax cell --rank energy \\
    --relax-steps 200 --prescreen-keep 4 --relax-workers $RELAX_WORKERS \\
    --limit $LIMIT --seed 0 --device cuda
GEN=\$((SECONDS-t0))

t0=\$SECONDS
bash scripts/atombench/symmetrize.sh \\
    --csv "$CSP_RUNS/pilot_price/pred.csv" \\
    --out "$CSP_RUNS/pilot_price/pred_sym.csv" --symprec $SYMPREC
bash scripts/atombench/score.sh "$CSP_RUNS/pilot_price/pred_sym.csv"
POST=\$((SECONDS-t0))

printf '{"limit":%d,"n_test":%d,"generate_s":%d,"post_s":%d,"host":"%s","cell":"%s"}\\n' \\
    $LIMIT $N_TEST "\$GEN" "\$POST" "\$(hostname)" "$PILOT_CELL" \\
    > "\$OUT/generation_price.json"
echo "wrote \$OUT/generation_price.json"
cat "\$OUT/generation_price.json"
PRICESB
    if [ "${DRY:-0}" = "1" ]; then echo "would submit: sbatch $SB"; exit 0; fi
    JID=$(sbatch --parsable --export=ALL "$SB")
    printf 'pilot-price\tjob\t%s\n' "$JID" >> "$JOBLOG"
    echo "submitted $JID   watch: squeue -j $JID"
    echo "then:  bash 20_pilot.sh report"
    ;;

report)
    P="$RESULTS/50_costs/generation_price.json"
    [ -f "$P" ] || { echo "no $P -- run 'bash 20_pilot.sh price' first" >&2; exit 1; }
    python3 - "$P" "$RUN/history.json" "$TRAIN_EPOCHS" "$CSP_MAX_CONCURRENT" \
        "$SEEDS" <<'REPORT' | tee "$RESULTS/50_costs/schedule.txt"
import json, math, sys
price = json.load(open(sys.argv[1]))
epochs = int(sys.argv[3]); conc = int(sys.argv[4])
nseeds = len(sys.argv[5].split(","))

per_target = (price["generate_s"] + price["post_s"]) / price["limit"]
gen_h = per_target * price["n_test"] / 3600

train_h = None
try:
    hist = json.load(open(sys.argv[2]))
    # history.json carries no timestamps, so training time is taken from the
    # stage marker if there is one; otherwise it is reported as unmeasured
    # rather than guessed.
    import pathlib
    marker = pathlib.Path(sys.argv[2]).with_name(".stages") / "train.json"
    if marker.exists():
        rec = json.load(open(marker))
        ran = hist[-1]["epoch"]
        train_h = rec["elapsed_s"] / ran * epochs / 3600
except Exception:
    pass

print(f"\n  measured on {price['host']}, cell '{price['cell']}'")
print(f"  generation + scoring : {per_target:6.1f} s/target "
      f"({price['limit']} targets timed)")
print(f"  x {price['n_test']} targets        : {gen_h:6.2f} h per arm")
if train_h:
    print(f"  training ({epochs} ep)   : {train_h:6.2f} h per arm")
else:
    print(f"  training ({epochs} ep)   :      ? h  (no stage marker; run "
          f"'20_pilot.sh train' through the task runner)")
per_arm = gen_h + (train_h or 0.0)
n_elem = 4 * nseeds
rounds = math.ceil(n_elem / conc)
print(f"\n  per array element    : {per_arm:6.2f} h")
print(f"  {n_elem} elements / {conc} GPUs  : {rounds} sequential round(s) "
      f"-> {per_arm * rounds:6.2f} h wall clock")
pad = per_arm * 1.5
h = int(pad) + 1
print(f"\n  SET IN env.sh:  PHASE_TIME=\"{h:02d}:00:00\"   (+50% headroom)")
print("  The runner resumes at STAGE granularity: a training killed at 90%")
print("  restarts at 0%.  Over-asking costs nothing on this cluster -- every")
print("  partition is MaxTime=UNLIMITED.")
REPORT
    ;;

*) echo "usage: $0 {train|price|report}" >&2; exit 2 ;;
esac
