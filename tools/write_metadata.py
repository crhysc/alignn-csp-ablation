#!/usr/bin/env python3
"""Write one exhaustive metadata file per ablation run, from the ground truth.

    /data/ccamp104/envs/repo-tools-x86/bin/python tools/write_metadata.py [--no-hf-staging]

Reads every collected run under

    <harness>/results/<dataset>/<run_id>/10_runs/<arm>/seed<N>/        (this cluster)
    <harness>/results/<dataset>/<run_id>/20_benchmarks/<arm>_seed<N>/  (imported from another cluster)

and writes, for each, a git-tracked

    <harness>/results/<dataset>/<run_id>/ablations/<arm>_seed<N>.yaml

that says what the run is, how it differs from every other run in its set,
every hyperparameter it was trained with, how it was selected, what it scored,
what it cost, which SLURM jobs produced it on which hardware with which version
of the sampler, and the sha256 of every artifact -- so that a reader with no
access to the DVC-tracked tiers still knows exactly what exists.  It also
writes an INDEX.md per results directory and, unless told not to, stages the
browsable Hugging Face layout under hf/ (hardlinks; nothing is duplicated).

Everything here is derived: config.json, history.json, metrics_*.json, stage
markers, the job logs, sacct (when available) and alignn.inverse.ablations.
Nothing is typed in by hand except the small CELL_NOTES table at the bottom,
which records facts the files cannot know (what a cell is *for*, and what went
wrong on the way).
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "alignn"))
from alignn.inverse import ablations as ABL  # noqa: E402  (pure dict module)

SCHEMA = "alignn-csp-ablation/ablation-metadata/v1"

HARNESSES = {
    "supercon-alex-bond-angle-diffusion-line-graph-ablations-sep-2-2026": {
        "set_id": "lgmatrix",
        "title": "line graph x bond-angle diffusion, crossed (3x2 after the state matrix)",
        "tasks": {"jarvis": ("lg-angle-matrix", "lg-angle-state-matrix"),
                  "alex": ("lg-angle-matrix-alex", "lg-angle-state-matrix-alex")},
    },
    "supercon-alex-bond-angle-diffusion-ablation-sep-2-2026": {
        "set_id": "angle-ablation",
        "title": "the A-suite angular-diffusion ablations (A0-A6, nolg, A3_nogate)",
        "tasks": {"jarvis": ("angle-ablation", "ablation-linegraph"),
                  "alex": ("angle-ablation-alex", "ablation-linegraph-alex")},
    },
}

DATASETS = {
    "jarvis": {
        "name": "JARVIS-DFT dft_3d, superconducting-Tc subset (Supercon-3D)",
        "repo_dir": "datasets/jarvis_supercon3d",
        "n_test": 103,
    },
    "alex": {
        "name": "Alexandria DS-A/DS-B (AtomBench split)",
        "repo_dir": "datasets/alexandria_dsab",
        "n_test": 825,
    },
}

# Which cell label / tier each config dir realises.  Built from the tables in
# alignn.inverse.ablations so a rename there is caught here.
def cell_table() -> dict:
    out = {}
    for mname in ("parameter", "state"):
        for label, c in ABL.matrix_cells(mname).items():
            out.setdefault(c["config"], {})
            out[c["config"]].update({"label": label, "matrix": mname,
                                     "ablation": c["ablation"],
                                     "alignn_layers": c["alignn_layers"],
                                     "gcn_layers": c["gcn_layers"]})
    return out

CELLS = cell_table()

TIER = {"off": "none", "derived_aux": "derived", "independent": "independent",
        None: "none"}

# The sampler gained the per-crystal divergence quarantine on 2026-09-04 at
# ~13:30 EDT (alignn commit f8121f4 records it).  Jobs whose *generate* stage
# ran before that used the earlier sampler; on a healthy trajectory the two
# are bit-for-bit identical (proven on a 40-structure batch), so this only
# matters for whether a diverged candidate would have aborted the batch.
QUARANTINE_CUTOFF = dt.datetime.fromisoformat("2026-09-04T13:30:00-04:00")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(p: Path):
    try:
        return json.loads(p.read_text().replace("NaN", "null").replace("Infinity", "null"))
    except Exception:
        return None


def sacct(jobid: str) -> dict | None:
    try:
        out = subprocess.run(
            ["sacct", "-j", jobid, "-X", "-n", "-P",
             "-o", "JobID,JobName,State,ExitCode,Partition,NodeList,Start,End,Elapsed,ReqTRES"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception:
        return None
    for line in out:
        f = line.split("|")
        if f and f[0] == jobid:
            return dict(zip(["jobid", "name", "state", "exit", "partition", "nodes",
                             "start", "end", "elapsed", "req_tres"], f))
    return None


def jobs_for_unit(prov: Path, unit: str) -> list[dict]:
    """Every job log in 00_provenance/logs whose first '=== unit' line is this unit."""
    jobs = []
    for log in sorted((prov / "logs").glob("*.out")) if (prov / "logs").is_dir() else []:
        head = log.read_text(errors="ignore")[:6000]
        m = re.search(r"=== unit ([A-Za-z0-9_]+-seed\d+)", head)
        if not m or m.group(1) != unit:
            continue
        jid = re.search(r"-(\d+)\.out$", log.name).group(1)
        text = log.read_text(errors="ignore")
        rec = {"jobid": int(jid), "log": str(log.relative_to(ROOT)),
               "host": (re.search(r"^host:\s+(\S+)", text, re.M) or [None, None])[1],
               "gpu": (re.search(r"gpu_sampler: device '([^']+)'", text) or [None, None])[1],
               "stages_run": re.findall(r"^\[[^\]]+/([a-z-]+)\] \$", text, re.M),
               "failed_stage": (re.search(r"^\[[^\]]+/([a-z-]+)\] FAILED", text, re.M) or [None, None])[1],
               "sacct": sacct(jid)}
        dm = re.search(r"dropped (\d+) of (\d+) candidates", text)
        rec["dropped_diverged"] = {"n": int(dm.group(1)), "of": int(dm.group(2))} if dm else None
        jobs.append(rec)
    return jobs


def flat_metrics(m: dict | None) -> dict | None:
    if not m:
        return None
    R = m.get("RMSE", {}).get("AtomGen", {}); A = m.get("MAE", {}).get("average_mae", {}); K = m.get("KLD", {})
    def mean(d, ks):
        v = [d[k] for k in ks if d.get(k) is not None]
        return round(sum(v) / len(v), 6) if v else None
    return {
        "match_rate": R.get("match_rate"), "n_matched": R.get("n_matched"), "n_total": R.get("n_total"),
        "rmsd_cartesian_A": R.get("mean_cartesian_rms_angstrom"),
        "rmsd_normalized": R.get("mean_normalized_cartesian_rms"),
        "ccrmse": (m.get("ccRMSE") or {}).get("value"),
        "mae_lattice_lengths_A": mean(A, "abc"), "mae_lattice_angles_deg": mean(A, ("alpha", "beta", "gamma")),
        "kld_mean": mean(K, ("a", "b", "c", "alpha", "beta", "gamma")),
        "full": m,
    }


def history_summary(h: list | None) -> dict | None:
    if not h:
        return None
    key = lambda e: e["val"].get("loss_structural", e["val"].get("loss", 9e9))
    best = min(h, key=key)
    tot = min(h, key=lambda e: e["val"].get("loss", 9e9))
    def pick(e): return {k: e["val"].get(k) for k in ("loss", "loss_structural", "loss_lattice", "loss_frac", "loss_angle")}
    return {"epochs_recorded": len(h),
            "selected_epoch_argmin_structural": best.get("epoch"),
            "val_at_selected": pick(best),
            "train_at_selected": {k: v for k, v in best.get("train", {}).items() if isinstance(v, (int, float))},
            "argmin_total_epoch": tot.get("epoch"),
            "val_at_argmin_total": pick(tot),
            "val_final": pick(h[-1]),
            "note": "best_model.pt is the argmin of val loss_structural (select_on=structural); "
                    "the two argmins differ in the independent-mode cells."}


def differentiators(cfg: dict) -> dict:
    """The switches that distinguish this run from its neighbours in the set."""
    return {
        "angular_tier": TIER.get(cfg.get("angle_mode")),
        "angle_mode": cfg.get("angle_mode"),
        "ablation_key": cfg.get("ablation"),
        "line_graph": bool(cfg.get("alignn_layers", 0) > 0),
        "alignn_layers": cfg.get("alignn_layers"), "gcn_layers": cfg.get("gcn_layers"),
        "topology": cfg.get("topology"), "gate_pair_messages": cfg.get("gate_pair_messages"),
        "angle_feedback": cfg.get("angle_feedback"), "angle_basis": cfg.get("angle_basis"),
        "angle_state_basis": cfg.get("angle_state_basis"), "angle_state_harmonics": cfg.get("angle_state_harmonics"),
        "angle_loss_weighting": cfg.get("angle_loss_weighting"), "angle_weight": cfg.get("angle_weight"),
        "seed": cfg.get("seed"), "epochs": cfg.get("epochs"), "n_parameters": cfg.get("n_parameters"),
    }


def describe(cfg_dir: str, ablation: str) -> str:
    c = CELLS.get(cfg_dir)
    if c:
        table = ABL.MATRIX_DESCRIPTIONS if c["matrix"] == "parameter" else ABL.MATRIX_STATE_DESCRIPTIONS
        return f"{c['label']}: {table.get(c['label'], '')}"
    try:
        return ABL.describe(ablation)
    except Exception:
        return ""


def write_run_yaml(rec: dict, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(rec, sort_keys=False, width=100, allow_unicode=True))


def collected_run(harness: Path, ds: str, run_id: str, rundir: Path, prov: Path, manifest: dict | None) -> dict:
    arm, seed = rundir.parent.name, int(rundir.name.replace("seed", ""))
    cfg = jload(rundir / "config.json") or {}
    cfg_dir = f"{ds}_{arm}"
    hist = jload(rundir / "history.json")
    stages = {}
    for s in sorted((rundir / "stages").glob("*.json")) if (rundir / "stages").is_dir() else []:
        d = jload(s) or {}
        stages[s.stem] = {"elapsed_s": d.get("elapsed_s"), "host": d.get("host")}
    unit = f"{cfg_dir}-seed{seed}"
    jobs = jobs_for_unit(prov, unit)
    # sampler provenance from the generate job's start time
    sampler = "unknown"
    gen_jobs = [j for j in jobs if "generate" in (j["stages_run"] or []) and j["failed_stage"] != "generate"]
    if gen_jobs:
        j = gen_jobs[-1]
        if j["dropped_diverged"] is not None:
            sampler = "quarantine (post 2026-09-04 fix; drop count logged)"
        else:
            # the generate stage starts (job end) - (generate + the stages after it)
            end = (j["sacct"] or {}).get("end")
            post = sum(((stages.get(s) or {}).get("elapsed_s") or 0)
                       for s in ("generate", "symmetrize", "score-nosym", "score-sym"))
            if end and end != "Unknown":
                t_end = dt.datetime.fromisoformat(end).replace(tzinfo=dt.timezone(dt.timedelta(hours=-4)))
                t_gen = t_end - dt.timedelta(seconds=post)
                sampler = ("quarantine (post 2026-09-04 fix)" if t_gen >= QUARANTINE_CUTOFF
                           else "pre-quarantine")
    files = []
    for p in sorted(rundir.iterdir()):
        if p.is_file():
            files.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    missing = [n for n in ("bench_raw.csv", "metrics_raw.json", "bench_rawsym.csv", "metrics_rawsym.json",
                           "angle_eval.json", "angle_eval_rawsym.json") if not (rundir / n).exists()]
    hset = HARNESSES[harness.name]
    task = hset["tasks"][ds][1 if cfg.get("angle_mode") == "independent" else 0]
    notes = CELL_NOTES.get((hset["set_id"], ds, arm), [])
    m_sym, m_nosym = jload(rundir / "metrics_sym.json"), jload(rundir / "metrics_nosym.json")
    gpu_h = round(sum((v["elapsed_s"] or 0) for v in stages.values()) / 3600, 4) if stages else None
    return {
        "schema": SCHEMA,
        "identity": {
            "experiment_set": hset["set_id"], "experiment_set_title": hset["title"],
            "harness_dir": harness.name, "dataset_key": ds, "dataset": DATASETS[ds]["name"],
            "run_id": run_id, "arm": arm, "config_dir": cfg_dir, "seed": seed, "unit": unit,
            "task": task, "cell_label": CELLS.get(cfg_dir, {}).get("label"),
            "matrix": CELLS.get(cfg_dir, {}).get("matrix"),
            "status": "complete" if not missing or all("raw" in n or "angle_eval" in n for n in missing) else "partial",
        },
        "description": describe(cfg_dir, cfg.get("ablation", "")),
        "differentiators": differentiators(cfg),
        "what_this_cell_tests": notes,
        "model_config_verbatim": {k: v for k, v in cfg.items() if k not in ("output", "data_dir", "device")},
        "training": {
            "optimizer": "AdamW", "lr_peak": cfg.get("lr"), "schedule": "OneCycleLR pct_start=0.05",
            "weight_decay": cfg.get("weight_decay"), "grad_clip": cfg.get("grad_clip"),
            "batch_size": cfg.get("batch_size"), "epochs": cfg.get("epochs"), "ema_decay": cfg.get("ema_decay"),
            "loss_weights": {"lattice": cfg.get("lattice_weight"), "frac": cfg.get("frac_weight"), "angle": cfg.get("angle_weight")},
            "checkpoint_selection": cfg.get("select_on"), "augment": cfg.get("augment"),
            "history": history_summary(hist),
        },
        "data": {"repo_dir": DATASETS[ds]["repo_dir"], "split_meta": jload(ROOT / DATASETS[ds]["repo_dir"] / "split_meta.json"),
                 "canonical_basis": "primitive + Niggli (AtomBench scoring basis); augmentation off because of it"},
        "generation": {**{k: v for k, v in (jload(rundir / "generation_config.json") or {}).items()
                          if k in ("num_candidates", "guidance", "n_corrector", "step_lr", "relax", "relax_steps",
                                   "relax_fmax", "rank", "prescreen_keep", "max_batch_nodes", "seed", "use_ema")},
                       "symprec": 0.1, "symprec_chosen_on_validation": False,
                       "sampler_version": sampler,
                       "candidates_dropped_as_diverged": next((j["dropped_diverged"] for j in jobs if j["dropped_diverged"]), None)},
        "metrics": {"variant_sym": flat_metrics(m_sym), "variant_nosym": flat_metrics(m_nosym),
                    "variant_raw": None, "variant_rawsym": None,
                    "note": "sym/nosym are AFTER 32-candidate ALIGNN-FF prescreen+relaxation; raw/rawsym (the "
                            "generator alone) are not yet produced -- see NEXT_TASK.md"},
        "compute": {"stages": stages, "gpu_hours_billed": gpu_h,
                    "hardware": {"gpu": next((j["gpu"] for j in jobs if j["gpu"]), None),
                                 "hosts": sorted({j["host"] for j in jobs if j["host"]}),
                                 "site": "atomgptlab (JHU WSE), 2x NVIDIA GB10, aarch64 GPU nodes"},
                    "slurm_jobs": jobs},
        "code": {"alignn_repo": "https://github.com/crhysc/alignn", "alignn_branch": "lg-angle-diffusion-matrix",
                 "alignn_commit_archived": git_rev(ROOT / "alignn"),
                 "note": "jobs executed the live working tree; at run time it was fb151f6 plus the uncommitted "
                         "changes later committed as f8121f4 (independent angular state, MATRIX_STATE, sampler quarantine)."},
        "artifacts": {"dvc_tracked_dir": str((harness / "results" / ds / run_id / "10_runs").relative_to(ROOT)),
                      "files": files, "missing": missing,
                      "manifest_sha256_agrees": manifest_agrees(manifest, arm, seed, files)},
    }


def manifest_agrees(manifest, arm, seed, files):
    if not manifest:
        return None
    for r in manifest.get("runs", []):
        if r.get("arm") == arm and r.get("seed") == seed:
            mine = {f["name"]: f["sha256"] for f in files}
            return all(mine.get(n) == v.get("sha256") for n, v in r.get("files", {}).items())
    return None


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def imported_run(harness: Path, ds: str, run_id: str, bdir: Path, stats: dict, costs: dict) -> dict:
    """A run whose checkpoints are not on this machine: metadata from the benchmark tier."""
    m = re.match(r"(.+)_seed(\d+)$", bdir.name); arm, seed = m.group(1), int(m.group(2))
    ablation = "A3" if arm == "A3_nogate" else ("A0" if arm == "nolg" else arm)
    try:
        cfg = ABL.ablation_config(ablation)
    except Exception:
        cfg = {}
    if arm == "A3_nogate":
        cfg = {**cfg, "gate_pair_messages": False}
    layers = (0, 9) if arm == "nolg" else (3, 3)
    per_run = (costs.get("per_run") or {}).get(bdir.name, {})
    arm_stats = (stats.get("arms") or {}).get(arm, {})
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(bdir.iterdir()) if p.is_file()]
    raw = harness / "results" / ds / run_id / "20_benchmarks_rawsym" / bdir.name
    hset = HARNESSES[harness.name]
    return {
        "schema": SCHEMA,
        "identity": {"experiment_set": hset["set_id"], "experiment_set_title": hset["title"],
                     "harness_dir": harness.name, "dataset_key": ds, "dataset": DATASETS[ds]["name"],
                     "run_id": run_id, "arm": arm, "config_dir": f"{ds}_{arm}", "seed": seed,
                     "task": hset["tasks"][ds][1 if arm == "nolg" else 0], "status": "complete (benchmarks only)"},
        "description": ABL.describe(ablation) if arm != "A3_nogate" else
                       "A3 with the pair-message gate switched off (confound control: is the A3 gain from the gate?)",
        "differentiators": {"angular_tier": "derived" if cfg.get("angle_diffusion") else "none",
                            "angle_mode": "derived_aux" if cfg.get("angle_diffusion") else "off",
                            "ablation_key": ablation, "line_graph": layers[0] > 0,
                            "alignn_layers": layers[0], "gcn_layers": layers[1],
                            "topology": cfg.get("topology"), "gate_pair_messages": cfg.get("gate_pair_messages"),
                            "angle_feedback": cfg.get("angle_feedback"), "angle_basis": cfg.get("angle_basis"),
                            "seed": seed, "epochs": per_run.get("num_epochs"), "n_parameters": per_run.get("n_parameters")},
        "what_this_cell_tests": CELL_NOTES.get((hset["set_id"], ds, arm), []),
        "training": {"epochs": per_run.get("num_epochs"), "batch_size": per_run.get("batch_size"),
                     "checkpoint_selection": "validation total loss (this suite predates --select-on structural)",
                     "note": "config.json/history.json for this run are not archived here; see artifacts.checkpoints"},
        "data": {"repo_dir": DATASETS[ds]["repo_dir"], "split_meta": jload(ROOT / DATASETS[ds]["repo_dir"] / "split_meta.json")},
        "generation": {"num_candidates": 32, "guidance": 2.0, "relax": "cell", "rank": "energy", "prescreen_keep": 4,
                       "symprec": 0.1, "sampler_version": "pre-quarantine (2026-08-30)"},
        "metrics": {"variant_sym": flat_metrics(jload(bdir / "metrics.json")),
                    "variant_rawsym": flat_metrics(jload(raw / "metrics.json")) if raw.is_dir() else None,
                    "arm_over_seeds": arm_stats},
        "compute": {"stages_s": per_run.get("stage_s"), "hosts": per_run.get("hosts"),
                    "gpu_hours_billed": per_run.get("total_h"), "train_s_per_step": per_run.get("train_s_per_step"),
                    "hardware": {"gpu": "NVIDIA A30 (pinned)", "site": "WVU Dolly Sods, partition gpu_7day"}},
        "code": {"alignn_repo": "https://github.com/crhysc/alignn", "alignn_branch": "angle-diffusion (at the time)",
                 "note": "INDEX.md records the collecting host had no git; treat the commit as 'angle-diffusion as of 2026-08-30'."},
        "artifacts": {"benchmark_dir": str(bdir.relative_to(ROOT)), "files": files,
                      "checkpoints": "NOT archived in this repository: they remain in /scratch/crc00042/alignn_csp on "
                                     "WVU Dolly Sods (see results/jarvis/2026-08-30_full/INDEX.md)."},
    }


def stage_hf(run_rec: dict, rundir: Path):
    """Hardlink the browsable Hugging Face layout: hf/model/<set>/<ds>/<arm>_seed<N>/."""
    idn = run_rec["identity"]
    dest = ROOT / "hf" / "model" / idn["experiment_set"] / idn["dataset_key"] / f"{idn['arm']}_seed{idn['seed']}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("best_model.pt", "config.json", "history.json", "metrics_sym.json",
                 "metrics_nosym.json", "generation_config.json"):
        src = rundir / name
        if src.exists():
            tgt = dest / name
            if tgt.exists():
                tgt.unlink()
            try:
                os.link(src, tgt)
            except OSError:
                shutil.copy2(src, tgt)
    (dest / "ABLATION.yaml").write_text(yaml.safe_dump(run_rec, sort_keys=False, width=100))


def write_index(resdir: Path, recs: list[dict]):
    L = [f"# Ablations in `{resdir.relative_to(ROOT)}`", "",
         "One YAML per run in `ablations/`, generated by `tools/write_metadata.py` from the run's own files.",
         "Bulk artifacts (weights, predictions, histories) are DVC-tracked in `10_runs/`; `dvc pull` restores them.", "",
         "| arm | seed | tier | line graph | angle_mode | params | best ep | L_struct | match (sym) | RMSD (sym) | ccRMSE (sym) | status |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        d, i = r["differentiators"], r["identity"]
        h = (r.get("training") or {}).get("history") or {}
        ms = (r["metrics"].get("variant_sym") or {})
        L.append(f"| {i['arm']} | {i['seed']} | {d.get('angular_tier')} | {'yes' if d.get('line_graph') else 'no'} | "
                 f"{d.get('angle_mode')} | {d.get('n_parameters')} | {h.get('selected_epoch_argmin_structural','-')} | "
                 f"{(h.get('val_at_selected') or {}).get('loss_structural','-')} | {ms.get('match_rate','-')} | "
                 f"{ms.get('rmsd_cartesian_A','-')} | {ms.get('ccrmse','-')} | {i.get('status')} |")
    (resdir / "ablations" / "INDEX.md").write_text("\n".join(L) + "\n")


# Facts the files cannot know.  Keyed by (experiment set, dataset, arm).
_LG = "the line graph (3 ALIGNN layers, 3 pair convs)"
_NOLG = "no line graph (9 pair convs, parameter-matched)"
CELL_NOTES = {}
for ds in ("jarvis", "alex"):
    CELL_NOTES.update({
        ("lgmatrix", ds, "nolg"): [f"Baseline of both 2x2s: no angular channel, {_NOLG}, hard kNN topology.",
                                   "Shared by config name between MATRIX and MATRIX_STATE; trained once."],
        ("lgmatrix", ds, "A0"): [f"No angular channel, {_LG}: angles enter only as ALIGNN input features (manuscript Table 3 arm A).",
                                 "Shared between MATRIX and MATRIX_STATE; trained once."],
        ("lgmatrix", ds, "nolg_ad"): [f"Legacy derived-aux angular supervision (A3) with {_NOLG}: triplets built and supervised, no triplet message reaches the trunk.",
                                      "Angular target = wrap(theta_t - theta_0) induced by the (F,L) corruption; no forward kernel, no state."],
        ("lgmatrix", ds, "A3"): [f"Legacy derived-aux angular supervision (A3) with {_LG}: the manuscript's 'proposed' arm before the state was made independent.",
                                 "Carries radius topology + gated pair messages, as every angular cell does (continuity rides with the angular factor)."],
        ("lgmatrix", ds, "nolg_b3"): [f"Independent bond-angle diffusion (B3, angle_mode=independent) with {_NOLG}: persistent dense triplet index, own wrapped-normal VE kernel, jointly denoised, discarded at readout.",
                                      "Angular state reaches coordinates only through the shared trunk's gradient (z = emb(phi) + y_ij + y_jk after the pair convs)."],
        ("lgmatrix", ds, "B3"): [f"Independent bond-angle diffusion (B3) with {_LG}: the proposed model; angular state couples forward through angles -> bonds -> atoms.",
                                 "This is the only cell whose first generation attempt aborted (lattice divergence in one crystal); regenerated with the quarantine sampler."],
    })
CELL_NOTES[("lgmatrix", "jarvis", "B3")].append(
    "Job 12969 trained to completion then FAILED in generate chunk 1/4 (FloatingPointError: non-finite lattice); job 13082 re-ran generate+score only.")
CELL_NOTES[("lgmatrix", "alex", "A3")].append(
    "Originally queued as job 12965; cancelled before start to give its slot to alex_B3, then re-queued as job 13084 on the user's request.")
CELL_NOTES[("lgmatrix", "alex", "B3")].append(
    "Generation dropped 5 of 26400 candidates whose trajectory diverged (the quarantine's first production use).")
for arm, note in {"A0": "baseline: angles as features only, kNN topology",
                  "A1": "explicit (derived) angular denoising on the hard kNN line graph -- the arm the A5 comparison uses",
                  "A2": "smooth radius topology + gate, NO angular objective -- the topology-alone control",
                  "A3": "derived angular denoising + smooth topology -- the A-suite's proposed arm (= lgmatrix 'A3' = B1)",
                  "A3_nogate": "A3 with gate_pair_messages=False -- confound control for the pair-message gate",
                  "A4": "A3 with angle_feedback=False -- coupling control: angular loss as pure auxiliary supervision",
                  "A6": "A3 with the learnable Fourier angle basis instead of the cosine RBF -- representation ablation",
                  "nolg": "no line graph, 9 pair convs -- the manuscript Table 3 arm B baseline"}.items():
    CELL_NOTES[("angle-ablation", "jarvis", arm)] = [note, "3 seeds (0,1,2), 3000 epochs, A30 pinned, WVU Dolly Sods, 2026-08-30/31."]
    CELL_NOTES[("angle-ablation", "alex", arm)] = [note, "Alexandria port CANCELLED 2026-09-02 mid-training (A0 at epoch 375, A1 at 350; other arms never started)."]


def main() -> int:
    stage = "--no-hf-staging" not in sys.argv
    if stage:
        shutil.rmtree(ROOT / "hf" / "model", ignore_errors=True)
    total = 0
    for hname in HARNESSES:
        harness = ROOT / hname
        for ds_dir in sorted((harness / "results").glob("*")):
            ds = ds_dir.name
            for resdir in sorted(ds_dir.glob("*")):
                run_id = resdir.name; recs = []
                prov = resdir / "00_provenance"
                manifest = jload(prov / "manifest.json")
                for rundir in sorted((resdir / "10_runs").glob("*/seed*")) if (resdir / "10_runs").is_dir() else []:
                    if not (rundir / "config.json").exists():
                        continue
                    rec = collected_run(harness, ds, run_id, rundir, prov, manifest)
                    write_run_yaml(rec, resdir / "ablations" / f"{rundir.parent.name}_seed{rundir.name[4:]}.yaml")
                    if stage:
                        stage_hf(rec, rundir)
                    recs.append(rec); total += 1
                if not recs and (resdir / "20_benchmarks").is_dir():
                    stats = jload(resdir / "40_stats" / "stats.json") or {}
                    costs = jload(resdir / "50_costs" / "computational_costs.json") or {}
                    for bdir in sorted((resdir / "20_benchmarks").glob("*_seed*")):
                        rec = imported_run(harness, ds, run_id, bdir, stats, costs)
                        write_run_yaml(rec, resdir / "ablations" / f"{bdir.name}.yaml")
                        recs.append(rec); total += 1
                if recs:
                    write_index(resdir, recs)
                    print(f"{resdir.relative_to(ROOT)}: {len(recs)} ablation record(s)")
    if stage:
        d = ROOT / "hf" / "datasets"; shutil.rmtree(d, ignore_errors=True)
        for src in (ROOT / "datasets").rglob("*"):
            if src.is_file():
                tgt = d / src.relative_to(ROOT / "datasets"); tgt.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(src, tgt)
                except OSError:
                    shutil.copy2(src, tgt)
        for card, tgt in (("model-README.md", ROOT / "hf" / "model" / "README.md"),
                          ("datasets-README.md", ROOT / "hf" / "datasets" / "README.md")):
            c = ROOT / "hf" / "cards" / card
            if c.exists():
                shutil.copy2(c, tgt)
    print(f"wrote {total} ablation metadata files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
