#!/usr/bin/env python3
"""Check that ALIGNN-CSP is trained and scored on AtomBench's splits.

    python tools/check_atombench_splits.py \
        --alignn-csp-dir datasets/jarvis_supercon3d \
        --atomgpt-dir <split>/atomgpt --cdvae-dir <split>/cdvae --flowmm-dir <split>/flowmm \
        --bench-csv <harness>/results/jarvis/<run_id>/10_runs/A0/seed0/bench_sym.csv \
        --strict-order --report report.json

<split> is what AtomBench's own preprocessing writes for each model
(tc_supercon/scripts/data_preprocess.py or
alexandria/scripts/alexandria_preprocess.py, run with the arguments in its
run_{atomgpt,cdvae,flowmm}_data.sh).  tools/check_atombench_splits.sh builds
it from a pinned AtomBench commit and runs this end to end.

Adapted from AtomBench's split auditor, tc_supercon/scripts/checker.py at
github.com/atomgptlab/atombench commit 324ed9d.  That script proves AtomGPT,
CDVAE and FlowMM share one split (MatterGen reads CDVAE's CSVs, so it is
covered too): equal test IDs, AtomGPT train == CDVAE/FlowMM train ∪ val,
disjoint duplicate-free splits, and no leakage across splits by CIF-text or
canonical-structure hash.  All of that is kept.  Added:

  * ALIGNN-CSP as a fourth family, read from its prepared train/val/test.json,
    with the hygiene and leakage checks the original applies to CDVAE/FlowMM;
  * ALIGNN-CSP's test, val and train sets must equal each AtomBench model's
    (the test set in the same order with --strict-order);
  * cross-contamination: AtomBench test IDs in ALIGNN-CSP train/val, and
    ALIGNN-CSP test IDs in AtomBench's training data;
  * content of every test target: the same Tc, and the same crystal under
    StructureMatcher, which is cell-choice invariant -- needed because
    ALIGNN-CSP stores primitive+Niggli cells where AtomBench stores the
    database cell;
  * the benchmark CSVs ALIGNN-CSP was scored from must hold exactly its test
    IDs with the same targets (the rule of atombench.verify); and
  * the hash10 split fingerprint in split_meta.json (and so in every
    ablations/*.yaml) must be reproduced by the JSONs and, with
    --strict-order, by AtomGPT's id_prop.csv.

Like the original, it never stops at the first problem: everything is checked,
then it exits 1 if anything failed.  --report writes the full result,
including every differing ID, as JSON.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r"Issues encountered while parsing CIF: .* rounded to ideal values .*",
    category=UserWarning,
)
csv.field_size_limit(2**31 - 1)

SPLITS = ("train", "val", "test")

# --------------------------- misc utilities ---------------------------

def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def _as_list(x) -> List[str]:
    return [str(v).strip() for v in x if str(v).strip() != ""]

def _set(x: Sequence[str]) -> Set[str]:
    return set(_as_list(x))

def _dups(x: Sequence[str]) -> List[str]:
    """Return duplicates (by value) preserving encounter order."""
    seen = set()
    d = []
    for v in _as_list(x):
        if v in seen:
            d.append(v)
        else:
            seen.add(v)
    return d

def _overlap_pairs() -> List[Tuple[str, str]]:
    return [("train", "val"), ("train", "test"), ("val", "test")]

def _short_ids(ids: List[str], k: int = 3) -> str:
    ids = [i for i in ids if i]
    if not ids:
        return "[]"
    shown = ids[:k]
    extra = len(ids) - len(shown)
    if extra > 0:
        return "[" + ", ".join(shown) + f", ... +{extra}" + "]"
    return "[" + ", ".join(shown) + "]"

def hash10(values: Sequence[str]) -> str:
    """AtomBench's split fingerprint (data_preprocess.py), over IDs in order."""
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode())
        h.update(b",")
    return h.hexdigest()[:10]


# --------------------------- Error accumulator ---------------------------

@dataclass
class IssueLog:
    hard_failures: List[str]
    leak_summaries: List[str]
    leak_hashes: List[str]

    def __init__(self):
        self.hard_failures = []
        self.leak_summaries = []
        self.leak_hashes = []

    def add_fail(self, msg: str) -> None:
        self.hard_failures.append(msg)

    def add_leak_summary(self, msg: str) -> None:
        self.leak_summaries.append(msg)

    def add_leak_hash_line(self, msg: str) -> None:
        self.leak_hashes.append(msg)

    def any_fail(self) -> bool:
        return bool(self.hard_failures or self.leak_summaries or self.leak_hashes)

    def report_and_exit(self, report: dict, path: Optional[Path]) -> None:
        if path is not None:
            report = dict(report)
            report["passed"] = not self.any_fail()
            report["hard_failures"] = self.hard_failures
            report["leak_summaries"] = self.leak_summaries
            report["leak_hashes"] = self.leak_hashes
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"report written to {path}")

        if not self.any_fail():
            print("\n✓ All reviewer-facing split/leakage checks passed ✅\n")
            raise SystemExit(0)

        print("\n==================== SPLIT/LEAKAGE AUDIT: FAIL ====================", file=sys.stderr)
        if self.hard_failures:
            print("\n[HARD FAILURES]", file=sys.stderr)
            for msg in self.hard_failures:
                print(f"- {msg}", file=sys.stderr)

        if self.leak_summaries or self.leak_hashes:
            print("\n[LEAKAGE SUMMARY]", file=sys.stderr)
            for msg in self.leak_summaries:
                print(f"- {msg}", file=sys.stderr)

            if self.leak_hashes:
                print("\n[OVERLAPPING HASH TOKENS]", file=sys.stderr)
                for line in self.leak_hashes:
                    print(line, file=sys.stderr)

        print("\n====================================================================\n", file=sys.stderr)
        raise SystemExit(1)


# --------------------------- AtomGPT split parsing ---------------------------

def atomgpt_table(dir_: Path, issues: IssueLog) -> pd.DataFrame:
    """
    id_prop.csv as (path, target, id); headerless: col0=path, col1=target.
    """
    p = dir_ / "id_prop.csv"
    if not p.exists():
        issues.add_fail(f"AtomGPT id_prop.csv not found at: {p}")
        return pd.DataFrame(columns=["path", "target", "id"])

    df = pd.read_csv(p, header=None, names=["path", "target"])
    df["path"] = df["path"].astype(str)
    df["id"] = [Path(s).stem for s in df["path"]]  # strip extension
    return df

def atomgpt_split(ids: List[str], n_test: int, issues: IssueLog) -> Tuple[List[str], List[str]]:
    """
    Split AtomGPT ids into (train_ids, test_ids) as head/tail where tail length = n_test.
    """
    if not ids:
        return [], []

    if n_test <= 0:
        issues.add_fail(f"AtomGPT split: n_test must be > 0, got {n_test}")
        return [], []
    if n_test >= len(ids):
        issues.add_fail(
            f"AtomGPT split: n_test={n_test} must be smaller than total rows={len(ids)} in id_prop.csv"
        )
        return [], []

    train_ids = ids[:-n_test]
    test_ids = ids[-n_test:]
    if len(train_ids) + len(test_ids) != len(ids):
        issues.add_fail("AtomGPT split invariant violated (head+tail != total).")
        return [], []

    return train_ids, test_ids


# --------------------------- CDVAE / FlowMM split parsing ---------------------------

def read_ids_csv(path: Path, issues: IssueLog, col: str = "material_id") -> List[str]:
    if not path.exists():
        issues.add_fail(f"Missing expected split file: {path}")
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if col not in df.columns:
        issues.add_fail(f"Expected column '{col}' in {path}, found {list(df.columns)}")
        return []
    return df[col].astype(str).tolist()

def cdvae_splits(dir_: Path, issues: IssueLog) -> Tuple[List[str], List[str], List[str]]:
    return (
        read_ids_csv(dir_ / "train.csv", issues),
        read_ids_csv(dir_ / "val.csv", issues),
        read_ids_csv(dir_ / "test.csv", issues),
    )

def flowmm_splits(dir_: Path, issues: IssueLog) -> Tuple[List[str], List[str], List[str]]:
    return (
        read_ids_csv(dir_ / "train.csv", issues),
        read_ids_csv(dir_ / "val.csv", issues),
        read_ids_csv(dir_ / "test.csv", issues),
    )


# --------------------------- ALIGNN-CSP split parsing ---------------------------

def alignn_csp_splits(dir_: Path, issues: IssueLog) -> Dict[str, List[dict]]:
    """
    Rows of train/val/test.json as written by alignn/scripts/atombench/prepare_*data.py:
    material_id, target, lattice_mat, frac_coords, elements, target_poscar, ...
    """
    out: Dict[str, List[dict]] = {}
    for sp in SPLITS:
        p = dir_ / f"{sp}.json"
        if not p.exists():
            issues.add_fail(f"Missing expected split file: {p}")
            out[sp] = []
            continue
        out[sp] = json.loads(p.read_text())
    return out

def csp_ids(rows: List[dict]) -> List[str]:
    return [str(r["material_id"]).strip() for r in rows]

def csp_text(row: dict) -> str:
    """Exact serialisation of an ALIGNN-CSP structure (the analogue of CIF text)."""
    return json.dumps([row["lattice_mat"], row["elements"], row["frac_coords"]])

def csp_structure(text: str):
    from pymatgen.core import Structure
    lattice, elements, frac = json.loads(text)
    return Structure(lattice, elements, frac)


# --------------------------- Structures and benchmark CSVs ---------------------------

_CIF_BODY_RE = re.compile(
    r"_cell_length_a|_cell_angle_alpha|_atom_site_|loop_\s*\n\s*_",
    re.IGNORECASE,
)

def parse_cell(text: str):
    """
    Parse a structure the way atombench._structure_io.parse_structure does, for the
    formats AtomBench writes: CIF (CDVAE/FlowMM CSVs), POSCAR (AtomGPT's files) and
    POSCAR with literal \\n escapes (benchmark CSVs).
    """
    from pymatgen.core import Structure
    s = str(text)
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\n", "\n").replace("\\t", " ")
    s = s.strip()
    if s.lower().startswith("data_") or _CIF_BODY_RE.search(s):
        return Structure.from_str(s, fmt="cif")
    return Structure.from_str(s, fmt="poscar")

def _norm_poscar(text: str) -> str:
    s = str(text)
    if "\\n" in s and "\n" not in s:
        s = s.replace("\\n", "\n")
    return "\n".join(" ".join(line.split()) for line in s.splitlines() if line.strip())

def read_benchmark_csv(path: Path, issues: IssueLog) -> List[Tuple[str, str]]:
    """(id, target) rows of a benchmark CSV (id,target,prediction)."""
    if not path.exists():
        issues.add_fail(f"Missing benchmark CSV: {path}")
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"id", "target"} <= set(reader.fieldnames):
            issues.add_fail(f"{path} is not a benchmark CSV (needs id,target,prediction; has {reader.fieldnames})")
            return []
        return [(str(r["id"]).strip(), r["target"]) for r in reader]


# --------------------------- Leakage checks via text/Structure hashing ---------------------------

def _norm_cif_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()
    return s

def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def _import_pymatgen(issues: IssueLog):
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return Structure, SpacegroupAnalyzer
    except Exception as e:
        issues.add_fail(
            "pymatgen is required for structure-level checks. "
            "Install it or run with --no-structure-leakage --no-structure-compare.\n"
            f"Import error: {e}"
        )
        return None, None

def structure_hash(
    s,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    issues: IssueLog,
) -> Optional[str]:
    """
    Niggli reduce -> symmetry standardize -> deterministic site order -> round -> SHA-256.
    Returns None if canonicalization fails.
    """
    Structure, SpacegroupAnalyzer = _import_pymatgen(issues)
    if Structure is None:
        return None

    try:
        # Niggli reduction helps canonicalize lattice representation
        try:
            s = s.get_reduced_structure(reduction_algo="niggli")
        except Exception:
            pass

        # Symmetry-based standardization (spglib-backed)
        try:
            sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance)
            s = sga.get_primitive_standard_structure(international_monoclinic=True)
        except Exception:
            pass

        # Deterministic ordering of sites by (Z, frac coords)
        frac = np.mod(np.array(s.frac_coords, dtype=float), 1.0)
        Z = np.array([int(getattr(site.specie, "Z", site.specie.number)) for site in s.sites], dtype=np.int32)

        order = np.lexsort((
            np.round(frac[:, 2], decimals),
            np.round(frac[:, 1], decimals),
            np.round(frac[:, 0], decimals),
            Z
        ))
        frac = frac[order]
        Z = Z[order]

        lat = np.array(s.lattice.matrix, dtype=float)
        lat_r = np.round(lat, decimals=decimals)
        frac_r = np.round(frac, decimals=decimals)

        payload = (
            lat_r.astype(np.float64).tobytes()
            + Z.astype(np.int32).tobytes()
            + frac_r.astype(np.float64).tobytes()
        )
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return None

def cif_structure(cif_text: str):
    from pymatgen.core import Structure
    return Structure.from_str(cif_text, fmt="cif")

def load_cifs_by_id(csv_path: Path, issues: IssueLog) -> Dict[str, str]:
    """
    Load mapping: material_id -> cif string from a split csv.
    """
    if not csv_path.exists():
        issues.add_fail(f"Missing expected split file: {csv_path}")
        return {}

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
    if "material_id" not in df.columns or "cif" not in df.columns:
        issues.add_fail(
            f"{csv_path} must contain columns 'material_id' and 'cif'. Found {list(df.columns)}"
        )
        return {}

    out: Dict[str, str] = {}
    for mid, cif in zip(df["material_id"].astype(str), df["cif"].astype(str)):
        mid = mid.strip()
        if mid == "":
            continue
        out.setdefault(mid, _norm_cif_text(cif))
    return out

def _collect_overlapping_tokens(
    tokens_by_split: Dict[str, Set[str]]
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Return:
      - union_overlaps: all tokens that overlap across ANY split pair
      - pair_to_tokens: mapping "a|b" -> overlapping token set
    """
    pair_to_tokens: Dict[str, Set[str]] = {}
    union_overlaps: Set[str] = set()
    for a, b in _overlap_pairs():
        inter = tokens_by_split.get(a, set()) & tokens_by_split.get(b, set())
        key = f"{a}|{b}"
        if inter:
            pair_to_tokens[key] = inter
            union_overlaps |= inter
    return union_overlaps, pair_to_tokens

def _report_token_overlaps(
    name: str,
    tag: str,
    union: Set[str],
    pair_to: Dict[str, Set[str]],
    tok_to_ids_by_split: Dict[str, Dict[str, List[str]]],
    issues: IssueLog,
) -> None:
    tok_to_pairs: Dict[str, List[str]] = {t: [] for t in union}
    for pair, toks in pair_to.items():
        for t in toks:
            tok_to_pairs.setdefault(t, []).append(pair)

    for t in sorted(tok_to_pairs.keys()):
        pairs = ",".join(sorted(tok_to_pairs[t]))
        issues.add_leak_hash_line(
            f"{name} {tag}  {t}  splits={pairs}  "
            f"train={_short_ids(tok_to_ids_by_split['train'].get(t, []))}  "
            f"val={_short_ids(tok_to_ids_by_split['val'].get(t, []))}  "
            f"test={_short_ids(tok_to_ids_by_split['test'].get(t, []))}"
        )

def leakage_check(
    name: str,
    maps: Dict[str, Dict[str, str]],
    to_structure: Callable[[str], object],
    text_tag: str,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    issues: IssueLog,
) -> None:
    """
    For one family's splits, maps[split] = {material_id: exact structure text},
    check overlap across splits by:
      - material_id
      - exact text hash (normalized CIF for CDVAE/FlowMM, JSON for ALIGNN-CSP)
      - structure hash (canonicalized)

    Output policy: print hash tokens AND example IDs per split (no snippets);
    accumulate results; do not early-exit.
    """
    # If reading failed badly, don't cascade; issues already recorded.
    if not any(maps.get(sp) for sp in SPLITS):
        return

    # 1) material_id overlap (hard leakage)
    ids = {k: set(v.keys()) for k, v in maps.items()}
    for a, b in _overlap_pairs():
        inter = ids.get(a, set()) & ids.get(b, set())
        if inter:
            issues.add_leak_summary(f"{name}: material_id overlap between {a} and {b}: {len(inter)}")

    # 2) exact text hash overlap
    text_tokens_by_split: Dict[str, Set[str]] = {}
    text_tok_to_ids_by_split: Dict[str, Dict[str, List[str]]] = {sp: {} for sp in SPLITS}
    for sp, mp in maps.items():
        for mid, text in mp.items():
            tok = _sha256_hex(text)
            text_tokens_by_split.setdefault(sp, set()).add(tok)
            text_tok_to_ids_by_split[sp].setdefault(tok, []).append(mid)

    text_union, text_pair_to = _collect_overlapping_tokens(text_tokens_by_split)
    if text_union:
        issues.add_leak_summary(f"{name}: {text_tag}-text hash overlaps across splits: {len(text_union)}")
        _report_token_overlaps(name, f"{text_tag:<4}", text_union, text_pair_to, text_tok_to_ids_by_split, issues)

    # 3) STRUCTURE hash overlap (tolerant canonicalization; controlled by symprec/decimals)
    bad = 0
    total = 0
    struct_tokens_by_split: Dict[str, Set[str]] = {sp: set() for sp in SPLITS}
    struct_tok_to_ids_by_split: Dict[str, Dict[str, List[str]]] = {sp: {} for sp in SPLITS}

    for sp, mp in maps.items():
        for mid, text in mp.items():
            total += 1
            try:
                h = structure_hash(to_structure(text), symprec, angle_tolerance, decimals, issues)
            except Exception:
                h = None
            if h is None:
                bad += 1
                continue
            struct_tokens_by_split[sp].add(h)
            struct_tok_to_ids_by_split[sp].setdefault(h, []).append(mid)

    if bad > 0:
        warn(f"{name}: structure-hash parse failures: {bad}/{total} (skipped for STRUCTURE hash overlap checks)")

    struct_union, struct_pair_to = _collect_overlapping_tokens(struct_tokens_by_split)
    if struct_union:
        issues.add_leak_summary(f"{name}: STRUCTURE-hash overlaps across splits: {len(struct_union)}")
        _report_token_overlaps(name, "STRC", struct_union, struct_pair_to, struct_tok_to_ids_by_split, issues)

    if not (text_union or struct_union) and all(
        (ids.get(a, set()) & ids.get(b, set()) == set()) for a, b in _overlap_pairs()
    ):
        ok(f"{name}: no leakage detected across train/val/test (by id, {text_tag}-hash, structure-hash)")

def leakage_check_cdvae_flowmm(
    name: str,
    dir_: Path,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    issues: IssueLog,
) -> None:
    maps = {sp: load_cifs_by_id(dir_ / f"{sp}.csv", issues) for sp in SPLITS}
    leakage_check(name, maps, cif_structure, "CIF", symprec, angle_tolerance, decimals, issues)


# --------------------------- Core assertions (accumulating) ---------------------------

def assert_no_dups(label: str, ids: Sequence[str], issues: IssueLog) -> None:
    d = _dups(ids)
    if d:
        uniq = sorted(list(set(d)))
        issues.add_fail(f"{label}: duplicates within split: {len(d)} (unique duplicated IDs: {len(uniq)})")

def assert_disjoint(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str], issues: IssueLog) -> None:
    inter = _set(A) & _set(B)
    if inter:
        issues.add_fail(f"Split overlap: {labelA} ∩ {labelB} has {len(inter)} IDs")

def assert_equal_sets(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str], issues: IssueLog) -> bool:
    SA, SB = _set(A), _set(B)
    if SA != SB:
        issues.add_fail(
            f"Set mismatch: {labelA} vs {labelB}  "
            f"({len(SA - SB)} only-in-{labelA}, {len(SB - SA)} only-in-{labelB})"
        )
        return False
    return True

def assert_equal_order(labelA: str, A: Sequence[str], labelB: str, B: Sequence[str], issues: IssueLog) -> None:
    if list(_as_list(A)) != list(_as_list(B)):
        issues.add_fail(f"Order mismatch: {labelA} != {labelB} (use without --strict-order to compare as sets)")


# --------------------------- ALIGNN-CSP against AtomBench's models ---------------------------

@dataclass
class Reference:
    """One AtomBench model's split, as AtomBench's preprocessing wrote it."""
    name: str
    test: List[str]
    train: Optional[List[str]] = None       # set when the model keeps train and val apart
    val: Optional[List[str]] = None
    train_pool: Optional[List[str]] = None  # everything it trained or validated on
    cells: Dict[str, str] = field(default_factory=dict)      # test id -> structure text
    targets: Dict[str, float] = field(default_factory=dict)  # test id -> property value

    def pool(self) -> Set[str]:
        if self.train_pool is not None:
            return _set(self.train_pool)
        return _set(self.train or []) | _set(self.val or [])

def reference_from_csvs(name: str, dir_: Path) -> Optional[Reference]:
    frames = {}
    for sp in SPLITS:
        p = dir_ / f"{sp}.csv"
        if not p.exists():
            return None  # already reported
        frames[sp] = pd.read_csv(p, dtype=str, keep_default_na=False, na_filter=False)
    if any("material_id" not in f.columns for f in frames.values()):
        return None  # already reported
    test = frames["test"]
    ids = _as_list(test["material_id"])
    prop = test.columns[-1]  # both AtomBench CSV factories write the property last
    return Reference(
        name=name,
        test=ids,
        train=_as_list(frames["train"]["material_id"]),
        val=_as_list(frames["val"]["material_id"]),
        cells=dict(zip(ids, test["cif"])) if "cif" in test.columns else {},
        targets={m: float(v) for m, v in zip(ids, test[prop])},
    )

def reference_from_atomgpt(dir_: Path, df: pd.DataFrame, n_test: int) -> Optional[Reference]:
    if df.empty or not 0 < n_test < len(df):
        return None  # already reported by atomgpt_split
    head, tail = df.iloc[:-n_test], df.iloc[-n_test:]
    cells = {}
    for mid, rel in zip(tail["id"], tail["path"]):
        p = dir_ / rel
        if p.is_file():
            cells[mid] = p.read_text()
    return Reference(
        name="AtomGPT",
        test=tail["id"].tolist(),
        train_pool=head["id"].tolist(),
        cells=cells,
        targets={m: float(v) for m, v in zip(tail["id"], tail["target"])},
    )

class CrystalComparer:
    """StructureMatcher on ALIGNN-CSP rows against AtomBench cells, cached per (id, text)."""

    def __init__(self, ltol: float, stol: float, angle_tol: float):
        from pymatgen.analysis.structure_matcher import StructureMatcher
        self.matcher = StructureMatcher(
            ltol=ltol, stol=stol, angle_tol=angle_tol,
            primitive_cell=True, scale=False, attempt_supercell=False,
        )
        self.cache: Dict[Tuple[str, str], Tuple[Optional[bool], Optional[float]]] = {}

    def same(self, row: dict, ref_text: str) -> Tuple[Optional[bool], Optional[float]]:
        """(True, rms) on a match, (False, None) on a mismatch, (None, None) if unparsable."""
        key = (str(row["material_id"]), _sha256_hex(str(ref_text)))
        if key not in self.cache:
            try:
                a = csp_structure(csp_text(row))
                b = parse_cell(ref_text)
            except Exception:
                self.cache[key] = (None, None)
            else:
                if a.composition.reduced_composition != b.composition.reduced_composition:
                    self.cache[key] = (False, None)
                else:
                    rms = self.matcher.get_rms_dist(a, b)
                    self.cache[key] = (rms is not None, None if rms is None else float(rms[0]))
        return self.cache[key]

def compare_to_reference(
    csp: Dict[str, List[dict]],
    ref: Reference,
    comparer: Optional[CrystalComparer],
    strict_order: bool,
    issues: IssueLog,
) -> dict:
    c_train, c_val, c_test = (csp_ids(csp[sp]) for sp in SPLITS)
    A, B = _set(c_test), _set(ref.test)
    tag = f"ALIGNN-CSP vs {ref.name}"
    n_fail = len(issues.hard_failures)

    row: dict = {
        "model": ref.name,
        "n_test_atombench": len(B),
        "n_test_alignn_csp": len(A),
        "common": len(A & B),
        "only_alignn_csp": sorted(A - B),
        "only_atombench": sorted(B - A),
        "same_order": c_test == _as_list(ref.test),
    }

    # ---------- split membership ----------
    if assert_equal_sets("ALIGNN-CSP test", c_test, f"{ref.name} test", ref.test, issues) and strict_order:
        assert_equal_order("ALIGNN-CSP test", c_test, f"{ref.name} test", ref.test, issues)
    if ref.train is not None and ref.val is not None:
        assert_equal_sets("ALIGNN-CSP train", c_train, f"{ref.name} train", ref.train, issues)
        assert_equal_sets("ALIGNN-CSP val", c_val, f"{ref.name} val", ref.val, issues)
    else:
        assert_equal_sets("ALIGNN-CSP train∪val", list(_set(c_train) | _set(c_val)),
                          f"{ref.name} train(head)", list(ref.pool()), issues)

    # ---------- cross-contamination ----------
    in_train, in_val = sorted(_set(c_train) & B), sorted(_set(c_val) & B)
    leaked = sorted(ref.pool() & A)
    row["atombench_test_in_alignn_csp_train"] = in_train
    row["atombench_test_in_alignn_csp_val"] = in_val
    row["alignn_csp_test_in_atombench_train"] = leaked
    if in_train or in_val:
        issues.add_fail(
            f"{tag}: {len(in_train) + len(in_val)} AtomBench test IDs are in ALIGNN-CSP training data "
            f"({len(in_train)} train, {len(in_val)} val), e.g. {_short_ids(in_train + in_val)}"
        )
    if leaked:
        issues.add_fail(
            f"{tag}: {len(leaked)} ALIGNN-CSP test IDs are in {ref.name}'s training data, e.g. {_short_ids(leaked)}"
        )

    # ---------- content of the shared test targets ----------
    by_id = {str(r["material_id"]).strip(): r for r in csp["test"]}
    prop_bad, crystal_bad, unparsable, rms = [], [], [], []
    for mid in sorted(A & B):
        if mid in ref.targets and abs(float(by_id[mid]["target"]) - ref.targets[mid]) > 1e-6:
            prop_bad.append(mid)
        if comparer is not None and mid in ref.cells:
            same, r = comparer.same(by_id[mid], ref.cells[mid])
            if same is None:
                unparsable.append(mid)
            elif not same:
                crystal_bad.append(mid)
            else:
                rms.append(r)
    row.update({
        "property_compared": bool(ref.targets),
        "property_mismatch": prop_bad,
        "crystal_compared": comparer is not None and bool(ref.cells),
        "crystal_mismatch": crystal_bad,
        "crystal_unparsable": unparsable,
        "crystal_rms_max": max(rms) if rms else None,
    })
    if prop_bad:
        issues.add_fail(f"{tag}: property differs on {len(prop_bad)} test IDs, e.g. {_short_ids(prop_bad)}")
    if crystal_bad:
        issues.add_fail(f"{tag}: different crystal on {len(crystal_bad)} test IDs, e.g. {_short_ids(crystal_bad)}")
    if unparsable:
        issues.add_fail(f"{tag}: could not parse {len(unparsable)} test structures, e.g. {_short_ids(unparsable)}")

    row["passed"] = len(issues.hard_failures) == n_fail
    if row["passed"]:
        ok(f"{tag}: identical split, properties and crystals")
    return row

def check_bench_csv(
    csp: Dict[str, List[dict]],
    path: Path,
    comparer: Optional[CrystalComparer],
    strict_order: bool,
    issues: IssueLog,
) -> dict:
    """The CSV ALIGNN-CSP was scored from must hold its test set, with its targets."""
    label = f"bench CSV {'/'.join(path.parts[-4:])}"
    n_fail = len(issues.hard_failures)
    rows = read_benchmark_csv(path, issues)
    ids = [i for i, _ in rows]
    c_test = csp_ids(csp["test"])
    by_id = {str(r["material_id"]).strip(): r for r in csp["test"]}

    assert_no_dups(label, ids, issues)
    if assert_equal_sets(label, ids, "ALIGNN-CSP test", c_test, issues) and strict_order:
        assert_equal_order(label, ids, "ALIGNN-CSP test", c_test, issues)

    target_bad = []
    for mid, target in rows:
        if mid not in by_id or _norm_poscar(target) == _norm_poscar(by_id[mid].get("target_poscar", "")):
            continue
        same = comparer.same(by_id[mid], target)[0] if comparer is not None else False
        if not same:
            target_bad.append(mid)
    if target_bad:
        issues.add_fail(f"{label}: target differs from test.json on {len(target_bad)} IDs, e.g. {_short_ids(target_bad)}")

    passed = len(issues.hard_failures) == n_fail
    if passed:
        ok(f"{label}: {len(ids)} rows == ALIGNN-CSP test set, targets identical")
    return {"path": str(path), "rows": len(ids), "target_mismatch": target_bad, "passed": passed}

def print_summary(rows: List[dict]) -> None:
    head = ("AtomBench model", "n_test", "n_csp", "common", "only_csp", "only_atombench", "order",
            "atombench_test∈csp_trn/val", "csp_test∈atombench_trn", "Tc≠", "crystal≠", "rms_max")
    table = [head]
    for r in rows:
        table.append((
            r["model"], str(r["n_test_atombench"]), str(r["n_test_alignn_csp"]), str(r["common"]),
            str(len(r["only_alignn_csp"])), str(len(r["only_atombench"])), "same" if r["same_order"] else "diff",
            str(len(r["atombench_test_in_alignn_csp_train"]) + len(r["atombench_test_in_alignn_csp_val"])),
            str(len(r["alignn_csp_test_in_atombench_train"])),
            str(len(r["property_mismatch"])) if r["property_compared"] else "n/a",
            str(len(r["crystal_mismatch"])) if r["crystal_compared"] else "n/a",
            "n/a" if r["crystal_rms_max"] is None else f"{r['crystal_rms_max']:.2e}",
        ))
    widths = [max(len(t[i]) for t in table) for i in range(len(head))]
    print("\nALIGNN-CSP split against each AtomBench model's split")
    for i, t in enumerate(table):
        print("  " + "  ".join(c.ljust(w) for c, w in zip(t, widths)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))
    print()


# --------------------------- main ---------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="ALIGNN-CSP vs AtomBench split auditor (hard-fail after full run).")
    ap.add_argument("--alignn-csp-dir", required=True, type=Path,
                    help="Prepared ALIGNN-CSP split: train.json, val.json, test.json, split_meta.json.")
    ap.add_argument("--label", default=None, help="Name of this comparison, recorded in the report.")

    ap.add_argument("--atomgpt-dir", type=Path, help="AtomBench's AtomGPT split (id_prop.csv + POSCARs).")
    ap.add_argument("--cdvae-dir", type=Path, help="AtomBench's CDVAE split (train/val/test.csv).")
    ap.add_argument("--flowmm-dir", type=Path, help="AtomBench's FlowMM split (train/val/test.csv).")
    ap.add_argument("--bench-csv", action="append", default=[], type=Path,
                    help="A benchmark CSV ALIGNN-CSP was scored from; repeatable.")

    ap.add_argument("--strict-order", action="store_true",
                    help="Require identical ordering where applicable (test sets).")
    ap.add_argument("--n-test", type=int, default=None,
                    help="Override AtomGPT test length; default: infer from CDVAE test.csv length.")

    ap.add_argument("--no-structure-leakage", action="store_true",
                    help="Skip CIF/structure hashing leakage checks.")
    ap.add_argument("--symprec", type=float, default=1e-2,
                    help="symprec for symmetry standardization in structure hashing.")
    ap.add_argument("--angle-tolerance", type=float, default=5.0,
                    help="angle_tolerance for symmetry standardization in structure hashing.")
    ap.add_argument("--decimals", type=int, default=6,
                    help="Rounding decimals for structure hashing.")

    ap.add_argument("--no-structure-compare", action="store_true",
                    help="Skip StructureMatcher comparison of test targets.")
    ap.add_argument("--ltol", type=float, default=0.02, help="StructureMatcher ltol for target comparison.")
    ap.add_argument("--stol", type=float, default=0.05, help="StructureMatcher stol for target comparison.")
    ap.add_argument("--match-angle-tol", type=float, default=1.0,
                    help="StructureMatcher angle_tol (degrees) for target comparison.")

    ap.add_argument("--report", type=Path, default=None, help="Write the full result as JSON here.")

    args = ap.parse_args(argv)
    issues = IssueLog()
    report: dict = {
        "label": args.label,
        "alignn_csp_dir": str(args.alignn_csp_dir),
        "atombench_dirs": {"atomgpt": str(args.atomgpt_dir), "cdvae": str(args.cdvae_dir),
                           "flowmm": str(args.flowmm_dir)},
        "strict_order": args.strict_order,
    }

    # ---------- ALIGNN-CSP split hygiene ----------
    csp = alignn_csp_splits(args.alignn_csp_dir, issues)
    c_train, c_val, c_test = (csp_ids(csp[sp]) for sp in SPLITS)
    for sp, ids in zip(SPLITS, (c_train, c_val, c_test)):
        assert_no_dups(f"ALIGNN-CSP {sp}", ids, issues)
    assert_disjoint("ALIGNN-CSP train", c_train, "ALIGNN-CSP val", c_val, issues)
    assert_disjoint("ALIGNN-CSP train", c_train, "ALIGNN-CSP test", c_test, issues)
    assert_disjoint("ALIGNN-CSP val", c_val, "ALIGNN-CSP test", c_test, issues)

    csp_hash = hash10(c_train + c_val + c_test)
    report["alignn_csp_split"] = {"n_train": len(c_train), "n_val": len(c_val), "n_test": len(c_test),
                                  "hash10_ids": csp_hash}
    meta_path = args.alignn_csp_dir / "split_meta.json"
    if meta_path.exists():
        recorded = json.loads(meta_path.read_text()).get("hash10_ids")
        if recorded != csp_hash:
            issues.add_fail(f"ALIGNN-CSP split_meta.json hash10_ids={recorded} but the JSONs hash to {csp_hash}")
        else:
            ok(f"ALIGNN-CSP JSONs reproduce split_meta.json hash10_ids={csp_hash}")
    else:
        warn(f"{meta_path} not found; hash10 fingerprint not checked")

    refs: List[Reference] = []

    # ---------- AtomBench's AtomGPT / CDVAE / FlowMM splits (original checker) ----------
    artifact_dirs = (args.atomgpt_dir, args.cdvae_dir, args.flowmm_dir)
    if any(artifact_dirs) and not all(artifact_dirs):
        issues.add_fail("--atomgpt-dir, --cdvae-dir and --flowmm-dir must be given together")
    elif all(artifact_dirs):
        n_before = len(issues.hard_failures)

        # Load CDVAE/FlowMM splits (explicit)
        cd_train, cd_val, cd_test = cdvae_splits(args.cdvae_dir, issues)
        fm_train, fm_val, fm_test = flowmm_splits(args.flowmm_dir, issues)

        # Infer AtomGPT test size
        n_test = args.n_test if args.n_test is not None else len(cd_test)
        ag_table = atomgpt_table(args.atomgpt_dir, issues)
        ag_train, ag_test = atomgpt_split(ag_table["id"].tolist(), n_test=n_test, issues=issues)

        # ---------- Basic split hygiene ----------
        assert_no_dups("CDVAE train", cd_train, issues)
        assert_no_dups("CDVAE val", cd_val, issues)
        assert_no_dups("CDVAE test", cd_test, issues)

        assert_no_dups("FlowMM train", fm_train, issues)
        assert_no_dups("FlowMM val", fm_val, issues)
        assert_no_dups("FlowMM test", fm_test, issues)

        assert_no_dups("AtomGPT train(head)", ag_train, issues)
        assert_no_dups("AtomGPT test(tail)", ag_test, issues)

        assert_disjoint("CDVAE train", cd_train, "CDVAE val", cd_val, issues)
        assert_disjoint("CDVAE train", cd_train, "CDVAE test", cd_test, issues)
        assert_disjoint("CDVAE val", cd_val, "CDVAE test", cd_test, issues)

        assert_disjoint("FlowMM train", fm_train, "FlowMM val", fm_val, issues)
        assert_disjoint("FlowMM train", fm_train, "FlowMM test", fm_test, issues)
        assert_disjoint("FlowMM val", fm_val, "FlowMM test", fm_test, issues)

        assert_disjoint("AtomGPT train(head)", ag_train, "AtomGPT test(tail)", ag_test, issues)

        if len(issues.hard_failures) == n_before:
            ok("Within-family split disjointness and de-duplication checks passed.")

        # ---------- Cross-model equivalence per requested rules ----------
        n_before = len(issues.hard_failures)
        assert_equal_sets("CDVAE test", cd_test, "FlowMM test", fm_test, issues)
        assert_equal_sets("CDVAE test", cd_test, "AtomGPT test(tail)", ag_test, issues)

        if args.strict_order:
            assert_equal_order("CDVAE test", cd_test, "FlowMM test", fm_test, issues)
            assert_equal_order("CDVAE test", cd_test, "AtomGPT test(tail)", ag_test, issues)

        cd_train_equiv = list(_set(cd_train) | _set(cd_val))
        fm_train_equiv = list(_set(fm_train) | _set(fm_val))

        assert_equal_sets("AtomGPT train(head)", ag_train, "CDVAE train∪val", cd_train_equiv, issues)
        assert_equal_sets("AtomGPT train(head)", ag_train, "FlowMM train∪val", fm_train_equiv, issues)

        if len(issues.hard_failures) == n_before:
            ok("Cross-model split equivalence checks passed (using the requested mapping).")

        # ---------- The fingerprint AtomBench prints and ALIGNN-CSP records ----------
        if not ag_table.empty:
            ag_hash = hash10(ag_table["id"].tolist())
            report["atomgpt_hash10_ids"] = ag_hash
            if args.strict_order:
                if ag_hash != csp_hash:
                    issues.add_fail(f"Fingerprint mismatch: AtomGPT id_prop.csv hash10={ag_hash}, ALIGNN-CSP hash10={csp_hash}")
                else:
                    ok(f"AtomGPT id_prop.csv and ALIGNN-CSP share hash10_ids={csp_hash}")

        refs = [r for r in (
            reference_from_csvs("CDVAE", args.cdvae_dir),
            reference_from_csvs("FlowMM", args.flowmm_dir),
            reference_from_atomgpt(args.atomgpt_dir, ag_table, n_test),
        ) if r is not None]
    elif not args.bench_csv:
        issues.add_fail("Nothing to compare against: give --atomgpt-dir/--cdvae-dir/--flowmm-dir and/or --bench-csv")

    # ---------- ALIGNN-CSP against each AtomBench model ----------
    comparer = None
    if not args.no_structure_compare and _import_pymatgen(issues)[0] is not None:
        comparer = CrystalComparer(args.ltol, args.stol, args.match_angle_tol)
    report["structure_matcher"] = None if comparer is None else {
        "ltol": args.ltol, "stol": args.stol, "angle_tol": args.match_angle_tol,
        "primitive_cell": True, "scale": False,
    }

    rows = [compare_to_reference(csp, r, comparer, args.strict_order, issues) for r in refs]
    report["models"] = rows
    if rows:
        print_summary(rows)

    report["bench_csvs"] = [check_bench_csv(csp, p, comparer, args.strict_order, issues) for p in args.bench_csv]

    # ---------- Structural leakage checks ----------
    if not args.no_structure_leakage:
        if all(artifact_dirs):
            leakage_check_cdvae_flowmm(
                name="CDVAE",
                dir_=args.cdvae_dir,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                decimals=args.decimals,
                issues=issues,
            )
            leakage_check_cdvae_flowmm(
                name="FlowMM",
                dir_=args.flowmm_dir,
                symprec=args.symprec,
                angle_tolerance=args.angle_tolerance,
                decimals=args.decimals,
                issues=issues,
            )
        leakage_check(
            "ALIGNN-CSP",
            {sp: {str(r["material_id"]).strip(): csp_text(r) for r in csp[sp]} for sp in SPLITS},
            csp_structure,
            "JSON",
            args.symprec,
            args.angle_tolerance,
            args.decimals,
            issues,
        )
    else:
        warn("Skipping CIF/structure hashing leakage checks (--no-structure-leakage).")

    issues.report_and_exit(report, args.report)


if __name__ == "__main__":
    main()
