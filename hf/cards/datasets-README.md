---
license: other
tags: [materials-science, crystal-structure-prediction, jarvis, alexandria, atombench]
pretty_name: ALIGNN-CSP ablation splits
---

# ALIGNN-CSP ablation datasets

The exact prepared splits the ablations in
https://github.com/crhysc/alignn-csp-ablation were trained and scored on, and
the raw inputs one of them was derived from. Every structure is stored in
AtomBench's primitive + Niggli scoring basis.

```
jarvis_supercon3d/{train,val,test}.json + split_meta.json
    JARVIS-DFT dft_3d, superconducting-Tc subset ("Supercon-3D"): 847 / 105 / 103
    {"dataset":"dft_3d","target_key":"Tc_supercon","seed":123,"max_size":1058,"hash10_ids":"3fd288e327"}
    derived by alignn/scripts/atombench/prepare_data.py from the jarvis-tools dft_3d download
alexandria_dsab/{train,val,test}.json + split_meta.json
    Alexandria DS-A/DS-B, the AtomBench split: 6603 / 825 / 825
    {"dataset":"alexandria_DS-A_DS-B","target_key":"Tc","seed":123,"max_size":8253,"hash10_ids":"5703564835"}
    derived by alignn/scripts/atombench/prepare_alex_data.py from raw/DS-A.pk.bz2 + raw/DS-B.pk.bz2 (included)
```

Record format: `material_id, formula, spacegroup, target, lattice_mat (3x3),
frac_coords (Nx3), atomic_numbers, elements, target_poscar`.

`hash10_ids` is the first ten hex digits of the SHA of the sorted id list;
`preflight.sh` in the harness checks the split on disk against it.

Upstream data: JARVIS-DFT (Choudhary et al.) and Alexandria (Schmidt et al.),
under their own licences; AtomBench (Campbell et al., 2026) defines the
DS-A/DS-B split and the metrics.
