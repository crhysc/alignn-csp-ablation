---
license: mit
tags: [dvc, materials-science]
pretty_name: alignn-csp-ablation DVC remote
---

# alignn-csp-ablation — DVC remote

This repository is a byte-for-byte mirror of the DVC remote for
https://github.com/crhysc/alignn-csp-ablation. It is content-addressed
(`files/md5/<xx>/<rest>`) and not meant to be browsed; the browsable mirrors
are `alignn-csp-angular-ablations` (weights) and `alignn-csp-ablation-datasets`
(splits) under the same namespace.

To use it:

```bash
git clone --recurse-submodules https://github.com/crhysc/alignn-csp-ablation.git && cd alignn-csp-ablation
pip install dvc huggingface_hub
HF_NAMESPACE=<this namespace> bash tools/hf_sync.sh pull      # hf download + dvc remote modify + dvc pull
```

DVC (3.67) has no native Hugging Face remote, which is why a directory
mirror is used; see `tools/hf_sync.sh`.
