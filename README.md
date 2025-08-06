# PERMA‑GNN‑Transformer: Cross‑Cultural Student Well‑being Prediction

*Official reproducibility package for the manuscript*

---

## 1. Overview

This repository accompanies the paper **“PERMA‑Guided Multi‑Topology Graph Neural Networks for Cross‑Cultural Student Well‑being Prediction.”**
It provides the full codebase, pre‑processed datasets, trained checkpoints, and experiment scripts required to reproduce every table and figure in the paper (NeurIPS’25 submission ID #3217).

Key contents

| Section                                   | What you will find                                                      |
| ----------------------------------------- | ----------------------------------------------------------------------- |
| [`/data`](#-datasets)                     | Kaggle & Zenodo links, download helpers, preprocessing pipeline         |
| [`/src`](#-code-structure)                | PERMA‑GNN‑Transformer implementation (PyTorch 2 + PyG 2)                |
| [`/configs`](#-quick-start)               | Hydra YAMLs that recreate *Table 1* and *Figure 2*                      |
| [`/notebooks`](#-analysis--visualisation) | Exploratory notebooks & attention‑map visualisation                     |
| [`/pretrained`](#-checkpoints)            | Ready‑to‑use models for **Lifestyle & Wellbeing** and **ISMH** datasets |

---

## 2. Contributions

1. **Theory‑driven representation learning** – first structured integration of Seligman’s PERMA theory with deep neural networks.
2. **Multi‑topology GNN** – four complementary student graphs (cosine, Euclidean, learning‑style, PERMA‑weighted) fused via graph‑level attention.
3. **PERMA‑aligned Transformer** – five dedicated attention heads (P/E/R/M/A) delivering interpretable predictions.
4. **Cross‑cultural validation** – consistent 18.9 % → 27.8 % MAE reduction on Western (n = 12,757) and East‑Asian (n = 268) cohorts.

---

## 3. Datasets  <a name="datasets"></a>

| Dataset                                        | Domain / Culture                                    | Split         | Link                                                                                                                                                                                       |
| ---------------------------------------------- | --------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Lifestyle & Wellbeing Data**                 | Western‑adult survey (12 757×23)                    | 70 / 20 / 10  | [https://www.kaggle.com/datasets/dartweichen/student-life](https://www.kaggle.com/datasets/dartweichen/student-life)                                                                       |
| **International Student Mental Health (ISMH)** | East‑Asian university cohort (268×29)               | 70 / 20 / 10  | [https://www.kaggle.com/datasets/walassetomaz/pisa-results-2000-2022-economics-and-education](https://www.kaggle.com/datasets/walassetomaz/pisa-results-2000-2022-economics-and-education) |
| **StudentWellbeing v1.0** (processed)          | Harmonised PERMA labels, four graphs, splits, stats | —             | [Zenodo 10.5281/zenodo.1111111](https://zenodo.org/record/1111111)                                                                                                                         |

Download all raw files **once**:

```bash
bash scripts/download_data.sh  # ~300 MB
```

This script verifies SHA‑256 checksums and unpacks to `data/raw/`.

---

## 4. Environment

```bash
conda create -n perma_gnn python=3.10 pytorch=2.1 pyg=2.4 -c pytorch -c pyg
conda activate perma_gnn
pip install -r requirements.txt  # hydra-core, wandb, captum, rich...
```

GPU ✧ NVIDIA ≥ Ampere (tested on RTX 4090); CPU‑only training is possible but slow.

---

## 5. Code Structure  <a name="code-structure"></a>

```text
src/
├── datasets/           # loaders + graph builders
├── models/
│   ├── perma_gnn.py    # PERMA‑GNN‑Transformer ⟵ Section 3.2 of paper
│   └── gnn_baselines/  # GCN, GAT, GraphSAGE, SGFormer
├── train.py            # entry point (Hydra driven)
└── evaluate.py         # metrics + PERMA consistency check
```

---

## 6. Quick Start  <a name="quick-start"></a>

```bash
# 1. Train on Lifestyle & Wellbeing
python src/train.py ++config=configs/lwd.yaml

# 2. Evaluate checkpoint on test split
python src/evaluate.py ckpt=pretrained/lwd_perma.pt datamodule=lwd

# 3. Reproduce ablation Figure 2
python src/experiments/run_ablation.py
```

All hyper‑parameters (learning rate 1e‑4, batch 32, hidden 256, heads 5) are stored in YAMLs.

---

## 7. Analysis & Visualisation  <a name="analysis--visualisation"></a>

Open `notebooks/attention_maps.ipynb` to inspect:

* five‑head alignment (0.85 → 0.94 diagonal)
* graph‑level attention weights per student type
* PERMA radar plots for high/medium/low wellbeing cases

---

## 8. Checkpoints  <a name="checkpoints"></a>

```bash
wget https://zenodo.org/record/1111111/files/lwd_perma.pt -P pretrained/
wget https://zenodo.org/record/1111111/files/ismh_perma.pt -P pretrained/
```

These reproduce *Table 1* exactly (seed = 42).

---

## 9. Citation

```bibtex
@inproceedings{zhang2025perma,
  title     = {PERMA-Guided Multi-Topology Graph Neural Networks for Cross-Cultural Student Well-being Prediction},
  author    = {Zhang, YuChen and ...},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  note      = {Reproducibility package available at https://github.com/<user>/perma-gnn-transformer}
}
```

---

## 10. Licence

Source code: **Apache 2.0**
Datasets: CC BY‑NC 4.0 (see individual dataset pages for details).

---

> **Questions?** Open an issue or e‑mail *[i24026647@student.newinti.edu.my](mailto:i24026647@student.newinti.edu.my)* – we are happy to help.
