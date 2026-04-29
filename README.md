# Gateway Advanced Tech — Course Materials

A hands-on applied-ML course. The "tabular" sections of the course are built around a single synthetic dataset. Modules 1–4 cover tabular ML (classification, regression, unsupervised); Module 5 covers NLP; Module 6 covers vision. Every notebook runs top-to-bottom with no external downloads, and every notebook is dual-runnable on Google Colab and locally.

## Local setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd code
uv sync
uv pip install -e .
```

In VSCode (or any Jupyter-aware IDE), select the **`.venv`** kernel for any notebook. The editable install registers the local `core` package so `from core.generators.transaction_risk_dgp import ...` works without `sys.path` hacks.

## Running on Colab

Click any Colab badge in the table below. The first cell of each notebook has a Colab setup block (commented out by default),  uncomment it to clone the repo and put `core` on the Python path. Leave it commented when running locally. (you don't need it locally)

## The tabular dataset (Modules 1–4)

All tabular notebooks generate the same dataset deterministically:

```python
from core.generators.transaction_risk_dgp import generate_transaction_risk_dataset
df = generate_transaction_risk_dataset(seed=1955)
```

- ~126,000 transactions, 24 columns, fixed seed = `1955`
- **Class imbalance**: ~4% fraud rate (`is_fraud`)
- **Continuous target**: `transaction_loss_amount` (zero-inflated, heavy right tail)
- **High-cardinality categoricals**: ~5,000 merchants (Zipf-distributed)
- **MNAR missingness**: `device_type`, `merchant_category` (intentional)
- **Leakage fields** to exclude: `chargeback_flag`, `manual_review_score`, `fraud_probability_latent`

The dataset regenerates on the fly each run, there are no files to download or version.

## Notebook naming convention

Within most modules:

| Suffix | Purpose |
|---|---|
| `_00_main` | Instructor demo, end-to-end walkthrough |
| `_01_exercise_guided` | exercise with structured TODOs |
| `_02_exercise_open` | Open-ended exercise |
| `_03_extension` (etc.) | Optional advanced extension |

## Repository structure

```
code/
├── notebooks/         # All participant-facing course material
├── src/core/          # Installable Python package
│   └── generators/    # Synthetic dataset generators
├── data/              # Cached datasets (gitignored)
├── archive/           # Older versions, scratch we won't use these (gitignored)
├── pyproject.toml     # Dependencies + package config
└── README.md
```

## Cross-notebook conventions

- **Random seed**: `1955` everywhere (any deviation is called out in-cell)
- **Leakage hygiene**: post-outcome fields are listed in a `LEAKAGE_COLS` constant and dropped before modeling
- **Pipelines**: preprocessing always happens inside `sklearn.pipeline.Pipeline` to prevent train/test contamination
- **Metrics under imbalance**: mostly focused on precision/recall/F1 and Average Precision (AP), not raw accuracy

## Notebooks

Quick link to the notebooks

| Module | Description | Colab |
|------|-------------|-------|
| Module 01_00 | Foundations of AI, ML, Data Science | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/01_00_main.ipynb) |
| Module 01_00 (hands-on) | End-to-end ML pipeline walkthrough (breast cancer dataset) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/01_00_handson_machine_learning_pipeline.ipynb)|
| Module 02_00 | Supervised Learning: Regression Demo | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/02_00_main.ipynb)|
| Module 02_01 | Supervised Learning: Regression Exercise | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/02_01_exercise_guided.ipynb)|
| Module 02_01 (solution) | Solution to the 02_01 guided regression exercise | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/solutions_02_01_exercise_guided.ipynb)|
| Module 02_02 | Supervised Learning: Regression Open Exercise | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/02_02_exercise_open.ipynb)|
| Module 02_03 | Regression - Extension | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/02_03_extension.ipynb)|
| Module 03_00 | Classification |[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/03_00_main.ipynb)|
| Module 03_01 | Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/03_01_exercise_guided.ipynb)|
| Module 03_02 | Classification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/03_02_exercise_open.ipynb)|
| Module 04_00 | Unsuper. Learning  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/04_00_main.ipynb)|
| Module 04_01 | Unsuper. Learning Guided | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/04_01_exercise_guided.ipynb)|
| Module 04_02 | Unsuper. Learning - Open | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/04_02_exercise_open.ipynb)|
| Module 05_00 | NLP intuition: tokenization + bag-of-words order-blindness | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/05_00_main_intuition.ipynb)|
| Module 05_01 | Classical NLP: linguistic features + lexicon sentiment + LR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/05_01_main_classical.ipynb)|
| Module 05_02 | TF-IDF + KMeans on a real OpenAlex AI-governance corpus | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/05_02_main_tfidf.ipynb)|
| Module 05_03 | Sentence embeddings + UMAP + c-TF-IDF subfield discovery | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/05_03_main_embeddings.ipynb)|
| Module 05_aux | Rigorous comparative pipeline (log-odds w/ Dirichlet prior) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/05_aux_v2_pipeline.ipynb)|
| Module 06_00 | Vision intuition: images as arrays, convolution by hand, pooling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/06_00_main_intuition.ipynb)|
| Module 06_01 | Classical CV: hand-engineered features + LR/RF on EuroSAT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/06_01_main_classical.ipynb)|
| Module 06_02 | CNN from scratch: dense baseline vs small CNN on EuroSAT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/06_02_main_cnn.ipynb)|
| Module 06_03 | Transfer learning: frozen backbone + fine-tuning on EuroSAT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunnel-ai/way/blob/main/notebooks/06_03_main_transfer.ipynb)|
