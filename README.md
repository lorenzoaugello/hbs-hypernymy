# Hierarchy Beyond Similarity: experiment code

This repository accompanies the work behind the paper **“Hierarchy Beyond Similarity: A Comparison between Taxonomic and Similarity Representations in Language Models.”**

## Environment

Use Python 3.10 or 3.11. The dataset preparation uses NLTK WordNet and `wordfreq`; differences in those data/package versions may change the generated dataset. The preparation script writes a manifest with the runtime versions, configuration, aggregate counts, and dataset SHA-256. Retain that manifest with your run outputs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Install a PyTorch build appropriate for your CPU/CUDA setup:
# https://pytorch.org/get-started/locally/
```

The preparation script downloads NLTK's `wordnet` and `omw-1.4` corpora automatically. They can also be installed ahead of time:

```bash
python -m nltk.downloader wordnet omw-1.4
```

The code uses the Hugging Face IDs `gpt2`, `EleutherAI/pythia-1b`, `Qwen/Qwen2.5-1.5B`, and `bert-base-uncased`. The causal interventions use the first three. On first run, Transformers downloads the weights and tokenizers. GPU execution is strongly recommended for steering and ablation; CPU is supported but can take a long time and require substantial memory. Ensure enough disk space for four checkpoints, cached embeddings, and outputs.

## Prepare the dataset

From the repository root, run:

```bash
python scripts/prepare_dataset.py
```

This writes:

- `work/hbs_dataset.pkl` — `(wdf, pairs, chains)`, consumed by the experiment scripts.
- `work/dataset_build_manifest.json` — configuration, package versions, counts, and SHA-256 for the generated pickle.

The dataset build contains 6,875 words, 477 branches, 1,307 chains, and 15,011 pair rows. The script prints a warning if generated aggregate counts differ. To explicitly rebuild and replace an existing pickle, use `python scripts/prepare_dataset.py --force`.

The source builder seeds its Python and NumPy RNGs and uses a CRC32 branch assignment for the 30/70 dev/test split. For exact row-level reproduction, also record the NLTK WordNet corpus version and the installed `wordfreq` version. `work/` is excluded from Git because it contains generated data and large model caches; commit or distribute the manifest separately if your release process requires it.

Check the generated dataset before running the experiments:

```bash
python scripts/check_inputs.py
```

## Run the experiments

We define five code blocks to run (A-E).

Run all blocks in order, including the final evidence ledger:

```bash
python scripts/run_experiments.py all
```

Run one analysis block:

```bash
python scripts/run_experiments.py A  # direction probe
python scripts/run_experiments.py B  # generality and lexical controls
python scripts/run_experiments.py C  # erasure and subspace decoding (also runs B to select layers)
python scripts/run_experiments.py D  # steering (also runs B to select layers)
python scripts/run_experiments.py E  # ablation (also runs B to select layers)
```

Block C, D, and E rely on the development-selected layers computed in Block B. Running those sections independently therefore reruns B first. The full run shares the Python process and cached representations, so it avoids that repeated work. To create the combined diagnostic plot after a full run:

```bash
python scripts/run_experiments.py all --figures
```

We recommend at least 1,000 bootstrap replicates for final results. Changing this setting changes reported intervals and should be documented. GPU kernels, package versions, checkpoint revisions, and bootstrap resampling can also affect exact numeric output. Record `pip freeze` and the resolved Hugging Face revisions with released results.

## Scripts and outputs

| Path | Purpose |
|---|---|
| `src/hbs_dataset.py` | WordNet dataset configuration and builder transcribed from the original data. |
| `scripts/prepare_dataset.py` | Build `work/hbs_dataset.pkl` and its manifest. |
| `src/hbs_core.py` | Shared experiment imports, data/representation loading, and methods. |
| `scripts/blocks/block_a_direction.py` | Block A: leak-free direction probes and frequency subsets. |
| `scripts/blocks/block_b_generality.py` | Block B: generality prediction and lexical confound controls. |
| `scripts/blocks/block_c_erasure.py` | Block C: INLP erasure curves and subspace decoding. |
| `scripts/blocks/block_d_steering_helpers.py`, `block_d_steering.py` | Block D: steering methods and causal-model run. |
| `scripts/blocks/block_e_ablation.py` | Block E: activation ablation and forced-choice evaluation. |
| `scripts/blocks/finalize_ledger.py` | Write an evidence ledger and print summary. |
| `scripts/blocks/make_figures.py` | Produce a combined diagnostic figure. |

CSV outputs are written to `work/followup/`. Embedding caches are written to `work/emb_<model>.npz`. Generated data, cached embeddings, and results are not committed. Keep the input row order unchanged: representations are aligned to `wdf.word` by position.

## Data and artifact disclosure

The data builder derives its examples from WordNet noun synsets, filtered by the criteria and settings preserved in `src/hbs_dataset.py`. The paper PDF is not included in this code repository. 
