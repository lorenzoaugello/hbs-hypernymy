#!/usr/bin/env python3
"""Check that the prerequisite dataset and optional representation caches exist."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "hbs_dataset.pkl"
if not DATA.exists():
    sys.exit(
        "Missing work/hbs_dataset.pkl. Run scripts/prepare_dataset.py to create it from the supplied WordNet construction workflow. See README.md, section 'Prepare the dataset'."
    )
try:
    value = pd.read_pickle(DATA)
except Exception as exc:
    sys.exit(f"Could not read {DATA}: {exc}")
if not isinstance(value, (tuple, list)) or len(value) != 3:
    sys.exit("Expected the pickle to contain (wdf, pairs, chains).")
wdf, pairs, chains = value
need_words = {"word", "split", "branch", "synset", "zipf", "ic", "depth", "logndesc"}
need_pairs = {"a", "b", "anchor", "rel", "k", "split"}
missing_w = sorted(need_words - set(wdf.columns))
missing_p = sorted(need_pairs - set(pairs.columns))
if missing_w or missing_p:
    sys.exit(f"Dataset schema mismatch. Missing word columns={missing_w}; pair columns={missing_p}.")
print(f"Dataset OK: {len(wdf):,} words, {wdf.branch.nunique():,} branches, {len(pairs):,} pair rows.")
print(f"Representation caches (optional; missing caches are extracted from Hugging Face models):")
for model in ("gpt2", "EleutherAI_pythia-1b", "Qwen_Qwen2.5-1.5B", "bert-base-uncased"):
    p = ROOT / "work" / f"emb_{model}.npz"
    print(f"  {'present' if p.exists() else 'not present'}  {p.relative_to(ROOT)}")
