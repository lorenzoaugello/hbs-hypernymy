#!/usr/bin/env python3
"""Create work/hbs_dataset.pkl with the WordNet builder from hierarchy_hbs_2."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from hbs_dataset import CFG, build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild and replace an existing dataset pickle")
    args = parser.parse_args()
    work = ROOT / CFG["WORK_DIR"]
    work.mkdir(parents=True, exist_ok=True)
    data_path = work / "hbs_dataset.pkl"
    if data_path.exists() and not args.force:
        parser.error(f"{data_path.relative_to(ROOT)} already exists; pass --force to rebuild it")

    import nltk
    import pandas as pd
    import scipy
    
    wdf, pairs, chains = build_dataset(seed=CFG["SEED"])
    pd.to_pickle((wdf, pairs, chains), data_path)
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    manifest = {
        "source_notebook": "notebooks/hierarchy_hbs_2_source.ipynb",
        "config": {k: v for k, v in CFG.items() if k not in {"TEMPLATES", "MODELS"}},
        "counts": {
            "words": len(wdf),
            "branches": int(wdf.branch.nunique()),
            "chains": len(chains),
            "pair_rows": len(pairs),
        },
        "sha256": digest,
        "runtime_versions": {
            "python": sys.version.split()[0],
            "nltk": nltk.__version__,
            "wordfreq": importlib.metadata.version("wordfreq"),
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "notes": "WordNet and wordfreq data versions are not recorded by the source notebook; preserve this manifest with the generated pickle.",
    }
    (work / "dataset_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {data_path.relative_to(ROOT)}")
    print(f"{len(wdf):,} words | {wdf.branch.nunique():,} branches | {len(chains):,} chains | {len(pairs):,} pair rows")
    print(f"SHA-256: {digest}")
    if (len(wdf), int(wdf.branch.nunique()), len(chains)) != (6875, 477, 1307):
        print("WARNING: counts differ from the supplied notebook's recorded run (6,875 words, 477 branches, 1,307 chains). Check WordNet/wordfreq package and data versions.")


if __name__ == "__main__":
    main()
