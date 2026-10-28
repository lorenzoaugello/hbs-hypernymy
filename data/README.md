# Input data

No precomputed dataset is bundled. Generate it from the included WordNet construction code by running `python scripts/prepare_dataset.py` from the repository root. The script writes `work/hbs_dataset.pkl` and `work/dataset_build_manifest.json`; both are excluded from Git.

Our source data records aggregate checks of 6,875 words, 477 taxonomy branches, 1,307 clean chains, and 15,011 pair rows. The manifest records the actual output's SHA-256 and runtime package versions. For exact row-level reproduction, retain the NLTK WordNet corpus and `wordfreq` versions along with the generated data.
