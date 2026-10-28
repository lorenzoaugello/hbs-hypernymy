"""WordNet dataset construction, transcribed from hierarchy_hbs_2.ipynb.

This module preserves the notebook's parameters and build_dataset implementation.
"""
# !pip install -q nltk wordfreq scikit-learn scipy pandas numpy matplotlib tqdm torch transformers
import os, re, math, random, zlib, json, warnings
from functools import lru_cache
import numpy as np, pandas as pd
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

CFG = dict(
    SEED=13, WORK_DIR="work",
    # ---- dataset
    MIN_ZIPF=2.5,          # only reasonably common words (wordfreq Zipf scale)
    MIN_DEPTH_A=6,         # anchors are deep nouns so that parent / uncle / grandparent are all non-trivial classes
    MIN_ANC_DEPTH=3,       # ignore ancestors shallower than this (entity, object, ...)
    BRANCH_DEPTH=4,        # branch = ancestor at this depth; used for dev/test and cross-validation grouping
    N_ANCHORS=2500, N_PER_REL=1500, SYN_MAX_RANK=2, DEV_PCT=30,
    # ---- representations
    TEMPLATES=['The {w}.', 'Word: {w}', 'I am thinking about the {w}.'],   # neutral; averaged. Ablate templates!
    MODELS=["gpt2", "EleutherAI/pythia-1b", "Qwen/Qwen2.5-1.5B", "bert-base-uncased"],   # add meta-llama/Llama-3.2-1B, google/gemma-2-2b (gated)
    PREP="center",         # raw | center | whiten
    PROBES=True, EVAL_LAYER_STRIDE=1,
)
os.makedirs(CFG["WORK_DIR"], exist_ok=True); random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])

# --- Original notebook cell 4: WordNet dataset builder ---
import nltk
nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# ---------------------------------------------------------------- helpers on WordNet nouns
@lru_cache(maxsize=None)
def zipf(w): return zipf_frequency(w, "en")

def usable(w):
    return w.isalpha() and w.islower() and len(w) >= 3 and zipf(w) >= CFG["MIN_ZIPF"]

@lru_cache(maxsize=None)
def head(ss, max_rank=1):
    # the first usable single-word lemma whose DOMINANT sense (WordNet rank < max_rank) is this synset
    for l in ss.lemmas():
        w = l.name()
        if usable(w) and ss in wn.synsets(w, pos="n")[:max_rank]: return w
    return None

def tree_like(ss): return len(ss.hypernyms()) == 1 and not ss.instance_hypernyms()

@lru_cache(maxsize=None)
def spath(ss): return tuple(ss.hypernym_paths()[0])             # root ... ss

@lru_cache(maxsize=None)
def clean_chain(ss):
    P = spath(ss); return all(tree_like(s) for s in P[1:])       # unique, unambiguous ancestry

@lru_cache(maxsize=None)
def n_desc(ss): return sum(1 + n_desc(c) for c in ss.hyponyms())

N_NOUNS = 82115
def ic_seco(ss): return 1 - math.log(n_desc(ss) + 1) / math.log(N_NOUNS)   # intrinsic information content (1 = specific)

def branch_of(ss):
    P = spath(ss); return P[min(CFG["BRANCH_DEPTH"], len(P) - 1)].name()

def word_record(w):
    ss = wn.synsets(w, pos="n")[0]; P = spath(ss)
    return dict(word=w, synset=ss.name(), depth=len(P) - 1, ic=ic_seco(ss), logndesc=math.log1p(n_desc(ss)),
                zipf=zipf(w), branch=branch_of(ss), nsenses=len(wn.synsets(w, pos="n")))

# ---------------------------------------------------------------- build pair tables
def build_dataset(seed=CFG["SEED"]):
    rng = random.Random(seed)
    all_syn = [s for s in wn.all_synsets("n") if clean_chain(s) and len(spath(s)) - 1 >= CFG["MIN_DEPTH_A"]]
    rng.shuffle(all_syn)
    pairs, chains, anchors = [], [], 0
    for a in all_syn:
        if anchors >= CFG["N_ANCHORS"]: break
        ha = head(a)
        if not ha: continue
        P = spath(a); p, gp = P[-2], P[-3]
        hp = head(p)
        if not hp or len(P) - 1 - 1 < CFG["MIN_ANC_DEPTH"]: continue
        uncles = [u for u in gp.hyponyms() if u != p and head(u) and u.hyponyms()]      # INTERNAL siblings of the parent
        if not uncles: continue
        u = rng.choice(uncles); anchors += 1
        pairs.append(dict(rel="hyper", a=ha, b=hp, k=1, anchor=ha))
        pairs.append(dict(rel="uncle", a=ha, b=head(u), k=1, anchor=ha))
        sibs = [s for s in p.hyponyms() if s != a and head(s)]
        if sibs: pairs.append(dict(rel="cohyp", a=ha, b=head(rng.choice(sibs)), k=0, anchor=ha))
        anc = {k: (head(P[-1 - k]) if len(P) - 1 - k >= CFG["MIN_ANC_DEPTH"] else None) for k in (1, 2, 3)}
        for k in (2, 3):
            if anc[k]: pairs.append(dict(rel="hyper", a=ha, b=anc[k], k=k, anchor=ha))
        if anc[1] and anc[2]:
            pairs.append(dict(rel="anc_anc", a=anc[1], b=anc[2], k=1, anchor=ha))          # two hypernyms of the same word
            chains.append(dict(a=ha, b=anc[1], c=anc[2]))
    # synonyms, meronyms, antonyms: sampled independently of the anchors
    syn_pairs, mero_pairs, anto_pairs = [], [], []
    for s in wn.all_synsets("n"):
        ws = [l.name() for l in s.lemmas() if usable(l.name()) and s in wn.synsets(l.name(), pos="n")[:CFG["SYN_MAX_RANK"]]]
        ws = list(dict.fromkeys(ws))
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)): syn_pairs.append(dict(rel="syn", a=ws[i], b=ws[j], k=0))
        hs = head(s)
        if hs:
            for m in s.part_meronyms():
                hm = head(m)
                if hm: mero_pairs.append(dict(rel="mero", a=hm, b=hs, k=0))        # part -> whole
        for l in s.lemmas():
            for an in l.antonyms():
                if usable(l.name()) and usable(an.name()): anto_pairs.append(dict(rel="anto", a=l.name(), b=an.name(), k=0))
    for lst in (syn_pairs, mero_pairs, anto_pairs):
        rng.shuffle(lst); pairs += lst[:CFG["N_PER_REL"]]
    base_words = sorted({r["a"] for r in pairs if r["rel"] in ("hyper", "uncle", "cohyp")} | {r["b"] for r in pairs if r["rel"] in ("hyper", "uncle", "cohyp")})
    for _ in range(CFG["N_PER_REL"]):
        a, b = rng.sample(base_words, 2); pairs.append(dict(rel="rand", a=a, b=b, k=0))
    pairs = pd.DataFrame(pairs); chains = pd.DataFrame(chains)
    words = sorted(set(pairs.a) | set(pairs.b) | set(chains.a) | set(chains.b) | set(chains.c))
    wdf = pd.DataFrame([word_record(w) for w in words])
    # branch-level dev/test split (deterministic; unseen branches at test time)
    import zlib
    wdf["split"] = wdf.branch.map(lambda b: "dev" if zlib.crc32(b.encode()) % 100 < CFG["DEV_PCT"] else "test")
    wsplit = dict(zip(wdf.word, wdf.split)); wbranch = dict(zip(wdf.word, wdf.branch))
    pairs["branch"] = pairs.a.map(wbranch); pairs["split"] = pairs.a.map(wsplit)
    chains["branch"] = chains.a.map(wbranch); chains["split"] = chains.a.map(wsplit)
    return wdf, pairs, chains
