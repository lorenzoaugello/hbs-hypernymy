"""Shared runtime extracted from hierarchy_hbs_7_source.ipynb.

Run commands from the repository root so WORK_DIR resolves to ./work.
"""


# --- Original notebook cell 1 ---
import os, re, math, random, zlib, json, warnings, itertools, time
from functools import lru_cache
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x
try:
    from IPython.display import display
except Exception:
    display = print

CFG = dict(
    SEED=13, WORK_DIR="work", OUT_DIR="work/followup",
    MODELS=["gpt2", "EleutherAI/pythia-1b", "Qwen/Qwen2.5-1.5B", "bert-base-uncased"],   # same four models as before
    CAUSAL_MODELS=["gpt2", "EleutherAI/pythia-1b", "Qwen/Qwen2.5-1.5B"],                  # used by the behavioural test
    TEMPLATES=['The {w}.', 'Word: {w}', 'I am thinking about the {w}.'],                    # must match the original extraction
    PREP="center",
    N_LAYER_PTS=7,        # layers evaluated per model (evenly spaced, always incl. layer 0 and the last)
    K_GRID=(16, 64),      # probe input ranks the inner CV chooses from (capacity sweep in Exp 1b uses 16/64/256 explicitly)
    N_OUTER=5,            # outer branch-grouped folds
    N_BOOT=500,           # cluster-bootstrap replicates (use >=1000 for final tables)
    EQUIV_MARGIN=0.05,    # equivalence margin around AUC = .5
)
os.makedirs(CFG["OUT_DIR"], exist_ok=True); random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])
LEDGER = []   # every experiment appends rows here; the last cell turns them into the evidence table
def log(exp, model, metric, value, lo=np.nan, hi=np.nan, **extra):
    LEDGER.append(dict(exp=exp, model=model, metric=metric, value=value, lo=lo, hi=hi, **extra))
def layer_grid(n_total, n_pts=None):
    n_pts = n_pts or CFG["N_LAYER_PTS"]
    return sorted(set(np.linspace(0, n_total - 1, n_pts).round().astype(int).tolist()))


# --- Original notebook cell 3 ---
# ------------------------------------------------------------------ data produced by the original notebook
DATA_F = f"{CFG['WORK_DIR']}/hbs_dataset.pkl"
if not os.path.exists(DATA_F):
    raise FileNotFoundError("Run hierarchy_hbs_2.ipynb first (it writes work/hbs_dataset.pkl and the cached embeddings).")
wdf, pairs, chains = pd.read_pickle(DATA_F)
words = list(wdf.word); widx = {w: i for i, w in enumerate(words)}
dev_idx = np.where(wdf.split.values == "dev")[0]; test_idx = np.where(wdf.split.values == "test")[0]
print(len(words), "words |", wdf.branch.nunique(), "branches |", len(pairs), "pairs")

# ------------------------------------------------------------------ representation loader (cache first; extraction code as in the original notebook)
def _chunks(xs, n):
    for i in range(0, len(xs), n): yield xs[i:i + n]

def extract_word_reps(model_name, words, templates=None, batch_size=64, pooling="mean"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    templates = templates or CFG["TEMPLATES"]; device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"; acc = None
    with torch.no_grad():
        for tpl in templates:
            pre, post = tpl.split("{w}"); outs = []
            for batch in tqdm(list(_chunks(words, batch_size)), desc=f"{model_name.split('/')[-1]} | {tpl[:24]}", leave=False):
                texts = [pre + w + post for w in batch]
                enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True); offs = enc.pop("offset_mapping")
                ends = torch.tensor([len(pre) + len(w) for w in batch])[:, None]
                m = (offs[..., 0] < ends) & (offs[..., 1] > len(pre)) & (offs[..., 1] > offs[..., 0])
                if pooling == "last":
                    last = m.long().cumsum(1).argmax(1); m = torch.zeros_like(m); m[torch.arange(len(batch)), last] = True
                assert m.any(1).all(), "a word was not found in its prompt"
                hs = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True).hidden_states
                H = torch.stack(hs, 0); mm = m.to(device)[None, :, :, None].to(H.dtype)
                outs.append(((H * mm).sum(2) / mm.sum(2).clamp(min=1)).float().cpu())
            T = torch.cat(outs, 1).numpy(); acc = T if acc is None else acc + T
    del model
    return acc / len(templates)

def get_reps(model_name, words, tag=""):
    f = f"{CFG['WORK_DIR']}/emb_{model_name.replace('/', '_')}{tag}.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        if list(z["words"]) == list(words): return z["E"].astype(np.float32)
    E = extract_word_reps(model_name, list(words))
    np.savez_compressed(f, E=E.astype(np.float16), words=np.array(words, dtype=object)); return E


# --- Original notebook cell 4 ---
# ------------------------------------------------------------------ helpers copied from the original notebook (unchanged semantics)
HBS_MIN_MATCHED_ROWS = 200; HBS_MAX_SMD = .10; HBS_MIN_COMMON_SUPPORT = .50; HBS_COSINE_AUC_CEILING = .95

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan

def fit_transform_space(E, fit_idx, mode="center", k=256, floor=.05):
    E = np.asarray(E, np.float32); fit_idx = np.asarray(fit_idx, int)
    mu = E[fit_idx].mean(0, keepdims=True)
    if mode == "raw": return E.copy()
    X = E - mu
    if mode == "center": return X
    p = PCA(n_components=min(k, len(fit_idx) - 1, E.shape[1]), svd_solver="randomized", random_state=0).fit(X[fit_idx])
    s = np.sqrt(p.explained_variance_); return (p.transform(X) / (s + floor * s.max())).astype(np.float32)

def pair_stats(X, ia, ib):
    a, b = X[ia], X[ib]; na = np.linalg.norm(a, axis=1) + 1e-9; nb = np.linalg.norm(b, axis=1) + 1e-9
    dot = (a * b).sum(1); orth = (b * (a - b)).sum(1) / (nb * np.linalg.norm(a - b, axis=1) + 1e-9)
    return {"cos": dot / (na * nb), "ci_int": dot / nb ** 2, "ci_ext": dot / na ** 2,
            "orth_raw": orth, "orth_zero": -np.abs(orth), "lognorm": np.log(na) - np.log(nb)}

def smd(df, ycol, covs):
    out = {}
    for c0 in covs:
        t = df.loc[df[ycol] == 1, c0]; q = df.loc[df[ycol] == 0, c0]
        out[c0] = float((t.mean() - q.mean()) / np.sqrt((t.var() + q.var()) / 2 + 1e-12))
    return out

def propensity_match(df, ycol, covs, caliper=.2, seed=13):
    rng = np.random.default_rng(seed); Z = StandardScaler().fit_transform(df[covs].values); y = df[ycol].values.astype(int)
    ps = np.clip(LogisticRegression(max_iter=2000).fit(Z, y).predict_proba(Z)[:, 1], 1e-4, 1 - 1e-4)
    tr = np.where(y == 1)[0]; co = np.where(y == 0)[0]; lo = max(ps[tr].min(), ps[co].min()); hi = min(ps[tr].max(), ps[co].max())
    support = float(np.mean((ps >= lo) & (ps <= hi))) if hi >= lo else 0.
    lg = np.log(ps / (1 - ps)); cal = caliper * lg.std(); used = np.zeros(len(co), bool); keep = []
    for i in rng.permutation(tr):
        d = np.abs(lg[co] - lg[i]); d[used] = np.inf; j = int(d.argmin())
        if d[j] <= cal: used[j] = True; keep.extend([i, co[j]])
    return df.iloc[keep].copy(), support

def boot_ci(groups, stat_fn, n_boot=None, seed=13):
    # fast cluster bootstrap: resample whole clusters; stat_fn(row_indices) -> float
    n_boot = CFG["N_BOOT"] if n_boot is None else n_boot
    if n_boot <= 0: return (np.nan, np.nan)
    groups = np.asarray(groups); uniq, inv = np.unique(groups, return_inverse=True)
    if len(uniq) < 2: return (np.nan, np.nan)
    order = np.argsort(inv, kind="stable"); cuts = np.cumsum(np.bincount(inv))[:-1]
    members = np.split(order, cuts); rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        ix = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        v = stat_fn(ix)
        if np.isfinite(v): vals.append(v)
    return (float(np.quantile(vals, .025)), float(np.quantile(vals, .975))) if len(vals) > 10 else (np.nan, np.nan)

def matched_anc_uncle(X, widx_, wdf_, pairs_, split, seed=13, caliper=.2, covs=None, n_boot=0):
    P = pairs_[(pairs_.split == split) & (((pairs_.rel == "hyper") & (pairs_.k == 1)) | (pairs_.rel == "uncle"))].copy()
    W0 = wdf_.set_index("word")
    P = P[(P.a.map(W0.split) == split) & (P.b.map(W0.split) == split) & (P.a.map(W0.branch) == P.b.map(W0.branch))].copy()
    P["y"] = (P.rel == "hyper").astype(int); ia = P.a.map(widx_).values; ib = P.b.map(widx_).values
    for name, val in pair_stats(X, ia, ib).items(): P[name] = val
    P["zipf_a"] = P.a.map(W0.zipf); P["zipf_b"] = P.b.map(W0.zipf); P["logndesc_b"] = P.b.map(W0.logndesc)
    raw_auc = auc(P.y, P.cos); covs = list(covs or ["cos", "zipf_a", "zipf_b", "logndesc_b"])
    M, support = propensity_match(P, "y", covs, caliper=caliper, seed=seed)
    bal = smd(M, "y", covs) if len(M) else {}; max_smd = max([abs(v) for v in bal.values()], default=np.inf)
    valid = (len(M) >= HBS_MIN_MATCHED_ROWS and max_smd <= HBS_MAX_SMD and support >= HBS_MIN_COMMON_SUPPORT and raw_auc < HBS_COSINE_AUC_CEILING)
    out = {"anc_n_unmatched": len(P), "anc_n_matched": len(M), "anc_cos_auc_unmatched": raw_auc, "anc_common_support": support,
           "anc_max_smd_after": max_smd, "anc_valid": bool(valid)}
    for name in ("ci_int", "ci_ext", "orth_zero"):
        out[f"anc_auc_{name}"] = auc(M.y, M[name]) if len(M) else np.nan
        out[f"anc_auc_{name}_unmatched"] = auc(P.y, P[name])
        if n_boot > 0 and len(M):
            Mr = M.reset_index(drop=True); y_, s_ = Mr.y.values, Mr[name].values
            out[f"anc_auc_{name}_lo"], out[f"anc_auc_{name}_hi"] = boot_ci(Mr.anchor.values, lambda ix: auc(y_[ix], s_[ix]), n_boot)
    return out, M

def best_cos_layer(E):
    # layer whose raw cosine separates hyper vs uncle best on DEV pairs = the strictest similarity control for matching
    best = (-1, 0)
    for l in layer_grid(E.shape[0], 13):
        X = fit_transform_space(E[l], dev_idx, CFG["PREP"])
        st, _ = matched_anc_uncle(X, widx, wdf, pairs, "dev")
        if st["anc_cos_auc_unmatched"] > best[0]: best = (st["anc_cos_auc_unmatched"], l)
    return best[1]


# --- Original notebook cell 6 ---
W = wdf.set_index("word")

def fold_assign(wdf_, n_folds, seed=13):
    cnt = wdf_.groupby("branch").size().sort_values(ascending=False); load = np.zeros(n_folds); m = {}
    rng = np.random.default_rng(seed)
    for b, c in cnt.items():
        f = int(np.argmin(load + rng.random(n_folds) * 1e-6)); m[b] = f; load[f] += c
    return m
BRANCH_FOLD = fold_assign(wdf, CFG["N_OUTER"]); WFOLD = dict(zip(wdf.word, wdf.branch.map(BRANCH_FOLD)))
FOLD_WORDS = {f: [w for w, ff in WFOLD.items() if ff == f] for f in range(CFG["N_OUTER"])}

def build_pair_table(neg_rel, seed=13):
    pos = pairs[(pairs.rel == "hyper") & (pairs.k == 1)][["a", "b", "anchor"]].copy(); pos["y"] = 1
    if neg_rel == "direction":
        neg = pos.rename(columns={"a": "b", "b": "a"}).copy(); neg["y"] = 0                      # same words, reversed
    elif neg_rel == "rand_in_fold":
        rng = np.random.default_rng(seed); neg = pos.copy()
        neg["b"] = [rng.choice(FOLD_WORDS[WFOLD[a]]) for a in pos.a]; neg = neg[neg.a != neg.b]; neg["y"] = 0
    else:
        neg = pairs[pairs.rel == neg_rel][["a", "b", "anchor"]].copy(); neg["y"] = 0
    T = pd.concat([pos, neg], ignore_index=True); T["anchor"] = T.anchor.fillna(T.a)
    T["fold"] = T.a.map(WFOLD); T = T[(T.b.map(WFOLD) == T.fold)]                                 # both words in the same outer fold -> no word leakage
    T = T.drop_duplicates(["a", "b", "y"]).reset_index(drop=True); T["branch"] = T.a.map(W.branch)
    return T

def make_Z(X, fit_idx, k):
    sc = StandardScaler().fit(X[fit_idx]); Xs = sc.transform(X)
    kk = min(k, len(fit_idx) - 1, X.shape[1]); p = PCA(n_components=kk, random_state=0).fit(Xs[fit_idx])
    Z = p.transform(Xs); return (Z / (Z[fit_idx].std(0, keepdims=True) + 1e-9)).astype(np.float32)

def pair_feature_blocks(X, Z, df):
    ia = df.a.map(widx).values; ib = df.b.map(widx).values; st = pair_stats(X, ia, ib)
    base = np.c_[st["cos"], st["lognorm"], df.a.map(W.zipf).values, df.b.map(W.zipf).values, df.a.map(W.logndesc).values, df.b.map(W.logndesc).values]
    Za, Zb = Z[ia], Z[ib]
    return dict(base=base, sym=np.c_[base, Za * Zb, np.abs(Za - Zb)], asym=np.c_[base, Zb - Za], full=np.c_[base, Zb - Za, Za * Zb],
                a_only=np.c_[base, Za], b_only=np.c_[base, Zb])
FAMILIES = ["base", "sym", "asym", "full", "a_only", "b_only"]

N_BASE, BASE_BOOST = 6, 30.   # the 6 similarity/lexical covariates always come first; boosting them weakens their L2 penalty so that
                              # every family is at least as good as `base` (otherwise 128 noisy block features swamp 6 informative ones)
def _prep(Ftr, Fte):
    sc = StandardScaler().fit(Ftr); A = sc.transform(Ftr); B = sc.transform(Fte); A[:, :N_BASE] *= BASE_BOOST; B[:, :N_BASE] *= BASE_BOOST; return A, B

def fit_predict(Ftr, ytr, gtr, Fte):
    A, B = _prep(Ftr, Fte)
    cv = list(GroupKFold(min(3, len(np.unique(gtr)))).split(A, ytr, gtr))
    clf = LogisticRegressionCV(Cs=[3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1], cv=cv, scoring="roc_auc", max_iter=2000).fit(A, ytr)
    return clf.predict_proba(B)[:, 1]

def select_layer(E, layers, Ttr, fit_words, ks):
    # inner grouped CV over (layer, k): returns the best pair; k is the PCA rank fed to the pair features
    best = (-1, layers[0], ks[0]); g = Ttr.branch.values; y = Ttr.y.values
    cv = list(GroupKFold(min(3, len(np.unique(g)))).split(Ttr, y, g))
    for l in layers:
        X = fit_transform_space(E[l], fit_words, CFG["PREP"])
        for k in ks:
            Z = make_Z(X, fit_words, k); F = pair_feature_blocks(X, Z, Ttr)["full"]
            F, _ = _prep(F, F[:1]); oof = np.full(len(y), np.nan)
            for tr, va in cv: oof[va] = LogisticRegression(C=.003, max_iter=1000).fit(F[tr], y[tr]).predict_proba(F[va])[:, 1]
            a = auc(y, oof)
            if a > best[0]: best = (a, l, k)
    return best[1], best[2]

def run_pair_probe_cv(E, neg_rel, k=None, layers=None):
    ks = [k] if k else list(CFG["K_GRID"]); layers = layers or layer_grid(E.shape[0])
    T = build_pair_table(neg_rel); oof = {f: np.full(len(T), np.nan) for f in FAMILIES}; chosen = []
    for f in range(CFG["N_OUTER"]):
        te = np.where(T.fold.values == f)[0]; tr = np.where(T.fold.values != f)[0]; Ttr, Tte = T.iloc[tr], T.iloc[te]
        if len(np.unique(Ttr.y)) < 2 or len(np.unique(Tte.y)) < 2: continue
        fit_words = np.unique(np.r_[Ttr.a.map(widx).values, Ttr.b.map(widx).values])
        l, kk = select_layer(E, layers, Ttr, fit_words, ks); chosen.append((int(l), int(kk)))
        X = fit_transform_space(E[l], fit_words, CFG["PREP"]); Z = make_Z(X, fit_words, kk)
        Btr, Bte = pair_feature_blocks(X, Z, Ttr), pair_feature_blocks(X, Z, Tte); g = Ttr.branch.values
        for fam in FAMILIES: oof[fam][te] = fit_predict(Btr[fam], Ttr.y.values, g, Bte[fam])
    return T, oof, chosen

def summarize_pair_cv(T, oof, model, neg_rel, n_boot=None):
    ok = ~np.isnan(oof["full"]); T = T[ok].reset_index(drop=True); o = {k: v[ok] for k, v in oof.items()}
    y = T.y.values; grp = T.anchor.values; rows = []
    for fam, s in o.items():
        lo, hi = boot_ci(grp, lambda ix, s=s: auc(y[ix], s[ix]), n_boot); rows.append(dict(model=model, task=neg_rel, family=fam, auc=auc(y, s), lo=lo, hi=hi, n=len(y)))
    bases = ("base", "sym", "a_only", "b_only")
    gain = lambda ix: auc(y[ix], o["full"][ix]) - max(auc(y[ix], o[f][ix]) for f in bases)
    asym = lambda ix: auc(y[ix], o["asym"][ix]) - auc(y[ix], o["sym"][ix])
    allix = np.arange(len(y)); out = []
    for name, fn in (("gain_full_over_best_baseline", gain), ("asym_minus_sym", asym)):
        lo, hi = boot_ci(grp, fn, n_boot); out.append(dict(model=model, task=neg_rel, metric=name, value=fn(allix), lo=lo, hi=hi))
        log("1", model, f"{neg_rel}: {name}", fn(allix), lo, hi)
    return pd.DataFrame(rows), pd.DataFrame(out)




# --- Original notebook cell 7 ---
def load_taxonomy(wdf_):
    from nltk.corpus import wordnet as wn
    return {w: tuple(s.name() for s in wn.synset(sn).hypernym_paths()[0]) for w, sn in zip(wdf_.word, wdf_.synset)}

PATHS = load_taxonomy(wdf)
wdf["coarse"] = [PATHS[w][min(3, len(PATHS[w]) - 1)] for w in wdf.word]; W = wdf.set_index("word")

def make_targets(wdf_, dev_i, test_i):
    zf = wdf_.zipf.values; ic = wdf_.ic.values; dep = wdf_.depth.values.astype(float); out = np.zeros((len(wdf_), 2))
    r_ic = ic - np.polyval(np.polyfit(zf[dev_i], ic[dev_i], 1), zf); r_dp = dep - np.polyval(np.polyfit(zf[dev_i], dep[dev_i], 1), zf)
    for idx in (dev_i, test_i):                                                                 # demean within coarse category, per split
        g = wdf_.coarse.values[idx]
        for j, r in enumerate((r_ic, r_dp)):
            s = pd.Series(r[idx]); out[idx, j] = (s - s.groupby(g).transform("mean")).values
    return out
TARGETS = make_targets(wdf, dev_idx, test_idx)
vc_dev = wdf.coarse.iloc[dev_idx].value_counts(); vc_te = wdf.coarse.iloc[test_idx].value_counts()
TOPIC_CLASSES = [c for c in vc_dev.index if vc_dev[c] >= 20 and vc_te.get(c, 0) >= 5]
YTOP = wdf.coarse.values
print(len(TOPIC_CLASSES), "coarse categories;", "majority-class accuracy on test =",
      round(vc_te[TOPIC_CLASSES].max() / vc_te[TOPIC_CLASSES].sum(), 3))

def make_word_space(Elayer, k=256):
    X = fit_transform_space(Elayer, dev_idx, CFG["PREP"]); return make_Z(X, dev_idx, k)

def select_layer_gen(E, layers):
    # dev-only selection: out-of-branch Ridge Spearman on the generality target
    best = (-9, layers[0]); groups = wdf.branch.values[dev_idx]; cv = list(GroupKFold(min(4, len(np.unique(groups)))).split(dev_idx, groups=groups))
    for l in layers:
        Z = make_word_space(E[l], 128); oof = np.zeros(len(dev_idx))
        for tr, va in cv: oof[va] = RidgeCV(alphas=np.logspace(0, 4, 5)).fit(Z[dev_idx[tr]], TARGETS[dev_idx[tr], 0]).predict(Z[dev_idx[va]])
        r = spearmanr(oof, TARGETS[dev_idx, 0])[0]
        if r > best[0]: best = (r, l)
    return best[1]

def _orth_against(Q, Qall):
    if Qall is not None and Qall.shape[1]: Q = Q - Qall @ (Qall.T @ Q)
    return np.linalg.qr(Q)[0]

def dirs_topic(Zc, q):
    m = np.isin(YTOP[dev_idx], TOPIC_CLASSES); lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=.1).fit(Zc[dev_idx][m], YTOP[dev_idx][m])
    return np.linalg.qr(lda.scalings_[:, :q])[0]

def dirs_hier(Zc, q):
    alpha = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Zc[dev_idx], TARGETS[dev_idx]).alpha_; Zt = Zc.copy(); out = []
    while sum(d.shape[1] for d in out) < q:
        coef = Ridge(alpha=alpha).fit(Zt[dev_idx], TARGETS[dev_idx]).coef_; Qr = np.linalg.qr(coef.T)[0]; out.append(Qr); Zt = Zt - (Zt @ Qr) @ Qr.T
    return np.hstack(out)[:, :q]

def dirs_random(Zc, q, seed=0): return np.linalg.qr(np.random.default_rng(seed).normal(size=(Zc.shape[1], q)))[0]

def iterative_dirs(Z, fit_fn, total, step):
    Qall = np.zeros((Z.shape[1], 0)); Zc = Z.copy()
    while Qall.shape[1] < total:
        Q = _orth_against(fit_fn(Zc, step), Qall); Qall = np.hstack([Qall, Q]); Zc = Z - (Z @ Qall) @ Qall.T
    return Qall[:, :total]

def eval_space(Zc, n_boot=200):
    clf = LogisticRegression(C=.1, max_iter=300); dm = np.isin(YTOP[dev_idx], TOPIC_CLASSES); tm = np.isin(YTOP[test_idx], TOPIC_CLASSES)
    clf.fit(Zc[dev_idx][dm], YTOP[dev_idx][dm]); correct = (clf.predict(Zc[test_idx][tm]) == YTOP[test_idx][tm]).astype(float)
    grp = wdf.branch.values[test_idx][tm]; lo, hi = boot_ci(grp, lambda ix: correct[ix].mean(), n_boot)
    out = dict(topic_acc=correct.mean(), topic_lo=lo, topic_hi=hi)
    rg = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Zc[dev_idx], TARGETS[dev_idx]); pred = rg.predict(Zc[test_idx]); grp_t = wdf.branch.values[test_idx]
    for j, nm in enumerate(("gen", "depth")):
        rho = spearmanr(pred[:, j], TARGETS[test_idx, j])[0]; lo, hi = boot_ci(grp_t, lambda ix, j=j: spearmanr(pred[ix, j], TARGETS[test_idx][ix, j])[0], n_boot)
        out.update({f"{nm}_rho": rho, f"{nm}_lo": lo, f"{nm}_hi": hi})
    return out




# --- Original notebook cell 8 ---
def beh_prefixes(x):
    art = "an" if x[0] in "aeiou" else "a"
    return [f"Every {x} is a kind of", f"{art.capitalize()} {x} is a type of"]
N_TPL = 2

def cont_logprob(model, tok, prefixes, conts, bs=32):
    import torch
    dev = next(model.parameters()).device; out = np.zeros(len(prefixes)); bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    with torch.no_grad():
        for i in range(0, len(prefixes), bs):
            seqs, spans = [], []
            for p, c in zip(prefixes[i:i + bs], conts[i:i + bs]):
                ip = tok(p, add_special_tokens=False).input_ids; ic = tok(" " + c, add_special_tokens=False).input_ids
                s = len(bos) + len(ip); seqs.append(bos + ip + ic); spans.append((s, s + len(ic)))
            L = max(len(s) for s in seqs)
            ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=dev); att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=dev)
            logits = model(input_ids=ids, attention_mask=att).logits
            for j, (s, e) in enumerate(spans):
                lp = logits[j, s - 1:e - 1].float().log_softmax(-1); tgt = ids[j, s:e]; out[i + j] = lp.gather(-1, tgt[:, None]).sum().item()
    return out

def behavioural_scores(model, tok, df):
    a, b = df.a.tolist(), df.b.tolist(); acc = {k: 0. for k in ("fwd", "pmi_fwd", "pmi_rev")}
    for t in range(N_TPL):
        pf = lambda x: [beh_prefixes(w)[t] for w in x]; generic = [beh_prefixes("thing")[t]] * len(a)
        fwd = cont_logprob(model, tok, pf(a), b); rev = cont_logprob(model, tok, pf(b), a)
        acc["fwd"] += fwd / N_TPL; acc["pmi_fwd"] += (fwd - cont_logprob(model, tok, generic, b)) / N_TPL
        acc["pmi_rev"] += (rev - cont_logprob(model, tok, generic, a)) / N_TPL
    return pd.DataFrame(dict(a=a, b=b, fwd=acc["fwd"], pmi_fwd=acc["pmi_fwd"], pmi_rev=acc["pmi_rev"], asym=acc["pmi_fwd"] - acc["pmi_rev"]))

def run_behavioural(model_name, n_boot=None):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"; tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    E = get_reps(model_name, words); lc = best_cos_layer(E); X = fit_transform_space(E[lc], dev_idx, CFG["PREP"]); del E
    st, M = matched_anc_uncle(X, widx, wdf, pairs, "test", n_boot=0); M = M.reset_index(drop=True)
    allp = pd.concat([M[["a", "b"]], pairs[(pairs.split == "test") & (pairs.rel == "hyper")][["a", "b"]]]).drop_duplicates().reset_index(drop=True)
    S = behavioural_scores(model, tok, allp); del model
    Mx = M.merge(S, on=["a", "b"], how="left"); rows = []
    for sc in ("fwd", "pmi_fwd", "asym"):
        y, s = Mx.y.values, Mx[sc].values; lo, hi = boot_ci(Mx.anchor.values, lambda ix: auc(y[ix], s[ix]), n_boot)
        rows.append(dict(model=model_name, test="ancestor vs uncle (cosine-matched)", score=sc, value=auc(y, s), lo=lo, hi=hi, n=len(y), cos_layer=lc, cos_auc_after_matching=auc(y, Mx.cos.values)))
        log("4", model_name, f"matched AUC ({sc})", auc(y, s), lo, hi)
    H = pairs[(pairs.split == "test") & (pairs.rel == "hyper")].merge(S, on=["a", "b"], how="inner"); H["za"] = H.a.map(W.zipf); H["zb"] = H.b.map(W.zipf)
    H["hit"] = (H.pmi_fwd > H.pmi_rev) + .5 * (H.pmi_fwd == H.pmi_rev); H["freq_hit"] = (H.zb > H.za) + .5 * (H.zb == H.za); H["adv"] = H.zb < H.za - .25
    for name, sub in (("all k", H), ("k=1", H[H.k == 1]), ("k=2", H[H.k == 2]), ("k=3", H[H.k == 3]), ("adversarial (b rarer than a)", H[H.adv])):
        if not len(sub): continue
        h = sub.hit.values; lo, hi = boot_ci(sub.anchor.values, lambda ix: h[ix].mean(), n_boot)
        rows.append(dict(model=model_name, test=f"direction accuracy: {name}", score="pmi_fwd>pmi_rev", value=h.mean(), lo=lo, hi=hi, n=len(sub), freq_baseline=sub.freq_hit.mean()))
        log("4", model_name, f"direction acc {name}", h.mean(), lo, hi, freq_baseline=sub.freq_hit.mean())
    return pd.DataFrame(rows)




# --- Original notebook cell 9 ---
CFG.update(K3=16,                    # PCA rank of pair features in Block 2 (second-order feature count grows ~k^2)
           K3_GRID=(8, 16),          # ranks the inner CV may choose from in the nested run
           TASKS3=[("uncle", "cos_freq"), ("uncle", "cos_freq_wn"), ("direction", "cos"), ("direction", "cos_freq")],
           ADV_DELTA=.25,            # parent rarer than child by this much zipf = frequency-adversarial direction pair
           EQUIV_AUC=.01,            # equivalence margin on AUC differences
           ERASE_RANKS=(16, 32), N_RAND_ERASE=3,
           N_STEER=150, STEER_ALPHAS=(-.3, 0., .3), N_RAND_DIR=60, N_ABLATE_SEEDS=5, ABLATE_RANK=16)
FAMS3 = ["base", "S", "A", "SA", "SA1", "SA2", "diff", "concat", "a_only", "b_only"]
BASE_MODES = ("cos", "cos_freq", "cos_freq_wn")


# --- Original notebook cell 10 ---
def add_adv(T, delta=None):
    delta = CFG["ADV_DELTA"] if delta is None else delta
    za, zb = T.a.map(W.zipf).values, T.b.map(W.zipf).values
    T["adv"] = ((T.y.values == 1) & (zb < za - delta)) | ((T.y.values == 0) & (za < zb - delta)); return T          # true parent rarer than child, in either orientation

def oof_lr(F, T, Cs=(1e-2, 1e-1, 1., 10.)):
    oof = np.full(len(T), np.nan); y = T.y.values
    for f in range(CFG["N_OUTER"]):
        te = T.fold.values == f; tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2: continue
        sc = StandardScaler().fit(F[tr]); A, B = sc.transform(F[tr]), sc.transform(F[te]); g = T.branch.values[tr]
        cv = list(GroupKFold(min(3, len(np.unique(g)))).split(A, y[tr], g))
        oof[te] = LogisticRegressionCV(Cs=list(Cs), cv=cv, scoring="roc_auc", max_iter=2000).fit(A, y[tr]).predict_proba(B)[:, 1]
    return oof

def cos_layers(E, T, layers):
    ia, ib = T.a.map(widx).values, T.b.map(widx).values; C, N = [], []
    for l in layers:
        X = fit_transform_space(E[l], dev_idx, CFG["PREP"]); st = pair_stats(X, ia, ib); C.append(st["cos"]); N.append(st["lognorm"])
    return np.c_[tuple(C)], np.c_[tuple(N)]

def ladder(E, model, task, n_boot=None):
    T = add_adv(build_pair_table(task)); layers = layer_grid(E.shape[0]); C, N = cos_layers(E, T, layers); y = T.y.values; grp = T.anchor.values
    dev = (T.a.map(W.split) == "dev").values
    j = int(np.nanargmax([auc(y[dev], C[dev, i]) if task != "direction" else .5 for i in range(len(layers))]))
    za, zb = T.a.map(W.zipf).values, T.b.map(W.zipf).values; la, lb = T.a.map(W.logndesc).values, T.b.map(W.logndesc).values
    rungs = {"R1 cos (best layer)": C[:, [j]], "R2 cos (all layers)": C, "R3 + frequency + norm": np.c_[C, za, zb, N[:, j]], "R4 + WordNet counts": np.c_[C, za, zb, N[:, j], la, lb]}
    rows, oofs = [], {}
    for name, F in rungs.items():
        o = oof_lr(F, T); oofs[name] = o; ok = ~np.isnan(o); lo, hi = boot_ci(grp[ok], lambda ix, o=o[ok], yy=y[ok]: auc(yy[ix], o[ix]), n_boot)
        r = dict(model=model, task=task, rung=name, auc=auc(y[ok], o[ok]), lo=lo, hi=hi, n=int(ok.sum()))
        if task == "direction":
            m = ok & T.adv.values; r["auc_adversarial"] = auc(y[m], o[m])
        rows.append(r); log("ladder", model, f"{task}: {name}", r["auc"], lo, hi)
    names = list(rungs); ok = np.all([~np.isnan(oofs[n]) for n in names], axis=0)
    for a_, b_ in zip(names[:-1], names[1:]):
        fn = lambda ix, a_=a_, b_=b_: auc(y[ok][ix], oofs[b_][ok][ix]) - auc(y[ok][ix], oofs[a_][ok][ix]); lo, hi = boot_ci(grp[ok], fn, n_boot)
        log("ladder", model, f"{task}: step {a_[:2]}->{b_[:2]}", fn(np.arange(ok.sum())), lo, hi)
    return pd.DataFrame(rows)




# --- Original notebook cell 11 ---
def blocks3(X, Z, df, base_mode):
    ia, ib = df.a.map(widx).values, df.b.map(widx).values; st = pair_stats(X, ia, ib)
    za, zb = df.a.map(W.zipf).values, df.b.map(W.zipf).values; la, lb = df.a.map(W.logndesc).values, df.b.map(W.logndesc).values
    base = {"cos": np.c_[st["cos"]], "cos_freq": np.c_[st["cos"], st["lognorm"], za, zb], "cos_freq_wn": np.c_[st["cos"], st["lognorm"], za, zb, la, lb]}[base_mode]
    Za, Zb = Z[ia], Z[ib]; k = Z.shape[1]; iu = np.triu_indices(k, 0); ju = np.triu_indices(k, 1)
    S1, A1 = Za + Zb, Zb - Za
    S2 = Za[:, iu[0]] * Zb[:, iu[1]] + Za[:, iu[1]] * Zb[:, iu[0]]; A2 = Za[:, ju[0]] * Zb[:, ju[1]] - Za[:, ju[1]] * Zb[:, ju[0]]
    Qp = Za[:, iu[0]] * Za[:, iu[1]] + Zb[:, iu[0]] * Zb[:, iu[1]]; Qm = Zb[:, iu[0]] * Zb[:, iu[1]] - Za[:, iu[0]] * Za[:, iu[1]]
    Ssym, Aant = np.c_[S1, S2, Qp], np.c_[A1, A2, Qm]
    fam = dict(base=base, S=np.c_[base, Ssym], A=np.c_[base, Aant], SA=np.c_[base, Ssym, Aant], SA1=np.c_[base, Ssym, A1], SA2=np.c_[base, Ssym, A2, Qm],
               diff=np.c_[base, A1], concat=np.c_[base, Za, Zb], a_only=np.c_[base, Za], b_only=np.c_[base, Zb])
    return fam, base.shape[1]

def _prep2(Ftr, Fte, nb, boost=30.):
    sc = StandardScaler().fit(Ftr); A, B = sc.transform(Ftr), sc.transform(Fte); A[:, :nb] *= boost; B[:, :nb] *= boost; return A, B

def fit_predict3(Ftr, ytr, gtr, Fte, Cs, nb):
    A, B = _prep2(Ftr, Fte, nb); cv = list(GroupKFold(min(3, len(np.unique(gtr)))).split(A, ytr, gtr))
    return LogisticRegressionCV(Cs=list(Cs), cv=cv, scoring="roc_auc", max_iter=1500).fit(A, ytr).predict_proba(B)[:, 1]

def select_layer3(E, layers, Ttr, fit_words, ks, base_mode, fam="SA"):
    best = (-1, layers[0], ks[0]); g, y = Ttr.branch.values, Ttr.y.values; cv = list(GroupKFold(min(3, len(np.unique(g)))).split(Ttr, y, g))
    for l in layers:
        X = fit_transform_space(E[l], fit_words, CFG["PREP"])
        for k in ks:
            Z = make_Z(X, fit_words, k); fb, nb = blocks3(X, Z, Ttr, base_mode); F, _ = _prep2(fb[fam], fb[fam][:1], nb); oof = np.full(len(y), np.nan)
            for tr, va in cv: oof[va] = LogisticRegression(C=.003, max_iter=1000).fit(F[tr], y[tr]).predict_proba(F[va])[:, 1]
            a = auc(y, oof)
            if a > best[0]: best = (a, l, k)
    return best[1], best[2]

def run_cv3(E, task, base_mode, ks, layers, fams, Cs=(1e-4, 1e-3, 1e-2, 1e-1)):
    T = build_pair_table(task); T = add_adv(T) if task == "direction" else T; oof = {f: np.full(len(T), np.nan) for f in fams}; chosen = []
    for f in range(CFG["N_OUTER"]):
        te = np.where(T.fold.values == f)[0]; tr = np.where(T.fold.values != f)[0]; Ttr, Tte = T.iloc[tr], T.iloc[te]
        if len(np.unique(Ttr.y)) < 2 or len(np.unique(Tte.y)) < 2: continue
        fit_words = np.unique(np.r_[Ttr.a.map(widx).values, Ttr.b.map(widx).values])
        l, k = select_layer3(E, layers, Ttr, fit_words, ks, base_mode) if (len(layers) > 1 or len(ks) > 1) else (layers[0], ks[0]); chosen.append((int(l), int(k)))
        X = fit_transform_space(E[l], fit_words, CFG["PREP"]); Z = make_Z(X, fit_words, k); (Btr, nb), (Bte, _) = blocks3(X, Z, Ttr, base_mode), blocks3(X, Z, Tte, base_mode)
        for fam in fams: oof[fam][te] = fit_predict3(Btr[fam], Ttr.y.values, Ttr.branch.values, Bte[fam], Cs, nb)
    return T, oof, chosen

def summ3(T, oof, model, task, base_mode, n_boot=None):
    ok = ~np.isnan(oof[next(iter(oof))]); T = T[ok].reset_index(drop=True); o = {k: v[ok] for k, v in oof.items()}; y, grp = T.y.values, T.anchor.values; rows = []
    adv = T.adv.values if "adv" in T else np.zeros(len(T), bool)
    for fam, s in o.items():
        lo, hi = boot_ci(grp, lambda ix, s=s: auc(y[ix], s[ix]), n_boot); r = dict(model=model, task=task, base_mode=base_mode, family=fam, auc=auc(y, s), lo=lo, hi=hi, n=len(y))
        if adv.any(): r["auc_adv"] = auc(y[adv], s[adv])
        rows.append(r)
    others = [f for f in ("base", "diff", "concat", "a_only", "b_only") if f in o]
    fns = {"U_A (SA - S)": ("SA", "S"), "U_A1 first-order axis (SA1 - S)": ("SA1", "S"), "U_A2 rotations+quadratic (SA2 - S)": ("SA2", "S"), "U_S (SA - A)": ("SA", "A")}
    der = []
    for name, (p, q) in fns.items():
        fn = lambda ix, p=p, q=q: auc(y[ix], o[p][ix]) - auc(y[ix], o[q][ix]); lo, hi = boot_ci(grp, fn, n_boot); v = fn(np.arange(len(y)))
        der.append(dict(model=model, task=task, base_mode=base_mode, metric=name, value=v, lo=lo, hi=hi, equivalent_to_zero=bool(np.isfinite(lo) and lo > -CFG["EQUIV_AUC"] and hi < CFG["EQUIV_AUC"])))
        log("SA3", model, f"{task}/{base_mode}: {name}", v, lo, hi, equivalent=der[-1]["equivalent_to_zero"])
    fn = lambda ix: auc(y[ix], o["SA"][ix]) - max(auc(y[ix], o[f][ix]) for f in others); lo, hi = boot_ci(grp, fn, n_boot)
    der.append(dict(model=model, task=task, base_mode=base_mode, metric="SA over best existing", value=fn(np.arange(len(y))), lo=lo, hi=hi)); log("SA3", model, f"{task}/{base_mode}: SA over best existing", der[-1]["value"], lo, hi)
    if adv.sum() > 50:
        ya, ga = y[adv], grp[adv]; oa = {k: v[adv] for k, v in o.items()}; fn = lambda ix: auc(ya[ix], oa["SA"][ix]) - auc(ya[ix], oa["S"][ix]); lo, hi = boot_ci(ga, fn, n_boot)
        der.append(dict(model=model, task=task, base_mode=base_mode, metric="U_A on frequency-adversarial pairs", value=fn(np.arange(len(ya))), lo=lo, hi=hi)); log("SA3", model, f"{task}/{base_mode}: U_A adversarial", der[-1]["value"], lo, hi)
    return pd.DataFrame(rows), pd.DataFrame(der)




# --- Original notebook cell 12 ---
CFG.update(ERASE_CURVE_RANKS=(0, 4, 8, 16, 32, 64, 128), DECODE_RANKS=(8, 16, 32), SWEEP_RANK=32,
           N_STEER=200, STEER_ALPHAS=(-.3, -.15, 0., .15, .3), N_RAND_DIR=100,
           ABL_SPAN=4, N_ABL_RAND=10, N_FC=300, ABL_TOPIC_RANK=5, FREQ_BINS_RANK=5, LEX_MEANS_RANK=8)


# --- Original notebook cell 14 ---
def load_lexnames(wdf_):
    from nltk.corpus import wordnet as wn
    return [wn.synset(s).lexname() for s in wdf_.synset]

wdf["lex"] = load_lexnames(wdf); W = wdf.set_index("word"); YLEX = wdf.lex.values

def make_targets_by(col):
    zf = wdf.zipf.values; ic = wdf.ic.values; dep = wdf.depth.values.astype(float); out = np.zeros((len(wdf), 2))
    r_ic = ic - np.polyval(np.polyfit(zf[dev_idx], ic[dev_idx], 1), zf); r_dp = dep - np.polyval(np.polyfit(zf[dev_idx], dep[dev_idx], 1), zf)
    for idx in (dev_idx, test_idx):
        g = wdf[col].values[idx]
        for j, r in enumerate((r_ic, r_dp)): s = pd.Series(r[idx]); out[idx, j] = (s - s.groupby(g).transform("mean")).values
    return out
TARGETS2 = make_targets_by("lex")
vd, vt = wdf.lex.iloc[dev_idx].value_counts(), wdf.lex.iloc[test_idx].value_counts(); LEX_CLASSES = [c for c in vd.index if vd[c] >= 20 and vt.get(c, 0) >= 5]
tm_mask = np.isin(YLEX[test_idx], LEX_CLASSES); LEX_MAJ = pd.Series(YLEX[test_idx][tm_mask]).value_counts(normalize=True).iloc[0]
print(len(LEX_CLASSES), "lexname classes; majority-class accuracy on test =", round(LEX_MAJ, 3))

def select_layer_gen2(E, layers):
    best = (-9, layers[0]); groups = wdf.branch.values[dev_idx]; cv = list(GroupKFold(min(4, len(np.unique(groups)))).split(dev_idx, groups=groups))
    for l in layers:
        Z = make_word_space(E[l], 128); oof = np.zeros(len(dev_idx))
        for tr, va in cv: oof[va] = RidgeCV(alphas=np.logspace(0, 4, 5)).fit(Z[dev_idx[tr]], TARGETS2[dev_idx[tr], 0]).predict(Z[dev_idx[va]])
        r = spearmanr(oof, TARGETS2[dev_idx, 0])[0]
        if r > best[0]: best = (r, l)
    return best[1]
