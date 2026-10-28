def dirs_topic2(Zc, q):
    m = np.isin(YLEX[dev_idx], LEX_CLASSES); lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage=.1).fit(Zc[dev_idx][m], YLEX[dev_idx][m]); return np.linalg.qr(lda.scalings_[:, :q])[0]
def dirs_hier2(Zc, q):
    alpha = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Zc[dev_idx], TARGETS2[dev_idx]).alpha_; Zt = Zc.copy(); out = []
    while sum(d.shape[1] for d in out) < q:
        coef = Ridge(alpha=alpha).fit(Zt[dev_idx], TARGETS2[dev_idx]).coef_; Qr = np.linalg.qr(coef.T)[0]; out.append(Qr); Zt = Zt - (Zt @ Qr) @ Qr.T
    return np.hstack(out)[:, :q]

def eval_tm2(Zc):
    dm = np.isin(YLEX[dev_idx], LEX_CLASSES); tm = np.isin(YLEX[test_idx], LEX_CLASSES); te = test_idx[tm]
    clf = LogisticRegression(C=.1, max_iter=300).fit(Zc[dev_idx][dm], YLEX[dev_idx][dm]); correct = (clf.predict(Zc[te]) == YLEX[te]).astype(float)
    rg = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(Zc[dev_idx], TARGETS2[dev_idx, 0]); return correct, rg.predict(Zc[te]), TARGETS2[te, 0], wdf.branch.values[te]

def metrics_ci(res, n_boot=None):
    c, p, t, g = res; a = (c.mean(),) + boot_ci(g, lambda ix: c[ix].mean(), n_boot); r = (spearmanr(p, t)[0],) + boot_ci(g, lambda ix: spearmanr(p[ix], t[ix])[0], n_boot); return a, r

def contrasts(res, n_boot=None):
    g = res["hier"][3]; allix = np.arange(len(g)); dt = lambda ix: res["hier"][0][ix].mean() - res["topic"][0][ix].mean()
    dg = lambda ix: spearmanr(res["topic"][1][ix], res["topic"][2][ix])[0] - spearmanr(res["hier"][1][ix], res["hier"][2][ix])[0]; di = lambda ix: min(dt(ix), dg(ix))
    return {n: (f(allix),) + boot_ci(g, f, n_boot) for n, f in (("D_topic_abs", dt), ("D_gen_abs", dg), ("DI_abs", di))}

def r50(df, kind, col, chance, ranks):
    d = df[df.eraser == kind].sort_values("rank"); a0 = d[d["rank"] == 0][col].iloc[0]; thr = chance + .5 * (a0 - chance)
    hit = d[d[col] <= thr]; return int(hit["rank"].iloc[0]) if len(hit) else None

CURVE4, SWEEP4, DECODE4, R50 = [], [], [], []
for model_name in CFG["MODELS"]:
    E = get_reps(model_name, words); layers = layer_grid(E.shape[0]); bl = BEST_LAYER[model_name]
    for l in sorted(set(layers) | {bl}):
        Z = make_word_space(E[l]); full = l == bl; R = max(CFG["ERASE_CURVE_RANKS"]) if full else CFG["SWEEP_RANK"]
        Qs = {"topic": iterative_dirs(Z, dirs_topic2, R, 4), "hier": iterative_dirs(Z, dirs_hier2, R, 4), "random": dirs_random(Z, R, seed=0)}
        ranks = CFG["ERASE_CURVE_RANKS"] if full else (0, CFG["SWEEP_RANK"]); res_by_rank = {}
        for r in ranks:
            res = {"none": eval_tm2(Z)}
            for kind, Q in Qs.items(): res[kind] = res["none"] if r == 0 else eval_tm2(Z - (Z @ Q[:, :r]) @ Q[:, :r].T)
            res_by_rank[r] = res
            if full:
                for kind in ("topic", "hier", "random"):
                    a, rho = metrics_ci(res[kind]); CURVE4.append(dict(model=model_name, layer=l, eraser=kind, rank=r, acc=a[0], acc_lo=a[1], acc_hi=a[2], rho=rho[0], rho_lo=rho[1], rho_hi=rho[2]))
        rr = CFG["SWEEP_RANK"]; cs = contrasts(res_by_rank[rr]); SWEEP4.append(dict(model=model_name, layer=l, rank=rr, **{f"{k}": v[0] for k, v in cs.items()}, **{f"{k}_lo": v[1] for k, v in cs.items()}, **{f"{k}_hi": v[2] for k, v in cs.items()}))
        if full:
            for k_, v in cs.items(): log("erase4", model_name, f"rank{rr}: {k_}", v[0], v[1], v[2], layer=l)
            # subspace-decoding matrix: decode both targets from r coordinates inside each subspace
            for r in CFG["DECODE_RANKS"]:
                res = {}
                for kind, Q in Qs.items():
                    res[kind] = eval_tm2(Z @ Q[:, :r])
                g = res["hier"][3]; dgen = lambda ix: spearmanr(res["hier"][1][ix], res["hier"][2][ix])[0] - spearmanr(res["topic"][1][ix], res["topic"][2][ix])[0]
                dtop = lambda ix: res["topic"][0][ix].mean() - res["hier"][0][ix].mean(); allix = np.arange(len(g))
                for kind in res:
                    a, rho = metrics_ci(res[kind]); DECODE4.append(dict(model=model_name, layer=l, rank=r, subspace=kind, topic_acc=a[0], gen_rho=rho[0]))
                lo1, hi1 = boot_ci(g, dgen); lo2, hi2 = boot_ci(g, dtop)
                log("decode4", model_name, f"rank{r}: generality decodability, hier subspace minus topic subspace (rho)", dgen(allix), lo1, hi1, layer=l); log("decode4", model_name, f"rank{r}: topic decodability, topic subspace minus hier subspace (acc)", dtop(allix), lo2, hi2, layer=l)
    del E
curve4 = pd.DataFrame(CURVE4); sweep4 = pd.DataFrame(SWEEP4); decode4 = pd.DataFrame(DECODE4)
for nm, d in (("v4_inlp_curve", curve4), ("v4_erasure_sweep", sweep4), ("v4_subspace_decoding", decode4)): d.to_csv(f"{CFG['OUT_DIR']}/{nm}.csv", index=False)
for m in CFG["MODELS"]:
    c = curve4[curve4.model == m]; chance = LEX_MAJ
    rt, rh, rrnd = r50(c, "topic", "acc", chance, None), r50(c.assign(rho=c.rho), "hier", "rho", 0., None), r50(c, "random", "rho", 0., None)
    R50.append(dict(model=m, r50_topic_eraser_on_topic=rt, r50_hier_eraser_on_generality=rh, r50_random_on_generality=rrnd)); log("erase4", m, "r50 hierarchy eraser on generality (None = >128)", float(rh) if rh is not None else np.nan)
display(pd.DataFrame(R50)); display(sweep4[["model", "layer", "D_topic_abs", "D_topic_abs_lo", "D_topic_abs_hi", "D_gen_abs", "D_gen_abs_lo", "D_gen_abs_hi", "DI_abs"]].round(3))
display(decode4.pivot_table(index=["model", "rank"], columns="subspace", values=["topic_acc", "gen_rho"]).round(3))
