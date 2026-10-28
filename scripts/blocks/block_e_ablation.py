def forced_choice_table(n, seed=0):
    rng = np.random.default_rng(seed); te_words = [w for w in words if W.split[w] == "test"]; P = pairs[pairs.split == "test"]
    par = P[(P.rel == "hyper") & (P.k == 1)].drop_duplicates("anchor").set_index("anchor").b; unc = P[P.rel == "uncle"].drop_duplicates("anchor").set_index("anchor").b; coh = P[P.rel == "cohyp"].drop_duplicates("anchor").set_index("anchor").b
    anchors = [a for a in par.index if a in unc.index and a in coh.index]; rng.shuffle(anchors); rows = []
    for a in anchors[:n]:
        dist = rng.choice(te_words)
        while W.branch[dist] == W.branch[a] or dist in (par[a], unc[a], coh[a]): dist = rng.choice(te_words)
        rows.append(dict(a=a, cands=[par[a], unc[a], coh[a], dist]))
    return pd.DataFrame(rows)

def run_ablation(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"; tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    E = get_reps(model_name, words); nL = E.shape[0] - 1; l0 = max(BEST_LAYER[model_name], 1); span = CFG["ABL_SPAN"]; Ls = [L for L in range(l0, l0 + span) if L <= nL]
    mu, QG, QT, QF = {}, {}, {}, {}
    for L in Ls:
        dirs, unit, Bperp = directions_v4(E[L]); X = fit_transform_space(E[L], dev_idx, "center"); Xd = X[dev_idx]; mu[L] = E[L][dev_idx].mean(0); zf = wdf.zipf.values[dev_idx]
        QG[L] = dirs["gen_strat"][:, None]; QT[L] = group_mean_subspace(Xd, YLEX[dev_idx], CFG["ABL_TOPIC_RANK"]); QF[L] = group_mean_subspace(Xd, np.digitize(zf, np.quantile(zf, np.linspace(.1, .9, 9))), CFG["FREQ_BINS_RANK"])
    d = E.shape[2]; conds = {"none": [], "gen": [(L, project_out_fn(QG[L], mu[L])) for L in Ls], "topic": [(L, project_out_fn(QT[L], mu[L])) for L in Ls], "frequency": [(L, project_out_fn(QF[L], mu[L])) for L in Ls]}
    for s in range(CFG["N_ABL_RAND"]):
        rg = np.random.default_rng(s)
        conds[f"rand1_{s}"] = [(L, project_out_fn(np.linalg.qr(rg.normal(size=(d, 1)))[0], mu[L])) for L in Ls]
        conds[f"randR_{s}"] = [(L, project_out_fn(np.linalg.qr(rg.normal(size=(d, CFG["ABL_TOPIC_RANK"])))[0], mu[L])) for L in Ls]
    FC = forced_choice_table(CFG["N_FC"]); allc = sorted({c for cs in FC.cands for c in cs}); prior = dict(zip(allc, cont_logprob(model, tok, ["Every thing is a kind of"] * len(allc), allc)))
    acc4, accpu, dlp = {}, {}, {}; base_parent = np.zeros(len(FC))
    for cname, edits in conds.items():
        c4, cpu, dp = np.zeros(len(FC)), np.zeros(len(FC)), np.zeros(len(FC))
        for i, (a, cands) in enumerate(zip(FC.a, FC.cands)):
            lp = cand_lp(model, tok, a, cands, edits); pm = lp - np.array([prior[c] for c in cands]); c4[i] = float(pm.argmax() == 0); cpu[i] = float(pm[0] > pm[1])
            if cname == "none": base_parent[i] = lp[0]
            dp[i] = abs(lp[0] - base_parent[i])
        acc4[cname], accpu[cname], dlp[cname] = c4, cpu, dp
    grp = np.arange(len(FC)); rows = []
    for metric, M_ in (("forced-choice (4-way) accuracy", acc4), ("parent-vs-uncle accuracy", accpu)):
        r1 = [k for k in M_ if k.startswith("rand1_")]; rR = [k for k in M_ if k.startswith("randR_")]
        for gname, g_, rnd in (("gen", "gen", r1), ("topic", "topic", rR), ("frequency", "frequency", rR)):
            f = lambda ix: (M_["none"][ix].mean() - M_[g_][ix].mean()) - (M_["none"][ix].mean() - np.mean([M_[k][ix].mean() for k in rnd]))
            lo, hi = boot_ci(grp, f, 500); rows.append(dict(model=model_name, layer_start=l0, span=len(Ls), metric=metric, ablated=gname, acc_none=M_["none"].mean(), acc_ablated=M_[g_].mean(), acc_random=np.mean([M_[k].mean() for k in rnd]), necessity=f(grp), lo=lo, hi=hi))
            if gname == "gen": log("ablate4", model_name, f"{metric}: necessity (gen vs random)", f(grp), lo, hi)
    eff = {k: float(v.mean()) for k, v in dlp.items() if k in ("gen", "topic", "frequency")}; eff["random1_mean"] = float(np.mean([dlp[k].mean() for k in dlp if k.startswith("rand1_")]))
    log("ablate4", model_name, "edit effect: mean |dlogP(parent)| gen", eff["gen"]); log("ablate4", model_name, "edit effect: mean |dlogP(parent)| random1", eff["random1_mean"]); del model
    return pd.DataFrame(rows), eff

if HAVE_TORCH and CFG["CAUSAL_MODELS"]:
    AB4, EFF4 = [], {}
    for m in CFG["CAUSAL_MODELS"]: r, e = run_ablation(m); AB4.append(r); EFF4[m] = e
    ablate4 = pd.concat(AB4, ignore_index=True); ablate4.to_csv(f"{CFG['OUT_DIR']}/v4_ablation.csv", index=False); display(ablate4.round(4)); print("edit effects (mean |dlogP(parent)|):", EFF4)
else:
    print("torch / causal models not available: skipping Block E")
