def group_mean_subspace(X, groups, rank, min_n=5):
    g = np.asarray(groups); means = np.vstack([X[g == c].mean(0) for c in pd.unique(g) if (g == c).sum() >= min_n]); means = means - means.mean(0)
    return np.linalg.svd(means, full_matrices=False)[2][:rank].T.astype(np.float32)                              # d x rank, orthonormal columns

def stratified_gen_direction(X, idx, min_n=6):
    t = TARGETS2[idx, 0]; lex = YLEX[idx]; zf = wdf.zipf.values[idx]; terc = np.digitize(zf, np.quantile(zf, [1 / 3, 2 / 3])); Xi = X[idx]; vec = np.zeros(X.shape[1]); wsum = 0.
    for key in set(zip(lex, terc)):
        m = (lex == key[0]) & (terc == key[1])
        if m.sum() < min_n: continue
        tm_, Xm = t[m], Xi[m]; med = np.median(tm_); hi, lo = tm_ > med, tm_ <= med
        if hi.sum() < 3 or lo.sum() < 3: continue
        w = min(hi.sum(), lo.sum()); vec += w * (Xm[hi].mean(0) - Xm[lo].mean(0)); wsum += w
    return -vec / max(wsum, 1e-9)                                                                                  # + = more general (high IC = specific)

def unit_vec(v): return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

def directions_v4(Elayer):
    X = fit_transform_space(Elayer, dev_idx, "center"); Xd = X[dev_idx]; t = TARGETS2[dev_idx, 0]; zf = wdf.zipf.values[dev_idx]
    a1, a3 = np.quantile(t, [.25, .75]); f1, f3 = np.quantile(zf, [.25, .75])
    gen_dm = -(Xd[t >= a3].mean(0) - Xd[t <= a1].mean(0)); freq = Xd[zf >= f3].mean(0) - Xd[zf <= f1].mean(0)
    top2 = pd.Series(YLEX[dev_idx]).value_counts().index[:2]; topic = Xd[YLEX[dev_idx] == top2[0]].mean(0) - Xd[YLEX[dev_idx] == top2[1]].mean(0)
    gen_s = stratified_gen_direction(X, dev_idx); fbins = np.digitize(zf, np.quantile(zf, np.linspace(.1, .9, 9)))
    B = np.hstack([group_mean_subspace(Xd, fbins, CFG["FREQ_BINS_RANK"]), group_mean_subspace(Xd, YLEX[dev_idx], CFG["LEX_MEANS_RANK"])]); B = np.linalg.qr(B)[0]
    perp = gen_s - B @ (B.T @ gen_s)
    return {"gen_dm": unit_vec(gen_dm), "gen_strat": unit_vec(gen_s), "gen_strat_perp": unit_vec(perp), "frequency": unit_vec(freq), "topic": unit_vec(topic)}, float(np.linalg.norm(Xd, axis=1).mean()), B

def random_dirs(d, n, seed=0): rng = np.random.default_rng(seed); return [unit_vec(v) for v in rng.normal(size=(n, d))]

def get_blocks(model):
    for path in ("transformer.h", "gpt_neox.layers", "model.layers"):
        obj = model
        try:
            for p in path.split("."): obj = getattr(obj, p)
            return obj
        except AttributeError: continue
    raise RuntimeError("unknown architecture: add its block list to get_blocks")

def word_span(tok, a):
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []; i0 = len(bos) + len(tok("Every", add_special_tokens=False).input_ids)
    return i0, i0 + len(tok(" " + a, add_special_tokens=False).input_ids)

def edit_hook(i0, i1, fn):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out; h[:, i0:i1, :] = fn(h[:, i0:i1, :]); return out
    return hook

class Edits:
    # context manager: apply [(layer, fn), ...] at the word's token positions (output of block layer-1)
    def __init__(self, model, tok, a, edits): self.model, self.tok, self.a, self.edits, self.handles = model, tok, a, edits, []
    def __enter__(self):
        if self.edits:
            i0, i1 = word_span(self.tok, self.a); blocks = get_blocks(self.model)
            self.handles = [blocks[L - 1].register_forward_hook(edit_hook(i0, i1, fn)) for L, fn in self.edits if 1 <= L <= len(blocks)]
        return self
    def __exit__(self, *exc):
        for h in self.handles: h.remove()

def cand_lp(model, tok, a, cands, edits=None):
    with Edits(model, tok, a, edits or []): return cont_logprob(model, tok, [f"Every {a} is a kind of"] * len(cands), list(cands))

def add_fn(vec, scale):
    import torch
    v = torch.tensor(vec * scale); return lambda h: h + v.to(device=h.device, dtype=h.dtype)

def project_out_fn(Q, mu):
    import torch
    Qt0, m0 = torch.tensor(Q.astype(np.float32)), torch.tensor(mu.astype(np.float32))
    def f(h): Qt, mt = Qt0.to(device=h.device, dtype=h.dtype), m0.to(device=h.device, dtype=h.dtype); return h - ((h - mt) @ Qt) @ Qt.T
    return f

def expected_k(lp): p = np.exp(lp - lp.max()); p /= p.sum(); return float((np.arange(1, len(lp) + 1) * p).sum())

def anchors_table(n):
    H = pairs[(pairs.rel == "hyper") & (pairs.split == "test")][["anchor", "k", "b"]].drop_duplicates(["anchor", "k"]).pivot(index="anchor", columns="k", values="b").dropna()
    return H[[1, 2, 3]].iloc[:n]

def steer_slopes(model, tok, H, LP0, layer, vec, unit, alphas):
    sk, sl = [], []
    for i, a in enumerate(H.index):
        lps = [LP0[i] if al == 0 else cand_lp(model, tok, a, H.loc[a].values, [(layer, add_fn(vec, al * unit))]) for al in alphas]
        sk.append(np.polyfit(alphas, [expected_k(l) for l in lps], 1)[0]); sl.append(np.polyfit(alphas, [l[2] - l[0] for l in lps], 1)[0])
    return np.array(sk), np.array(sl)

def disruption_kl(model, tok, anchors, layer, vec, unit, alpha=.3):
    import torch
    kls = []; dev = next(model.parameters()).device; bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    with torch.no_grad():
        for a in anchors:
            ids = torch.tensor([bos + tok(f"Every {a} is a kind of", add_special_tokens=False).input_ids], device=dev); p0 = model(input_ids=ids).logits[0, -1].float().log_softmax(-1)
            with Edits(model, tok, a, [(layer, add_fn(vec, alpha * unit))]): p1 = model(input_ids=ids).logits[0, -1].float().log_softmax(-1)
            kls.append(float((p0.exp() * (p0 - p1)).sum()))
    return float(np.mean(kls))

def run_steering(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"; tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    E = get_reps(model_name, words); layers = [l for l in layer_grid(E.shape[0]) if l >= 1]; H = anchors_table(CFG["N_STEER"]); alphas = CFG["STEER_ALPHAS"]
    LP0_for = lambda l: [cand_lp(model, tok, a, H.loc[a].values) for a in H.index]; rows = []
    for l in layers:
        dirs, unit, _ = directions_v4(E[l]); LP0 = LP0_for(l)
        for name, vec in {**dirs, **{f"random{i}": v for i, v in enumerate(random_dirs(E.shape[2], 5, seed=l))}}.items():
            sk, sl = steer_slopes(model, tok, H, LP0, l, vec, unit, alphas); lo, hi = boot_ci(np.arange(len(sk)), lambda ix: sl[ix].mean(), 500)
            rows.append(dict(model=model_name, layer=l, direction=name, slope_Ek=sk.mean(), slope_nats=sl.mean(), nats_lo=lo, nats_hi=hi, n=len(sk)))
    sweep = pd.DataFrame(rows); lb = max(BEST_LAYER[model_name], 1); dirs, unit, _ = directions_v4(E[lb]); LP0 = LP0_for(lb)
    null_k, null_l = zip(*[steer_slopes(model, tok, H, LP0, lb, v, unit, alphas) for v in random_dirs(E.shape[2], CFG["N_RAND_DIR"], seed=123)]); null = np.array([x.mean() for x in null_l]); res = []
    kl_r = np.mean([disruption_kl(model, tok, list(H.index[:50]), lb, v, unit) for v in random_dirs(E.shape[2], 3, seed=7)])
    for name, vec in dirs.items():
        sk, sl = steer_slopes(model, tok, H, LP0, lb, vec, unit, alphas); lo, hi = boot_ci(np.arange(len(sl)), lambda ix: sl[ix].mean(), 500)
        p = (1 + (np.abs(null) >= abs(sl.mean())).sum()) / (len(null) + 1); z = (sl.mean() - null.mean()) / (null.std() + 1e-9); kl = disruption_kl(model, tok, list(H.index[:50]), lb, vec, unit)
        res.append(dict(model=model_name, layer=lb, direction=name, slope_nats=sl.mean(), lo=lo, hi=hi, slope_Ek=sk.mean(), p_vs_random_null=p, z_vs_random_null=z, kl_disruption=kl, kl_random=kl_r))
        log("steer4", model_name, f"{name}: slope (nats per unit alpha)", sl.mean(), lo, hi, p=p, z=z, kl=kl)
    log("steer4", model_name, "random null: sd of slope", float(null.std())); del model; return sweep, pd.DataFrame(res)
