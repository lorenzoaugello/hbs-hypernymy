def hashed_char_ngrams(ws, n=3, D=256):
    M = np.zeros((len(ws), D), np.float32)
    for i, w in enumerate(ws):
        s = f"^{w}$"
        for j in range(len(s) - n + 1): M[i, zlib.crc32(s[j:j + n].encode()) % D] += 1
    return M

def basic_conf():
    zf = wdf.zipf.values.astype(float); cols = [zf, zf ** 2, zf ** 3, wdf.word.str.len().values.astype(float)]
    if "nsenses" in wdf: cols.append(np.log1p(wdf.nsenses.values.astype(float)))
    return np.c_[tuple(cols)]
def rich_conf():
    mw = wdf.word.str.contains("_|-| ").values.astype(float)[:, None]; return np.c_[basic_conf(), mw, np.log1p(hashed_char_ngrams(list(wdf.word)))]
stdz = lambda C: StandardScaler().fit(C[dev_idx]).transform(C)
CONF_B, CONF_R = stdz(basic_conf()), stdz(rich_conf())

def where_v4(E, model, layers, k=128, n_boot=None):
    t = TARGETS2[:, 0]; tt = t[test_idx]; grp = wdf.branch.values[test_idx]; al = np.logspace(0, 5, 9)
    fitpred = lambda F: RidgeCV(alphas=al).fit(F[dev_idx], t[dev_idx]).predict(F[test_idx])
    pcb, pcr = fitpred(CONF_B), fitpred(CONF_R); P = {}
    for l in layers:
        Z = make_word_space(E[l], k); P[l] = (fitpred(Z), fitpred(np.c_[Z, CONF_B]), fitpred(np.c_[Z, CONF_R]))
    rho = lambda p, ix: spearmanr(p[ix], tt[ix])[0]; rows = []; l0 = layers[0]
    for l in layers:
        pz, pb, pr = P[l]
        fb = lambda ix: rho(pb, ix) - rho(pcb, ix); fr = lambda ix: rho(pr, ix) - rho(pcr, ix); fe = lambda ix: rho(pr, ix) - rho(P[l0][2], ix); allix = np.arange(len(tt)); r = {}
        for nm, fn in (("basic", fb), ("rich", fr), ("emergent", fe)): r[nm] = (fn(allix),) + boot_ci(grp, fn, n_boot)
        rows.append(dict(model=model, layer=l, rho_Z_only=rho(pz, allix), rho_conf_basic=rho(pcb, allix), rho_conf_rich=rho(pcr, allix),
                         d_basic=r["basic"][0], d_basic_lo=r["basic"][1], d_basic_hi=r["basic"][2], d_rich=r["rich"][0], d_rich_lo=r["rich"][1], d_rich_hi=r["rich"][2],
                         emergent=r["emergent"][0], em_lo=r["emergent"][1], em_hi=r["emergent"][2]))
    return pd.DataFrame(rows)

BEST_LAYER, WH4 = {}, []
for model_name in CFG["MODELS"]:
    E = get_reps(model_name, words); layers = layer_grid(E.shape[0]); BEST_LAYER[model_name] = int(select_layer_gen2(E, layers)); WH4.append(where_v4(E, model_name, layers)); del E
wh4 = pd.concat(WH4, ignore_index=True); wh4.to_csv(f"{CFG['OUT_DIR']}/v4_where_rich.csv", index=False)
for _, r in wh4.iterrows():
    if r.layer == BEST_LAYER[r.model]:
        log("whereV4", r.model, "delta rho over BASIC confounds", r.d_basic, r.d_basic_lo, r.d_basic_hi, layer=int(r.layer)); log("whereV4", r.model, "delta rho over RICH confounds", r.d_rich, r.d_rich_lo, r.d_rich_hi, layer=int(r.layer))
        log("whereV4", r.model, "emergent gain over layer 0 (rich)", r.emergent, r.em_lo, r.em_hi, layer=int(r.layer))
print("dev-selected layers:", BEST_LAYER); display(wh4.pivot_table(index="model", columns="layer", values="d_basic").round(3)); display(wh4.pivot_table(index="model", columns="layer", values="d_rich").round(3)); display(wh4.pivot_table(index="model", columns="layer", values="emergent").round(3))
