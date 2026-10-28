def direction_report(T, oof, model, base_mode, n_boot=None):
    ok = ~np.isnan(oof["SA"]); T = T[ok].reset_index(drop=True); o = {k: v[ok] for k, v in oof.items()}; y, grp = T.y.values, T.anchor.values
    za, zb = T.a.map(W.zipf).values, T.b.map(W.zipf).values
    masks = {"all": np.ones(len(y), bool), "adversarial (parent rarer)": T.adv.values, "frequency-matched": np.abs(za - zb) < CFG["ADV_DELTA"]}; rows = []
    for mname, m in masks.items():
        if m.sum() < 50: continue
        for fam, s in o.items():
            ym, gm, sm = y[m], grp[m], s[m]; lo, hi = boot_ci(gm, lambda ix: auc(ym[ix], sm[ix]), n_boot)
            rows.append(dict(model=model, base_mode=base_mode, subset=mname, family=fam, auc=auc(ym, sm), lo=lo, hi=hi, n=int(m.sum())))
            if fam in ("A", "SA1", "diff", "SA"): log("dirv4", model, f"{base_mode} | {mname} | {fam}: AUC", auc(ym, sm), lo, hi)
    return pd.DataFrame(rows)

DIRREP = []
for model_name in CFG["MODELS"]:
    E = get_reps(model_name, words)
    for bm in ("cos", "cos_freq"):
        T, oof, chosen = run_cv3(E, "direction", bm, list(CFG["K3_GRID"]), layer_grid(E.shape[0]), ["base", "S", "A", "SA", "SA1", "diff", "concat", "a_only", "b_only"])
        DIRREP.append(direction_report(T, oof, model_name, bm)); print(model_name, bm, "(layer,k) per fold", chosen)
    del E
dirrep = pd.concat(DIRREP, ignore_index=True); dirrep.to_csv(f"{CFG['OUT_DIR']}/v4_direction_report.csv", index=False)
for bm in ("cos", "cos_freq"):
    print("\nbase mode:", bm)
    display(dirrep[dirrep.base_mode == bm].pivot_table(index=["subset", "model"], columns="family", values="auc").round(3)[["base", "S", "A", "SA1", "diff", "SA", "concat", "a_only", "b_only"]])
