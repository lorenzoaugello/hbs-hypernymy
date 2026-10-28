ledger = pd.DataFrame(LEDGER); ledger.to_csv(f"{CFG['OUT_DIR']}/evidence_ledger_v4.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 100); display(ledger[ledger.exp.isin(["dirv4", "whereV4", "erase4", "decode4", "steer4", "ablate4"])].round(3))

def verdict():
    L = ledger; v = {}
    a = L[(L.exp == "dirv4") & (L.metric.str.startswith("cos | frequency-matched | A:"))]; v["D1 direction A on frequency-matched (cos base): CI lower > .55 (models)"] = f"{int((a.lo > .55).sum())} of {len(a)}  AUC={a.value.round(3).tolist()}"
    e = L[(L.exp == "whereV4") & (L.metric == "emergent gain over layer 0 (rich)")]; r = L[(L.exp == "whereV4") & (L.metric == "delta rho over RICH confounds")]
    v["D2 emergent gain over layer 0 (rich): CI>0 (models) / Δρ rich CI>0"] = f"{int((e.lo > 0).sum())} of {len(e)}  (Δρ rich CI>0: {int((r.lo > 0).sum())} of {len(r)}; emergent={e.value.round(3).tolist()})"
    s = L[(L.exp == "erase4") & (L.metric == "rank32: DI_abs")]; dm = L[(L.exp == "decode4") & (L.metric.str.contains("rank32"))]
    v["D3 DI_abs CI>0 at rank 32 (models) / decoding-matrix diagonal CI>0 (both contrasts, models)"] = f"{int((s.lo > 0).sum())} of {len(s)} / {int((dm.groupby('model').lo.min() > 0).sum())} of {dm.model.nunique()}"
    r50 = L[(L.exp == "erase4") & (L.metric.str.startswith("r50"))]; v["   r50 of the hierarchy eraser on generality (nan = >128)"] = r50.set_index("model").value.round(0).to_dict() if len(r50) else "n/a"
    st = L[(L.exp == "steer4") & (L.metric == "gen_strat: slope (nats per unit alpha)")]; sp = L[(L.exp == "steer4") & (L.metric == "gen_strat_perp: slope (nats per unit alpha)")]
    v["D4 steering: gen_strat p<.05 / gen_strat_perp p<.05 and slope>0 (models)"] = f"{int((st.p < .05).sum())} / {int(((sp.p < .05) & (sp.value > 0)).sum())} of {len(st)}" if len(st) else "not run"
    ab = L[(L.exp == "ablate4") & (L.metric == "forced-choice (4-way) accuracy: necessity (gen vs random)")]; ef = L[(L.exp == "ablate4") & (L.metric.str.startswith("edit effect"))]
    v["D5 ablation necessity CI>0 (models); edit-effect rows"] = f"{int((ab.lo > 0).sum())} of {len(ab)}; effects={ef.value.round(4).tolist()}" if len(ab) else "not run"
    return v
for k, val in verdict().items(): print(f"{k:96s} {val}")
