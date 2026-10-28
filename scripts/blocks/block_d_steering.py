try:
    import torch; HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False
if HAVE_TORCH and CFG["CAUSAL_MODELS"]:
    SW4, ST4 = [], []
    for m in CFG["CAUSAL_MODELS"]: s1, s2 = run_steering(m); SW4.append(s1); ST4.append(s2)
    steer_sweep4 = pd.concat(SW4, ignore_index=True); steer_main4 = pd.concat(ST4, ignore_index=True)
    steer_sweep4.to_csv(f"{CFG['OUT_DIR']}/v4_steering_layer_sweep.csv", index=False); steer_main4.to_csv(f"{CFG['OUT_DIR']}/v4_steering_null.csv", index=False)
    display(steer_main4.round(4)); display(steer_sweep4[~steer_sweep4.direction.str.startswith("random")].pivot_table(index=["model", "direction"], columns="layer", values="slope_nats").round(3))
else:
    print("torch / causal models not available: skipping Block D")
