"""阶段 1: 把成员预测 + 标签 + 掩码落成小 npz (避免反复载入 80 井)."""
import os, sys, gc, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_targets, default_mask

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
MEM = ["T15_seed42_150ep", "T16_seed43_150ep", "T16_seed44_150ep"]
TARGETS = ["POR", "PERM", "SW"]

wells = load_wells("./train", has_labels=True)
ids = [w["logId"].iloc[0] for w in wells]
va = pd.concat([w for w in wells if w["logId"].iloc[0] in set(ids[72:])], ignore_index=True)
del wells; gc.collect()
Y = make_targets(va)
DM = default_mask(Y)

cache = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
rt = cache["reg"]
store = {k: v for k, v in np.load(os.path.join(RUNS, "tcn_screen_outputs.npz")).items()}

out = {"well_of": va["logId"].to_numpy()}
for c in TARGETS:
    out[f"M__{c}"] = np.array([rt[c]] + [store[f"{t}__{c}__reg"] for t in MEM]).astype(np.float32)
    out[f"y__{c}"] = Y[c].to_numpy().astype(np.float32)
    out[f"res__{c}"] = (~DM[c].to_numpy()) & Y[c].notna().to_numpy()
np.savez_compressed(os.path.join(RUNS, "track_members.npz"), **out)
print("saved", os.path.join(RUNS, "track_members.npz"))
for c in TARGETS:
    r = out[f"res__{c}"]
    print(f"  {c}: N={len(out['y__'+c])}  储层段={int(r.sum()):,} ({r.mean():.4f})")
