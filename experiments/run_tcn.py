"""训练双向 TCN 序列模型（M3）：metric-aware 损失 + EMA + 按井验证 + 阈值寻优.

用法:
  python experiments/run_tcn.py --tag T1_smoke --epochs 4 --steps 30 --ch 64
  python experiments/run_tcn.py --tag T2_full --epochs 40 --steps 60 --ch 128
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import TCNNet, export_to_numpy, numpy_forward

LN10 = float(np.log10(np.e)) * np.log(10)  # = ln(10)
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")


def pointwise(c: str, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, float)
    t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS))
        t = np.log10(np.maximum(t, EPS))
    denom = TOLERANCE[c] * (np.abs(t) + EPS)
    return np.maximum(0.0, 1.0 - np.abs(p - t) / denom)


def decompose(c, pred, y, isdef):
    valid = ~np.isnan(y)
    d = valid & isdef
    r = valid & ~isdef
    return {
        "acc": pointwise(c, pred[valid], y[valid]).mean(),
        "acc_def": pointwise(c, pred[d], y[d]).mean() if d.sum() else float("nan"),
        "acc_real": pointwise(c, pred[r], y[r]).mean() if r.sum() else float("nan"),
    }


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().cpu().clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            vc = v.detach().cpu()
            if vc.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(vc.float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = vc.clone()

    def state(self):
        return {k: v.numpy() for k, v in self.shadow.items()}


def build_wells(wells, medians, stds):
    out = []
    for w in wells:
        X = make_features(w, context=False, priors=True)
        X = X.fillna(medians)
        X = (X - stds.replace(0, 1.0)) / stds.replace(0, 1.0)
        Y = make_targets(w)
        isd = default_mask(Y)
        item = {"X": X.to_numpy(dtype=np.float32), "logId": w["logId"].iloc[0]}
        for c in TARGETS:
            y = Y[c].to_numpy(dtype=np.float32)
            valid = ~np.isnan(y)
            t = np.log10(np.maximum(np.nan_to_num(y, nan=1.0), EPS)).astype(np.float32)
            item[f"{c}_valid"] = valid
            item[f"{c}_def"] = (isd[c].to_numpy() & valid).astype(np.float32)
            item[f"{c}_t"] = t
            item[f"{c}_y"] = y
        out.append(item)
    return out


def sample_batch(data, lengths, weights, L, batch):
    idx = np.random.choice(len(data), size=batch, p=weights)
    xs = []
    masks = {f"{c}_{k}": [] for c in TARGETS for k in ("valid", "def", "t")}
    for i in idx:
        w = data[i]
        T = int(lengths[i])
        s = 0 if T <= L else int(np.random.randint(0, T - L))
        xs.append(w["X"][s:s + L])
        for key in masks:
            masks[key].append(w[key][s:s + L])
    b = {"X": torch.from_numpy(np.stack(xs))}
    for key, arrs in masks.items():
        arr = np.stack(arrs)
        b[key] = torch.from_numpy(arr.astype(np.float32))
    return b


def metric_loss(pred_t, t, valid, real, c):
    """截断相对误差代理：带内线性、带外截断 + 小 L2 稳定."""
    d = (pred_t - t).abs()
    if c == "PERM":
        e = d / (TOLERANCE[c] * (t.abs() + EPS))
    else:
        e = LN10 * d / TOLERANCE[c]
    trunc = e.clamp(max=2.0)
    m = real
    if m.sum() == 0:
        return torch.tensor(0.0, device=pred_t.device)
    return trunc[m].mean() + 0.05 * ((pred_t - t) ** 2)[m].mean()


def validate(model, va_data, device, th=0.5, chunk=8192, overlap=300):
    model.eval()
    decs = []
    with torch.no_grad():
        for w in va_data:
            X = w["X"]
            T = X.shape[0]
            outs = {c: np.zeros((T, 2), dtype=np.float32) for c in TARGETS}
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                s2, e2 = max(0, s - overlap), min(T, e + overlap)
                xt = torch.from_numpy(X[s2:e2]).unsqueeze(0).to(device)
                o = model(xt)
                for c in TARGETS:
                    outs[c][s:e] = o[c].squeeze(0)[s - s2:e - s2].cpu().numpy()
            pred = {}
            for c in TARGETS:
                p = 1.0 / (1.0 + np.exp(-outs[c][:, 0]))
                reg_phys = np.power(10.0, outs[c][:, 1])
                pred[c] = np.where(p >= th, DEFAULT_VALUES[c], reg_phys)
            decs.append({c: decompose(c, pred[c], w[f"{c}_y"], w[f"{c}_def"] > 0.5) for c in TARGETS})
    model.train()
    mean_dec = {c: {k: float(np.nanmean([d[c][k] for d in decs])) for k in decs[0][c]} for c in TARGETS}
    total = sum(WEIGHTS[c] * mean_dec[c]["acc"] for c in TARGETS) * 100.0
    return total, mean_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./train")
    ap.add_argument("--tag", default="tcn")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, flush=True)

    wells = load_wells(args.data_dir, has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_wells = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va_wells = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df = pd.concat(tr_wells, ignore_index=True)
    base = make_features(tr_df, context=False, priors=True)
    medians, stds = base.median(), base.std()
    tr_data = build_wells(tr_wells, medians, stds)
    va_data = build_wells(va_wells, medians, stds)
    lengths = np.array([len(w["X"]) for w in tr_data], dtype=np.float64)
    weights = lengths / lengths.sum()
    n_features = tr_data[0]["X"].shape[1]

    dilations = (1, 2, 4, 8, 16)
    model = TCNNet(n_features, ch=args.ch, dilations=dilations, repeats=2).to(device)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    pos_weight = {}
    for c in TARGETS:
        nd = sum(float(w[f"{c}_def"].sum()) for w in tr_data)
        nr = sum(float((w[f"{c}_valid"].sum() - w[f"{c}_def"].sum())) for w in tr_data)
        pos_weight[c] = torch.tensor(max(0.1, min(10.0, nr / max(1.0, nd))), device=device)

    best = {"score": -1.0, "sd": None, "epoch": 0}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for _ in range(args.steps):
            b = sample_batch(tr_data, lengths, weights, args.window, args.batch)
            for k in list(b.keys()):
                b[k] = b[k].to(device)
            out = model(b["X"])
            loss = 0.0
            for c in TARGETS:
                valid = b[f"{c}_valid"] > 0.5
                defm = b[f"{c}_def"] > 0.5
                real = valid & ~defm
                logit = out[c][:, :, 0]
                if valid.any():
                    loss = loss + F.binary_cross_entropy_with_logits(logit[valid], b[f"{c}_def"][valid], pos_weight=pos_weight[c])
                loss = loss + metric_loss(out[c][:, :, 1], b[f"{c}_t"], valid, real, c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            run += float(loss.detach().cpu())
        sched.step()
        msg = f"epoch {epoch}/{args.epochs} loss={run / args.steps:.4f} lr={sched.get_last_lr()[0]:.2e}"
        if epoch % 2 == 0 or epoch == args.epochs:
            ema_model = TCNNet(n_features, ch=args.ch, dilations=dilations, repeats=2).to(device)
            ema_model.load_state_dict({k: torch.from_numpy(v).to(device) for k, v in ema.state().items()})
            score, dec = validate(ema_model, va_data, device)
            msg += f" | val@0.5={score:.3f} real=({dec['POR']['acc_real']:.3f},{dec['PERM']['acc_real']:.3f},{dec['SW']['acc_real']:.3f})"
            if score > best["score"]:
                best = {"score": score, "sd": ema.state(), "epoch": epoch}
        print(msg, flush=True)

    print(f"best val@0.5={best['score']:.3f} @epoch {best['epoch']} | {time.time() - t0:.0f}s")
    os.makedirs(RUNS_DIR, exist_ok=True)
    art_path = os.path.join(RUNS_DIR, f"tcn_{args.tag}.joblib")
    joblib.dump(
        {
            "sd": best["sd"],
            "dilations": list(dilations) * 2,
            "medians": medians,
            "stds": stds,
            "n_features": n_features,
            "ch": args.ch,
            "feature_cols": list(base.columns),
        },
        art_path,
    )
    print("saved", art_path)

    # 阈值寻优 + 分解报告（用 numpy 推理，验证导出链路）
    sd = best["sd"]
    dil_list = list(dilations) * 2
    cache = []
    for w in va_data:
        o = numpy_forward(w["X"], sd, dil_list)
        cache.append((w, o))
    grid = np.arange(0.05, 0.96, 0.05)
    th = {}
    for c in TARGETS:
        best_t, best_a = 0.5, -1.0
        for t in grid:
            accs = []
            for w, o in cache:
                p = 1.0 / (1.0 + np.exp(-o[c][:, 0]))
                pred = np.where(p >= t, DEFAULT_VALUES[c], np.power(10.0, o[c][:, 1]))
                y = w[f"{c}_y"]
                m = ~np.isnan(y)
                accs.append(pointwise(c, pred[m], y[m]).mean())
            a = float(np.mean(accs))
            if a > best_a:
                best_a, best_t = a, float(t)
        th[c] = best_t
    decs = []
    for w, o in cache:
        d = {}
        for c in TARGETS:
            p = 1.0 / (1.0 + np.exp(-o[c][:, 0]))
            pred = np.where(p >= th[c], DEFAULT_VALUES[c], np.power(10.0, o[c][:, 1]))
            d[c] = decompose(c, pred, w[f"{c}_y"], w[f"{c}_def"] > 0.5)
        decs.append(d)
    mean_dec = {c: {k: float(np.nanmean([d[c][k] for d in decs])) for k in decs[0][c]} for c in TARGETS}
    total = sum(WEIGHTS[c] * mean_dec[c]["acc"] for c in TARGETS) * 100.0
    print(
        f"[{args.tag}] TUNED total={total:.3f} th={ {k: round(v, 2) for k, v in th.items()} } | "
        + " ".join(f"{c}: acc={mean_dec[c]['acc']:.4f} real={mean_dec[c]['acc_real']:.4f} def={mean_dec[c]['acc_def']:.4f}" for c in TARGETS)
    )
    row = {
        "tag": args.tag, "mode": "screen", "folds": 1, "context": 0, "priors": 1,
        "ttransform": "all_log", "weights": 0, "mono": 0, "seg": "tcn", "seg_win": 0,
        "leaf": "", "min_leaf": "", "reg_iter": "", "reg_lr": "",
        "total": round(total, 3), "std": 0.0,
        **{f"{c}_acc": round(mean_dec[c]["acc"], 4) for c in TARGETS},
        **{f"{c}_real": round(mean_dec[c]["acc_real"], 4) for c in TARGETS},
        **{f"{c}_def": round(mean_dec[c]["acc_def"], 4) for c in TARGETS},
    }
    fieldnames = list(row.keys())
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, encoding="utf-8") as f:
            old_header = next(csv.reader(f), None)
        if old_header:
            fieldnames = list(old_header) + [k for k in row.keys() if k not in old_header]
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        if os.path.getsize(RESULTS_CSV) == 0:
            wcsv.writeheader()
        wcsv.writerow(row)
    print("appended to results.csv")


if __name__ == "__main__":
    main()
