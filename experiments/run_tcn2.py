"""M4 pass2：输入 = 测井特征 + pass1(T3) 因果平滑预测通道；scheduled sampling 训练.

训练通道: 以概率 p 用真实值因果平滑, 否则用 pass1 井内(in-sample)预测因果平滑; p 1->0 退火.
推理通道: pass1 预测因果平滑（验证井上 pass1 为 out-of-sample, 与测试同分布）.
"""
from __future__ import annotations

import argparse
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
from solution.tcn_model import TCNNet, numpy_forward
from experiments.run_tcn import EMA, decompose, metric_loss, pointwise

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv")
SMOOTH_WIN = 9


def causal_smooth(arr: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(win, min_periods=1).mean().to_numpy().astype(np.float32)


def pass1_channels(wells, art, cache_path):
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    medians, stds = art["medians"], art["stds"]
    out = {}
    for w in wells:
        Xw = make_features(w, context=False, priors=True).fillna(medians)
        Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        o = numpy_forward(Xw, art["sd"], art["dilations"])
        out[w["logId"].iloc[0]] = {c: causal_smooth(o[c][:, 1]) for c in TARGETS}
    joblib.dump(out, cache_path)
    return out


def build_wells(wells, medians, stds, ch_true, ch_pass1):
    out = []
    for w in wells:
        lid = w["logId"].iloc[0]
        X = make_features(w, context=False, priors=True).fillna(medians)
        X = ((X - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        Y = make_targets(w)
        isd = default_mask(Y)
        item = {"X": X, "logId": lid}
        for c in TARGETS:
            y = Y[c].to_numpy(dtype=np.float32)
            valid = ~np.isnan(y)
            t = np.log10(np.maximum(np.nan_to_num(y, nan=1.0), EPS)).astype(np.float32)
            item[f"{c}_valid"] = valid
            item[f"{c}_def"] = (isd[c].to_numpy() & valid).astype(np.float32)
            item[f"{c}_t"] = t
            item[f"{c}_y"] = y
            item[f"{c}_ch_true"] = causal_smooth(np.where(valid, t, 0.0))
            item[f"{c}_ch_p1"] = ch_pass1[lid][c] if ch_pass1 else np.zeros_like(t)
        out.append(item)
    return out


def sample_batch(data, lengths, weights, L, batch, p_true, rng):
    idx = rng.choice(len(data), size=batch, p=weights)
    xs, chs, masks = [], [], {f"{c}_{k}": [] for c in TARGETS for k in ("valid", "def", "t")}
    for i in idx:
        w = data[i]
        T = int(lengths[i])
        s = 0 if T <= L else int(rng.integers(0, T - L))
        sl = slice(s, s + L)
        xs.append(w["X"][sl])
        ch = []
        for c in TARGETS:
            use_true = rng.random() < p_true
            ch.append((w[f"{c}_ch_true"] if use_true else w[f"{c}_ch_p1"])[sl])
        chs.append(np.stack(ch, axis=1))  # (L, 3)
        for key in masks:
            masks[key].append(w[key][sl])
    b = {"X": torch.from_numpy(np.stack(xs)), "CH": torch.from_numpy(np.stack(chs))}
    for key, arrs in masks.items():
        b[key] = torch.from_numpy(np.stack(arrs).astype(np.float32))
    return b


def well_channels_infer(w, art, pass1_cache):
    lid = w["logId"].iloc[0]
    return np.stack([pass1_cache[lid][c] for c in TARGETS], axis=1)  # (T,3)


def numpy_forward2(x, ch, sd, dilations):
    return numpy_forward(np.concatenate([x, ch], axis=1).astype(np.float32), sd, dilations)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="T5_pass2")
    ap.add_argument("--pass1_tag", default="T3_sel80")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_wells = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va_wells = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df = pd.concat(tr_wells, ignore_index=True)
    base = make_features(tr_df, context=False, priors=True)
    medians, stds = base.median(), base.std()

    art = joblib.load(os.path.join(RUNS, f"tcn_{args.pass1_tag}.joblib"))
    p1_tr = pass1_channels(tr_wells, art, os.path.join(RUNS, f"pass1_train_{args.pass1_tag}.joblib"))
    p1_va = pass1_channels(va_wells, art, os.path.join(RUNS, f"pass1_val_{args.pass1_tag}.joblib"))

    tr_data = build_wells(tr_wells, medians, stds, None, p1_tr)
    va_data = build_wells(va_wells, medians, stds, None, p1_va)
    lengths = np.array([len(w["X"]) for w in tr_data], dtype=np.float64)
    weights = lengths / lengths.sum()
    n_features = tr_data[0]["X"].shape[1] + len(TARGETS)

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

    cache = {}
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        p_true = max(0.0, 1.0 - epoch / args.epochs)
        model.train()
        run = 0.0
        for _ in range(args.steps):
            b = sample_batch(tr_data, lengths, weights, args.window, args.batch, p_true, rng)
            for k in list(b.keys()):
                b[k] = b[k].to(device)
            x = torch.cat([b["X"], b["CH"]], dim=-1)
            out = model(x)
            loss = 0.0
            for c in TARGETS:
                valid = b[f"{c}_valid"] > 0.5
                defm = b[f"{c}_def"] > 0.5
                real = valid & ~defm
                if valid.any():
                    loss = loss + F.binary_cross_entropy_with_logits(out[c][:, :, 0][valid], b[f"{c}_def"][valid], pos_weight=pos_weight[c])
                loss = loss + metric_loss(out[c][:, :, 1], b[f"{c}_t"], valid, real, c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            run += float(loss.detach().cpu())
        sched.step()
        msg = f"epoch {epoch}/{args.epochs} loss={run / args.steps:.4f} p_true={p_true:.2f}"
        if epoch % 4 == 0 or epoch == args.epochs:
            sd_now = ema.state()
            outs = []
            for w in va_data:
                ch = np.stack([w[f"{c}_ch_p1"] for c in TARGETS], axis=1)
                outs.append(numpy_forward(w["X"], ch, sd_now, list(dilations) * 2) if False else numpy_forward2(w["X"], ch, sd_now, list(dilations) * 2))
            cache[epoch] = (sd_now, outs)
            dec = []
            for w, o in zip(va_data, outs):
                d = {}
                for c in TARGETS:
                    p = 1.0 / (1.0 + np.exp(-o[c][:, 0]))
                    pred = np.where(p >= 0.5, DEFAULT_VALUES[c], np.power(10.0, o[c][:, 1]))
                    d[c] = decompose(c, pred, w[f"{c}_y"], w[f"{c}_def"] > 0.5)
                dec.append(d)
            md = {c: {k: float(np.nanmean([d[c][k] for d in dec])) for k in dec[0][c]} for c in TARGETS}
            s05 = sum(WEIGHTS[c] * md[c]["acc"] for c in TARGETS) * 100.0
            msg += f" | val@0.5={s05:.3f} real=({md['POR']['acc_real']:.3f},{md['PERM']['acc_real']:.3f},{md['SW']['acc_real']:.3f})"
        print(msg, flush=True)

    grid_sel = np.arange(0.05, 0.96, 0.05)
    best_epoch, best_tuned = None, -1.0
    for ep, (_sd, outs) in cache.items():
        tot = 0.0
        for c in TARGETS:
            best_a = -1.0
            for t in grid_sel:
                accs = []
                for w, o in zip(va_data, outs):
                    p = 1.0 / (1.0 + np.exp(-o[c][:, 0]))
                    pred = np.where(p >= t, DEFAULT_VALUES[c], np.power(10.0, o[c][:, 1]))
                    y = w[f"{c}_y"]
                    m = ~np.isnan(y)
                    accs.append(pointwise(c, pred[m], y[m]).mean())
                best_a = max(best_a, float(np.mean(accs)))
            tot += WEIGHTS[c] * best_a
        print(f"  epoch {ep}: tuned={tot * 100:.3f}", flush=True)
        if tot > best_tuned:
            best_tuned, best_epoch = tot, ep
    print(f"best tuned={best_tuned * 100:.3f} @epoch {best_epoch} | {time.time() - t0:.0f}s")
    joblib.dump(
        {
            "sd": cache[best_epoch][0],
            "dilations": list(dilations) * 2,
            "medians": medians,
            "stds": stds,
            "n_features": n_features,
            "ch": args.ch,
            "pass1_tag": args.pass1_tag,
        },
        os.path.join(RUNS, f"tcn_{args.tag}.joblib"),
    )
    print("saved", os.path.join(RUNS, f"tcn_{args.tag}.joblib"))


if __name__ == "__main__":
    main()
