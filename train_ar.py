"""训练自回归因果 GRU 序列模型."""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch

from solution.config import DEFAULT_VALUES, TARGETS
from solution.data import default_mask, load_wells, make_features, make_targets, transform_target
from solution.metric import acc_single
from solution.ar_model import ARSeqNet, apply_ar_thresholds, ar_predict
from solution.seq_model import seq_loss


def extended_features(df: pd.DataFrame) -> pd.DataFrame:
    X = make_features(df)
    X["POR_D"] = (2.65 - X["DEN"]) / 1.65 * 100.0
    X["POR_S"] = (X["AC"] - 55.5) / (189.0 - 55.5) * 100.0
    X["POR_avg"] = (X["POR_D"] + X["CNL"]) / 2.0
    return X


def build_well(
    well: pd.DataFrame,
    medians: pd.Series,
    stds: pd.Series,
    reg_mean: dict[str, float],
    reg_std: dict[str, float],
) -> dict:
    X = extended_features(well).fillna(medians)
    X = (X - stds.replace(0, 1.0)) / stds.replace(0, 1.0)
    Y = make_targets(well)
    isd = default_mask(Y)
    T = len(X)
    signal = np.zeros((T, len(TARGETS)), dtype=np.float32)
    out = {"X": X.to_numpy(dtype=np.float32)}
    for ci, c in enumerate(TARGETS):
        valid = Y[c].notna().to_numpy()
        def_ = (isd[c] & Y[c].notna()).to_numpy()
        reg_mask = valid & ~def_
        yt = transform_target(c, Y[c]).to_numpy(dtype=np.float32)
        std_v = (yt - reg_mean[c]) / reg_std[c]
        reg_target = std_v.copy()
        reg_target[~reg_mask] = 0.0
        sig = std_v.copy()
        sig[~valid] = 0.0
        signal[:, ci] = sig
        out[f"{c}_valid"] = valid
        out[f"{c}_def"] = def_
        out[f"{c}_reg_mask"] = reg_mask
        out[f"{c}_reg_target"] = reg_target
        out[f"{c}_y"] = Y[c].to_numpy(dtype=np.float32)
    out["signal"] = signal
    return out


def sample_batch(wells: list[dict], length: int, batch_size: int):
    chosen = [wells[int(np.random.randint(len(wells)))] for _ in range(batch_size)]
    xs, prevs = [], []
    masks = {c: [] for c in TARGETS}
    for w in chosen:
        T = len(w["X"])
        s = 0 if T <= length else int(np.random.randint(0, T - length))
        sl = slice(s, s + length)
        xs.append(w["X"][sl])
        if s == 0:
            prev = np.vstack([np.zeros((1, len(TARGETS)), np.float32), w["signal"][0 : length - 1]])
        else:
            prev = w["signal"][s - 1 : s - 1 + length]
        prevs.append(prev)
        for c in TARGETS:
            masks[c].append(
                {
                    f"{c}_valid": w[f"{c}_valid"][sl],
                    f"{c}_def": w[f"{c}_def"][sl],
                    f"{c}_reg_mask": w[f"{c}_reg_mask"][sl],
                    f"{c}_reg_target": w[f"{c}_reg_target"][sl],
                }
            )
    batch = {"X": torch.from_numpy(np.stack(xs)), "prev": torch.from_numpy(np.stack(prevs))}
    for c in TARGETS:
        for k in masks[c][0]:
            arr = np.stack([m[k] for m in masks[c]])
            dt = torch.bool if "valid" in k or "mask" in k else torch.float32
            batch[k] = torch.from_numpy(arr.astype(np.float32)).to(dt)
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./train")
    ap.add_argument("--out_dir", default="./artifacts_ar")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps_per_epoch", type=int, default=150)
    ap.add_argument("--w_clf", type=float, default=1.0)
    ap.add_argument("--w_reg", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    wells = load_wells(args.data_dir, has_labels=True)
    all_df = pd.concat(wells, ignore_index=True)
    extended = extended_features(all_df)
    medians = extended.median()
    stds = extended.std()

    ids = pd.unique(all_df["logId"])
    tr_ids = set(ids[:72])
    va_ids = set(ids[72:])
    tr_wells = [w for w in wells if w["logId"].iloc[0] in tr_ids]
    va_wells = [w for w in wells if w["logId"].iloc[0] in va_ids]

    reg_mean: dict[str, float] = {}
    reg_std: dict[str, float] = {}
    for c in TARGETS:
        vals = []
        for w in tr_wells:
            Y = make_targets(w)
            isd = default_mask(Y)
            m = Y[c].notna() & ~isd[c]
            vals.append(transform_target(c, Y[c])[m].to_numpy())
        v = np.concatenate(vals)
        reg_mean[c] = float(np.mean(v))
        reg_std[c] = float(np.std(v) + 1e-6)
    default_std = {c: (transform_target(c, pd.Series([DEFAULT_VALUES[c]]))[0] - reg_mean[c]) / reg_std[c] for c in TARGETS}
    print("reg mean/std:", {c: (round(reg_mean[c], 3), round(reg_std[c], 3)) for c in TARGETS})
    print("default_std:", {c: round(default_std[c], 3) for c in TARGETS})

    tr_data = [build_well(w, medians, stds, reg_mean, reg_std) for w in tr_wells]
    va_data = [build_well(w, medians, stds, reg_mean, reg_std) for w in va_wells]

    n_features = extended.shape[1]
    model = ARSeqNet(n_features, hidden=args.hidden, n_layers=args.n_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    pos_weight = {}
    for c in TARGETS:
        ndef = sum(int(w[f"{c}_def"].sum()) for w in tr_data)
        nreal = sum(int(w[f"{c}_reg_mask"].sum()) for w in tr_data)
        pos_weight[c] = max(0.1, min(10.0, nreal / max(1, ndef)))
    betas = {"POR": 1.0, "PERM": 1.0, "SW": 1.0}

    def quick_val(th: float = 0.5) -> float:
        tot = []
        for w in va_data[:3]:
            th_dict = {c: th for c in TARGETS}
            pred, _, _ = ar_predict(model, w["X"], th_dict, reg_mean, reg_std, default_std, device)
            acc = {c: acc_single(c, pred[c], w[f"{c}_y"]) for c in TARGETS}
            tot.append((0.3 * acc["POR"] + 0.35 * acc["PERM"] + 0.35 * acc["SW"]) * 100)
        return float(np.mean(tot))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            batch = sample_batch(tr_data, args.window, args.batch_size)
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            opt.zero_grad()
            clf, reg = model(batch["X"], batch["prev"])
            loss = seq_loss(clf, reg, batch, pos_weight, betas, args.w_clf, args.w_reg)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
        sched.step()
        print(f"epoch {epoch}/{args.epochs}  loss={running/args.steps_per_epoch:.4f}  lr={sched.get_last_lr()[0]:.2e}")
        if epoch % 2 == 0:
            print(f"  [val@0.5] total={quick_val(0.5):.3f}")

    # 用参考阈值 0.5 缓存原始输出，用于快速阈值寻优
    ref_th = {c: 0.5 for c in TARGETS}
    cache = []
    for w in va_data:
        _, p_def, reg_out = ar_predict(model, w["X"], ref_th, reg_mean, reg_std, default_std, device)
        cache.append((p_def, reg_out, w))

    grid = np.arange(0.05, 0.96, 0.05)
    final_th = {}
    for c in TARGETS:
        best_th, best_acc = 0.5, -1.0
        for th in grid:
            accs = []
            for p_def, reg_out, w in cache:
                th_dict = {k: (th if k == c else 0.5) for k in TARGETS}
                pred = apply_ar_thresholds(p_def, reg_out, th_dict, reg_mean, reg_std)[c]
                y = w[f"{c}_y"]
                m = ~np.isnan(y)
                if m.sum() == 0:
                    continue
                accs.append(acc_single(c, pred[m], y[m]))
            if not accs:
                continue
            a = float(np.mean(accs))
            if a > best_acc:
                best_acc, best_th = a, float(th)
        final_th[c] = best_th
        print(f"{c}: val acc={best_acc:.4f} @ th={best_th}")

    tot = []
    for w in va_data:
        pred, _, _ = ar_predict(model, w["X"], final_th, reg_mean, reg_std, default_std, device)
        acc = {c: acc_single(c, pred[c], w[f"{c}_y"]) for c in TARGETS}
        tot.append((0.3 * acc["POR"] + 0.35 * acc["PERM"] + 0.35 * acc["SW"]) * 100)
    print("验证集总分(每井平均):", round(float(np.mean(tot)), 3))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "ar_model.pt"))
    joblib.dump(
        {
            "medians": medians,
            "stds": stds,
            "thresholds": final_th,
            "n_features": n_features,
            "hidden": args.hidden,
            "n_layers": args.n_layers,
            "feature_cols": list(extended.columns),
            "reg_mean": reg_mean,
            "reg_std": reg_std,
            "default_std": default_std,
        },
        os.path.join(args.out_dir, "ar_prep.joblib"),
    )
    print("saved to", args.out_dir)


if __name__ == "__main__":
    main()
