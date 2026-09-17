"""训练最终混合模型并保存到 models/ 目录.

用法:
    python train.py --data_dir ./train --out_dir ./models --cv_folds 5 --with_tcn 1

流程:
  1) 两阶段树模型（上下文特征 + 物理先验 + POR 单调约束）全量训练;
  2) 可选(torch): 双向 TCN pass1 + 自回归通道 pass2 (scheduled sampling);
  3) 组装 model.pkl / scaler.pkl（混合推理所需全部工件）.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.feature import default_mask, inverse_transform_target, load_wells, make_features, make_targets, transform_target
from src.model import (
    EPS,
    TARGETS,
    TwoStageModel,
    get_tcn_classes,
    has_torch,
    numpy_forward_tcn,
)
from src.utils import DEFAULT_VALUES, TOLERANCE, WEIGHTS, acc_single

# 验证切分上调优得到的决策阈值（见 EXPERIMENTS.md）
FINAL_THRESHOLDS = {"POR": 0.10, "PERM": 0.05, "SW": 0.05}
FINAL_BLEND_W = [0.4, 0.3, 0.3]  # tree / tcn pass1 / tcn pass2
DILATIONS = (1, 2, 4, 8, 16)
SMOOTH_WIN = 9
LN10 = float(np.log(np.float64(10.0)))


def causal_smooth(arr: np.ndarray, win: int = SMOOTH_WIN) -> np.ndarray:
    csum = np.cumsum(np.concatenate([[0.0], np.asarray(arr, dtype=np.float64)]))
    n = np.arange(1, arr.size + 1)
    idx = np.maximum(n - win, 0)
    return ((csum[1:] - csum[idx]) / np.minimum(n, win)).astype(np.float32)


def metric_loss(torch_F, pred_t, t, real, c):
    d = (pred_t - t).abs()
    if c == "PERM":
        e = d / (TOLERANCE[c] * (t.abs() + EPS))
    else:
        e = LN10 * d / TOLERANCE[c]
    if real.sum() == 0:
        return pred_t.sum() * 0.0
    return e.clamp(max=2.0)[real].mean() + 0.05 * ((pred_t - t) ** 2)[real].mean()


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


def build_wells(wells, medians, stds, channels=None):
    out = []
    for w in wells:
        X = make_features(w, context=False, priors=True).fillna(medians)
        X = ((X - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
        Y = make_targets(w)
        isd = default_mask(Y)
        item = {"X": X, "logId": w["logId"].iloc[0]}
        for c in TARGETS:
            y = Y[c].to_numpy(dtype=np.float32)
            valid = ~np.isnan(y)
            t = np.log10(np.maximum(np.nan_to_num(y, nan=1.0), EPS)).astype(np.float32)
            item[f"{c}_valid"] = valid
            item[f"{c}_def"] = (isd[c].to_numpy() & valid).astype(np.float32)
            item[f"{c}_t"] = t
            if channels is not None:
                item[f"{c}_ch"] = channels[w["logId"].iloc[0]][c]
        out.append(item)
    return out


def sample_batch(data, lengths, weights, L, batch, rng, use_ch=False, p_true=None):
    idx = rng.choice(len(data), size=batch, p=weights)
    xs, chs, masks = [], [], {f"{c}_{k}": [] for c in TARGETS for k in ("valid", "def", "t")}
    for i in idx:
        w = data[i]
        T = int(lengths[i])
        s = 0 if T <= L else int(rng.integers(0, T - L))
        sl = slice(s, s + L)
        xs.append(w["X"][sl])
        if use_ch:
            chs.append(np.stack([w[f"{c}_ch"][sl] for c in TARGETS], axis=1))
        for key in masks:
            masks[key].append(w[key][sl])
    b = {"X": np.stack(xs)}
    if use_ch:
        b["CH"] = np.stack(chs)
    for key, arrs in masks.items():
        b[key] = np.stack(arrs).astype(np.float32)
    return b


def train_tcn(data, lengths, weights, n_features, args, device, epochs, channels=False, seed=42):
    import torch

    if lengths is None:
        lengths = np.array([len(w["X"]) for w in data], dtype=np.float64)
        weights = lengths / lengths.sum()
    import torch.nn.functional as F

    TCNNet = get_tcn_classes()["TCNNet"]

    import os as _os

    if _os.environ.get("TRAIN_DEBUG"):
        import faulthandler as _fh

        _fh.enable()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if os.environ.get("TRAIN_DEBUG"):
        print("    [dbg] build model", flush=True)
    model = TCNNet(n_features, ch=args.ch, dilations=DILATIONS, repeats=2).to(device)
    if os.environ.get("TRAIN_DEBUG"):
        print("    [dbg] ema", flush=True)
    ema = EMA(model)
    if os.environ.get("TRAIN_DEBUG"):
        print("    [dbg] opt", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    if os.environ.get("TRAIN_DEBUG"):
        print("    [dbg] pos_weight", flush=True)
    pos_weight = {}
    for c in TARGETS:
        nd = sum(float(w[f"{c}_def"].sum()) for w in data)
        nr = sum(float((w[f"{c}_valid"].sum() - w[f"{c}_def"].sum())) for w in data)
        pos_weight[c] = torch.tensor(max(0.1, min(10.0, nr / max(1.0, nd))), device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        run = 0.0
        for si in range(args.steps):
            if os.environ.get("TRAIN_DEBUG"):
                print(f"    [dbg] step {si}", flush=True)
            b = sample_batch(data, lengths, weights, args.window, args.batch, rng, use_ch=channels)
            bt = {k: torch.from_numpy(v).to(device) for k, v in b.items()}
            x = torch.cat([bt["X"], bt["CH"]], dim=-1) if channels else bt["X"]
            out = model(x)
            loss = 0.0
            for c in TARGETS:
                valid = bt[f"{c}_valid"] > 0.5
                defm = bt[f"{c}_def"] > 0.5
                real = valid & ~defm
                if valid.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out[c][:, :, 0][valid], bt[f"{c}_def"][valid], pos_weight=pos_weight[c]
                    )
                loss = loss + metric_loss(F, out[c][:, :, 1], bt[f"{c}_t"], real, c)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            run += float(loss.detach().cpu())
        sched.step()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"    epoch {epoch}/{epochs} loss={run / args.steps:.4f}", flush=True)
    return ema.state()


def cv_report(X, Y, isdef, groups, folds):
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=folds)
    scores = []
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(X, Y, groups), start=1):
        model = TwoStageModel(monotonic=True).fit(X.iloc[tr_idx], Y.iloc[tr_idx], isdef.iloc[tr_idx])
        accs = {}
        for c in TARGETS:
            p = model.classifiers[c].predict_proba(X.iloc[va_idx])[:, 1]
            reg = inverse_transform_target(
                c, pd.Series(model.regressors[c].predict(X.iloc[va_idx]), index=X.iloc[va_idx].index)
            ).to_numpy()
            pred = np.where(p >= FINAL_THRESHOLDS[c], DEFAULT_VALUES[c], reg)
            accs[c] = acc_single(c, pred, Y.iloc[va_idx][c].to_numpy())
        s = sum(WEIGHTS[c] * accs[c] for c in TARGETS) * 100
        scores.append(s)
        print(f"  fold{fi}: total={s:.3f} " + " ".join(f"{c}={accs[c]:.4f}" for c in TARGETS))
    print(f"树模型 CV: {np.mean(scores):.3f} ± {np.std(scores):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./train")
    ap.add_argument("--out_dir", default="./models")
    ap.add_argument("--cv_folds", type=int, default=0)
    ap.add_argument("--with_tcn", type=int, default=1)
    ap.add_argument(
        "--stage",
        choices=["all", "tree", "tcn1", "tcn2", "assemble"],
        default="all",
        help="all 时以子进程依次执行各阶段（隔离 sklearn/torch 的 OpenMP 运行时，避免死锁）",
    )
    ap.add_argument("--epochs1", type=int, default=80)
    ap.add_argument("--epochs2", type=int, default=60)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu",
                    help="TCN 训练设备；默认 cpu 以规避 macOS MPS 驱动与 OpenMP 的交互问题")
    args = ap.parse_args()

    if args.stage == "all":
        import subprocess
        import sys

        base_cmd = [sys.executable, os.path.abspath(__file__)]
        for k, v in vars(args).items():
            if k == "stage":
                continue
            base_cmd += [f"--{k}", str(v)]
        stages = ["tree"] + (["tcn1", "tcn2"] if args.with_tcn and has_torch() else []) + ["assemble"]
        for st in stages:
            print(f"=== stage {st} ===", flush=True)
            subprocess.run(base_cmd + ["--stage", st], check=True)
        return

    stage_dir = os.path.join(args.out_dir, ".staging")
    os.makedirs(stage_dir, exist_ok=True)
    wells = load_wells(args.data_dir, has_labels=True)
    df = pd.concat(wells, ignore_index=True)

    if args.stage == "tree":
        print(f"训练井数: {len(wells)}  总深度点: {len(df):,}")
        X = make_features(df, context=True, priors=True)
        Y = make_targets(df)
        isdef = default_mask(Y)
        if args.cv_folds > 0:
            print(f"树模型 {args.cv_folds} 折 CV:")
            cv_report(X, Y, isdef, df["logId"].to_numpy(), args.cv_folds)
        print("全量训练树模型 ...", flush=True)
        tree = TwoStageModel(thresholds=dict(FINAL_THRESHOLDS), monotonic=True).fit(X, Y, isdef)
        joblib.dump(
            {
                "model_dict": tree.to_dict(),
                "feature_columns": list(X.columns),
            },
            os.path.join(stage_dir, "tree.joblib"),
        )
        print("tree stage done", flush=True)
        return

    if args.stage in ("tcn1", "tcn2"):
        import torch

        device = torch.device(args.device)
        print("device:", device, flush=True)
        base = make_features(df, context=False, priors=True)
        medians, stds = base.median(), base.std()
        if args.stage == "tcn1":
            data = build_wells(wells, medians, stds)
            n_feat = data[0]["X"].shape[1]
            print(f"TCN pass1: {args.epochs1} epochs ...", flush=True)
            sd1 = train_tcn(data, None, None, n_feat, args, device, args.epochs1, channels=False, seed=args.seed)
            joblib.dump({"sd": sd1, "medians": medians, "stds": stds}, os.path.join(stage_dir, "tcn1.joblib"))
        else:
            art1 = joblib.load(os.path.join(stage_dir, "tcn1.joblib"))
            medians, stds = art1["medians"], art1["stds"]
            channels = {}
            for w in wells:
                Xw = make_features(w, context=False, priors=True).fillna(medians)
                Xw = ((Xw - stds.replace(0, 1.0)) / stds.replace(0, 1.0)).to_numpy(dtype=np.float32)
                o = numpy_forward_tcn(Xw, art1["sd"], list(DILATIONS) * 2)
                channels[w["logId"].iloc[0]] = {c: causal_smooth(o[c][:, 1]) for c in TARGETS}
            data = build_wells(wells, medians, stds, channels=channels)
            n_feat = data[0]["X"].shape[1] + len(TARGETS)
            print(f"TCN pass2: {args.epochs2} epochs (自回归通道) ...", flush=True)
            sd2 = train_tcn(data, None, None, n_feat, args, device, args.epochs2, channels=True, seed=args.seed + 1)
            joblib.dump({"sd": sd2}, os.path.join(stage_dir, "tcn2.joblib"))
        print(f"{args.stage} stage done", flush=True)
        return

    # assemble
    tree_part = joblib.load(os.path.join(stage_dir, "tree.joblib"))
    model_dict = tree_part["model_dict"]
    scaler_dict = {
        "feature_columns": tree_part["feature_columns"],
        "default_values": DEFAULT_VALUES,
    }
    tcn1_path = os.path.join(stage_dir, "tcn1.joblib")
    if os.path.exists(tcn1_path):
        art1 = joblib.load(tcn1_path)
        art2 = joblib.load(os.path.join(stage_dir, "tcn2.joblib"))
        model_dict.update(
            tcn_sd=art1["sd"],
            tcn_dilations=list(DILATIONS) * 2,
            tcn2_sd=art2["sd"],
            tcn2_dilations=list(DILATIONS) * 2,
            blend_w=list(FINAL_BLEND_W),
        )
        scaler_dict.update(
            tcn_medians=art1["medians"],
            tcn_stds=art1["stds"],
            tcn_feature_columns=list(make_features(df, context=False, priors=True).columns),
        )
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model_dict, os.path.join(args.out_dir, "model.pkl"))
    joblib.dump(scaler_dict, os.path.join(args.out_dir, "scaler.pkl"))
    print(f"模型已保存到 {args.out_dir} (tcn={os.path.exists(tcn1_path)})")


if __name__ == "__main__":
    main()
