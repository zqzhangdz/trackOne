"""长感受野序列分段器探测：整井双向 BiLSTM 能否突破 AUC 天花板。

本任务唯一生死判据 = AUC。基线（72 训 / 8 验，官方口径）：
    点式 HistGB ±10/w21   POR 0.8966 / PERM 0.8906 / SW 0.9701
    双向 TCN w1024        POR 0.8935 / PERM 0.8909 / SW 0.9722
已有一次"双向 + 1024 窗口"尝试且 AUC 与点式树几乎相同。本脚本用**循环**连接
（而非膨胀卷积）重算：BiLSTM 在整井内的感受野无上限（最长 13078 点 = 1308 m），
TCN 约 497 点。输入同为 20 维点式特征，不做任何上下文扩张 —— 干净的模型类对比。

前提已排除（experiments/diag_boundary_noise.py）：标签**不是**噪声瓶颈。
POR/PERM 真实段取值离默认值 ≥ 20× 相对距离（2% 容差内 f = 0.000），
SW 真实段 2% 容差内仅 3.9% → AUC_max ≈ 0.981。故 0.897 是**特征信息**上限。

评估口径与部署一致：BiLSTM 只做"树 p_def"的 drop-in 替换，回归器 / TCN 分支 /
blend_w / clf_tcn_w 全部沿用 v7 缓存，阈值在 8 验证井上扫。

用法：
    python experiments/seq_clf_probe.py --smoke      # 2 epoch 试跑吞吐与显存
    python experiments/seq_clf_probe.py              # 12 epoch + 早停
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import resource
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import default_mask, load_wells, make_features, make_targets  # noqa: E402
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TRAIN_COLS, WEIGHTS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "experiments", "runs")
N_TRAIN_WELLS = 72

V7 = {"blend_w": {"POR": 0.50, "PERM": 0.50, "SW": 0.75},
      "thresholds": {"POR": 0.130, "PERM": 0.270, "SW": 0.050},
      "clf_tcn_w": {"POR": 0.60, "PERM": 0.70, "SW": 0.30}}
BASE_AUC = {"POR": 0.8966, "PERM": 0.8906, "SW": 0.9701}     # 点式树 ±10/w21
TCN_AUC = {"POR": 0.8935, "PERM": 0.8909, "SW": 0.9722}      # 双向 TCN w1024
V7_SCREEN = 73.749
TRUTH_SCREEN = 76.336
TH_GRID = np.arange(0.02, 0.981, 0.01)
T0 = time.time()


def elapsed():
    return time.time() - T0


def rss_gb():
    """峰值常驻集（GB）。macOS 的 ru_maxrss 单位随版本变：旧版 KB，Darwin 27 起是**字节**。

    用 `ps -o rss=`（恒定 KB）做参照判定单位：比值 ≥1000 说明 raw 是字节，否则是 KB。
    两个单位相差 1024 倍，而 ru_maxrss 是峰值（≥当前值），所以两档 [1,~100] 与
    [1024,~1e5] 之间没有真实重叠区。踩过一次：Darwin 27 上把字节当 KB 换算，
    把 2.62 GB 打成「2619.39 GB」。
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cur_kb = float(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(os.getpid())]).strip() or 1.0)
    return raw / 1e9 if raw / cur_kb >= 1000.0 else raw / 1e6


def load_wells_from(files: list[str]) -> list[pd.DataFrame]:
    """只读给定文件（与 src.feature.load_wells 同口径），避免把全 80 井读进内存."""
    out = []
    for f in files:
        df = pd.read_csv(f, skiprows=2, names=TRAIN_COLS, dtype=float)
        df["logId"] = os.path.splitext(os.path.basename(f))[0]
        out.append(df)
    return out


# ────────────────────────── 数据 ──────────────────────────

def prep():
    """逐井归一化 20 维点式特征 + 三目标 default_mask 标签。

    归一化用训练 72 井全局 median/std（与 experiments/run_tcn.py 同源），
    保留跨井可比性：储层/非储层由 DEN/AC 的**绝对**水平决定，
    逐井标准化会把信号抹掉。
    """
    wells = load_wells(os.path.join(ROOT, "train"), has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    va_ids = set(ids[N_TRAIN_WELLS:])
    tr_w = [w for w in wells if w["logId"].iloc[0] not in va_ids]
    va_w = [w for w in wells if w["logId"].iloc[0] in va_ids]

    base = make_features(pd.concat(tr_w, ignore_index=True), context=False, priors=True)
    med = base.median().to_numpy(np.float32)
    std = base.std().to_numpy(np.float32)
    std = np.where(std > 0, std, 1.0).astype(np.float32)
    cols = list(base.columns)
    med_df = pd.Series(med, index=cols)
    std_df = pd.Series(std, index=cols)

    def pack(ws):
        """返回 Xs（逐井特征）、lab（1=默认段，仅在 valid 处有效）、
        valid（目标非 NaN）。lab 与 valid 分离：官方指标只在 valid 上算。"""
        Xs, lab, val = [], {c: [] for c in TARGETS}, {c: [] for c in TARGETS}
        for w in ws:
            X = make_features(w, context=False, priors=True)[cols]
            Xs.append(((X.fillna(med_df) - med_df) / std_df).to_numpy(np.float32))
            Y = make_targets(w)
            D = default_mask(Y)
            for c in TARGETS:
                v = ~np.isnan(Y[c].to_numpy())
                lab[c].append((D[c].to_numpy() & v).astype(np.int8))
                val[c].append(v)
        return Xs, lab, val

    (tr_Xs, tr_lab, _), (va_Xs, va_lab, va_val) = pack(tr_w), pack(va_w)
    return tr_Xs, tr_lab, va_Xs, va_lab, va_val, med, std, cols


class WellIter:
    """逐井迭代：无 padding、无截断。

    反向趟从井尾向前推，任何 padding 都会把垃圾状态带进真实区；
    流式 chunk 已把激活内存与井长解耦，故不必 padding。
    """

    def __init__(self, Xs, lab, shuffle, seed=42):
        self.Xs, self.lab = Xs, lab
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(Xs))
        if shuffle:
            self.rng.shuffle(self.order)

    def reset(self):
        if self.shuffle:
            self.rng.shuffle(self.order)

    def __iter__(self):
        self.reset()
        for j in self.order:
            x = torch.from_numpy(self.Xs[j]).unsqueeze(0)
            yt = torch.stack([torch.from_numpy(self.lab[c][j].astype(np.float32))
                              for c in TARGETS], dim=-1).unsqueeze(0)
            yield x, yt


# ────────────────────────── 模型 ──────────────────────────

class BiLSTMClf(nn.Module):
    """2 层双向 LSTM → 3 个逐目标 default 概率头。

    **流式**双向：正反两趟各按 chunk 前推、携带 (h,c)，感受野 = 整井
    （最长 13078 点 = 1308 m），而激活内存只与 chunk 长相关（MPS 友好）。
    """

    def __init__(self, n_feat=20, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_feat, hidden)
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True,
                            bidirectional=False, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, 3)      # 两个方向拼接 → 2*hidden

    def stream(self, x):
        """x: (B, L, n_feat) → (B, L, 2*hidden)。正反两趟携带状态。"""
        B, L, _ = x.shape
        hsz, nlayer = self.lstm.hidden_size, self.lstm.num_layers
        CHUNK = 1024
        out = torch.empty(B, L, 2 * hsz, device=x.device, dtype=x.dtype)
        ninit = (nlayer, B, hsz)
        for rev in (False, True):
            h = torch.zeros(ninit, device=x.device, dtype=x.dtype)
            c = torch.zeros(ninit, device=x.device, dtype=x.dtype)
            for s in range(0, L, CHUNK):
                e = min(s + CHUNK, L)
                seg = x[:, s:e] if not rev else x[:, L - e:L - s].flip(1)
                o, (h, c) = self.lstm(seg, (h, c))
                if not rev:
                    out[:, s:e, :hsz] = o
                else:
                    out[:, L - e:L - s, hsz:] = o.flip(1)
        return out

    def forward(self, x):
        return self.head(self.stream(self.drop(self.proj(x))))


# ────────────────────────── 指标 ──────────────────────────

def acc_vec(c, pred, true):
    p, t = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    tol = 0.08 if c == "POR" else 0.05
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (tol * (np.abs(t) + EPS)))


def log10ify(c, r):
    """缓存里的 reg **三个目标全部已在 log10 空间**（实测 POR p50 0.9916≈log10(9.8%)、
    SW 1.9271≈log10(85)、PERM -0.893），与 diag_seg_bottleneck.py:73 的
    np.power(10.0, reg) 一致。**不是** src/feature.py 的 perm_log 语义（那只变换 PERM）。
    所以这里必须恒等 —— 早先版本对 POR/SW 又取了一次 log10，把 POR 真实段预测从中位 9.81
    打到 2.86（真值 11.31），使自检 Total 变成 99.864 而非 73.749。"""
    return np.asarray(r, np.float64)


def sweep(pmv, regb, c, y, m):
    """固定 (clf_tcn_w, blend_w)，只在阈值上扫。返回 (最优 acc, 最优 th)。"""
    pmv, regb, y = pmv[m], regb[m], y[m]
    fd = acc_vec(c, np.full(len(pmv), DEFAULT_VALUES[c]), y)
    fr = acc_vec(c, regb, y)
    order = np.argsort(pmv, kind="stable")
    pos = np.searchsorted(pmv[order], TH_GRID, side="left")
    Fd = np.concatenate([[0.0], np.cumsum(fd[order])])
    Fr = np.concatenate([[0.0], np.cumsum(fr[order])])
    n = len(pmv)
    acc = (Fr[pos] + (Fd[n] - Fd[pos])) / n
    j = int(np.argmax(acc))
    return float(acc[j]), float(TH_GRID[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="mps")
    ap.add_argument("--maxlen", type=int, default=0,
                    help="单井最大点数（0=不截断）。长井 13078 点会撑爆 batch 内存。")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--tag", type=str, default="seq_bilstm")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.patience = 2, 5

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device(
        args.device if (args.device == "cpu" or torch.backends.mps.is_available()) else "cpu")

    tr_Xs, tr_lab, va_Xs, va_lab, va_val, med, std, cols = prep()
    n_feat = tr_Xs[0].shape[1]
    print(f"device={device} torch={torch.__version__}   [{elapsed():.0f}s]")
    print(f"特征 {n_feat} 维 | 训练 {len(tr_Xs)} 井 {sum(len(x) for x in tr_Xs):,} 点 | "
          f"验证 {len(va_Xs)} 井 {sum(len(x) for x in va_Xs):,} 点   [{elapsed():.0f}s]")

    pw = {}
    for c in TARGETS:
        nd = sum(int(l.sum()) for l in tr_lab[c])
        nr = sum(len(l) - int(l.sum()) for l in tr_lab[c])
        pw[c] = float(np.clip(nd / max(nr, 1.0), 0.1, 20.0))
    print(f"pos_weight(def/real): " + "  ".join(f"{c}={pw[c]:.3f}" for c in TARGETS))

    model = BiLSTMClf(n_feat, args.hidden, args.layers, args.dropout).to(device)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    batcher = WellIter(tr_Xs, tr_lab, shuffle=True)

    def fwd(Xs):
        """逐井推理，返回 (N, 3) 的 sigmoid 输出。"""
        outs = []
        with torch.no_grad():
            for x in Xs:
                X = torch.from_numpy(x).unsqueeze(0).to(device)
                o = torch.sigmoid(model(X)).float().cpu().numpy()[0]
                outs.append(o)
                del X
        return np.concatenate(outs, axis=0)

    # 验证集掩码：va_valid = 目标非 NaN（官方指标口径）；va_def = 默认段标签
    va_valid = {c: np.concatenate(va_val[c]) for c in TARGETS}
    va_def = {c: np.concatenate(va_lab[c]).astype(bool) for c in TARGETS}
    print("验证 valid/默认段点数: " + "  ".join(
        f"{c}={int(va_valid[c].sum()):,}/{int((va_valid[c] & va_def[c]).sum()):,}"
        for c in TARGETS))

    def val_metrics(sd=None):
        """sd=None → 评估当前权重。返回三目标 AUC（只在 valid 上算）。"""
        if sd is not None:
            model.load_state_dict(sd)
        model.eval()
        out = fwd(va_Xs)
        return {c: float(roc_auc_score((va_def[c] & va_valid[c])[va_valid[c]],
                                       out[va_valid[c], i]))
                for i, c in enumerate(TARGETS)}

    best, best_auc, waited, hist = None, -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        batcher.reset()
        t_ep = time.time()
        losses, nst = [], 0
        for X, yt in batcher:
            X, yt = X.to(device), yt.to(device)
            logits = model(X)
            tot = torch.zeros((), device=device)
            for i, c in enumerate(TARGETS):
                tot = tot + nn.functional.binary_cross_entropy_with_logits(
                    logits[..., i], yt[..., i],
                    pos_weight=torch.tensor(pw[c], device=device), reduction="sum")
            loss = tot / 3.0
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
            nst += 1
        sched.step()
        auc = val_metrics(None)
        mA = float(np.mean(list(auc.values())))
        dt = time.time() - t_ep
        hist.append({"ep": ep, "sec": dt, "mean_auc": mA,
                     **{f"auc_{c}": auc[c] for c in TARGETS},
                     "loss": float(np.mean(losses)), "steps": nst})
        print(f"ep{ep:2d} {dt:6.1f}s  " + "  ".join(f"{c}={auc[c]:.4f}" for c in TARGETS)
              + f"  mean={mA:.4f}  loss={hist[-1]['loss']:.4f}  [{elapsed():.0f}s]",
              flush=True)
        if mA > best_auc + 1e-5:
            best, best_auc, waited = (
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, mA, 0)
            # 每见新高就落盘：本机内存紧张时进程可能被杀（实测 mps 版在 ep7 无栈被杀），
            # 增量保存保证中断后仍有最佳权重可续跑打分段。
            joblib.dump({"sd": best, "best_auc": best_auc, "hist": hist,
                         "cfg": {k: getattr(args, k) for k in
                                 ("hidden", "layers", "dropout", "lr", "epochs", "tag", "batch")},
                         "n_feat": n_feat},
                        os.path.join(RUNS, f"seq_bilstm_{args.tag}.joblib"))
        else:
            waited += 1
            if waited >= args.patience:
                print(f"早停 ep{ep}（patience={args.patience}，best mean AUC={best_auc:.4f}）")
                break

    model.load_state_dict(best)
    model.eval()
    # 存最佳权重：AUC 已定案时可用 --score_only 免重训，只跑打分段
    joblib.dump({"sd": best, "best_auc": best_auc, "hist": hist,
                 "cfg": {k: getattr(args, k) for k in
                         ("hidden", "layers", "dropout", "lr", "epochs", "tag", "batch")},
                 "n_feat": n_feat},
                os.path.join(RUNS, f"seq_bilstm_{args.tag}.joblib"))
    per_ep = float(np.mean([h["sec"] for h in hist]))
    print(f"\n最佳 ep{max(h['ep'] for h in hist if h['mean_auc']==best_auc)} "
          f"mean val AUC={best_auc:.4f} | 平均 {per_ep:.1f}s/epoch "
          f"({sum(h['sec'] for h in hist):.0f}s 训练)   [{elapsed():.0f}s]")

    # ── AUC 生死判据 ──
    pdf_va = fwd(va_Xs)                               # (72779, 3)
    # 落盘 pdf_va：训练约 11 min/次，但 λ 融合扫描是纯前向后的算术，不该为它重训。
    np.savez_compressed(os.path.join(RUNS, f"seq_bilstm_pdf_{args.tag}.npz"),
                        pdf_va=pdf_va, tag=args.tag)
    auc_seq = {c: float(roc_auc_score(va_def[c][va_valid[c]], pdf_va[va_valid[c], i]))
               for i, c in enumerate(TARGETS)}
    print("\n" + "=" * 100)
    print("AUC 对比（8 验证井，官方口径 screen）—— 本任务唯一生死判据")
    print("=" * 100)
    print(f"{'目标':6s}{'点式树±10':>11s}{'双向TCN w1024':>15s}{'BiLSTM整井':>12s}"
          f"{'Δvs树':>9s}{'突破?':>8s}")
    broke = {}
    for c in TARGETS:
        d = auc_seq[c] - BASE_AUC[c]
        broke[c] = d > 1e-4
        print(f"{c:6s}{BASE_AUC[c]:11.4f}{TCN_AUC[c]:15.4f}{auc_seq[c]:12.4f}{d:+9.4f}"
              f"{'  是' if broke[c] else '   否':>8s}")
    any_broke = any(broke.values())
    print(f"\n结论：BiLSTM 整井感受野 "
          f"{'突破了' if any_broke else '没有突破'} "
          f"点式树 AUC 天花板。   [{elapsed():.0f}s]")

    # ── 加载 v7 缓存，做 drop-in 融合与总分 ──
    # 先释放模型与训练数据：打分段只需要 pdf_va（已算完）+ va_Xs + 缓存。
    bnd = sum(len(x) for x in tr_Xs)
    del model, batcher, tr_Xs, tr_lab, tr_def
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    Z = joblib.load(os.path.join(RUNS, "screen_tree_cache.joblib"))
    p_tree = {c: np.float32(Z["p_def"][c]) for c in TARGETS}
    r_tree = {c: np.float32(Z["reg"][c]) for c in TARGETS}
    T = np.load(os.path.join(RUNS, "tcn_all80_parts.npz"))
    reg_t = {c: np.float32(T[f"reg_{c}"][bnd:]) for c in TARGETS}
    pdf_t = {c: np.float32(T[f"pdf_{c}"][bnd:]) for c in TARGETS}
    # 只读 8 口验证井的标签，不要 load_wells 全 80 井（多占数百 MB，会挤爆 16 GB）
    _files = sorted(glob.glob(os.path.join(ROOT, "train", "*.txt")))[N_TRAIN_WELLS:]
    va_df = pd.concat(load_wells_from(_files), ignore_index=True)
    Y = make_targets(va_df)
    yva = {c: Y[c].to_numpy(np.float64) for c in TARGETS}
    valid = {c: ~np.isnan(yva[c]) for c in TARGETS}
    assert all(valid[c].shape == va_valid[c].shape and (valid[c] == va_valid[c]).all()
               for c in TARGETS), "valid 掩码不一致"
    va_mask = va_def          # 别名：va_mask 恒指"默认段标签"，不是有效性掩码
    print(f"\n缓存对齐：树 {len(next(iter(p_tree.values()))):,} / TCN {len(next(iter(reg_t.values()))):,} "
          f"/ BiLSTM {len(pdf_va):,} / 验证行 {len(va_df):,}")

    def score_at(pmv, regb, c, y, m, th):
        pred = np.where(pmv >= th, DEFAULT_VALUES[c], regb)
        return float(acc_vec(c, pred[m], y[m]).mean())

    def i_of(c):
        return TARGETS.index(c)

    # v7 基线自检：必须落在 73.749 ± 0.05
    res_v7 = {}
    tot_v7 = 0.0
    for c in TARGETS:
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        pm = (1 - cw) * p_tree[c] + cw * pdf_t[c]
        rt = r_tree[c].astype(np.float64)      # 缓存已 log10（eval_v4.py:128），勿再取 log
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        m = valid[c]
        a, th = sweep(pm, regb, c, yva[c], m)
        tot_v7 += WEIGHTS[c] * a * 100
        res_v7[c] = {"acc": a, "th": th,
                     "auc_tree": float(roc_auc_score(va_def[c][valid[c]], p_tree[c][valid[c]]))}
    ok = abs(tot_v7 - V7_SCREEN) <= 0.05
    print(f"\n基线自检：harness v7 Total = {tot_v7:.3f}（部署 {V7_SCREEN}，"
          f"容差 ±0.05）→ {'PASS' if ok else 'FAIL'}   [{elapsed():.0f}s]")

    # 候选：BiLSTM 替换树的 p_def
    tot_seq = 0.0
    cand = {}
    for c in TARGETS:
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        pm = (1 - cw) * pdf_va[:, i_of(c)].astype(np.float32) + cw * pdf_t[c]
        rt = r_tree[c].astype(np.float64)      # 缓存已 log10（eval_v4.py:128），勿再取 log
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        m = valid[c]
        a, th = sweep(pm, regb, c, yva[c], m)
        a_v7th = score_at(pm, regb, c, yva[c], m, V7["thresholds"][c])
        tot_seq += WEIGHTS[c] * a * 100
        cand[c] = {"acc": a, "th": th, "acc_v7th": a_v7th,
                   "auc": float(roc_auc_score(va_def[c][valid[c]], pm[valid[c]])),
                   "auc_seq_raw": auc_seq[c]}
    print("\n" + "=" * 100)
    print("官方 screen Total（BiLSTM 做树 p_def 的 drop-in 替换；回归器/TCN/权重全沿用 v7）")
    print("=" * 100)
    print(f"{'目标':6s}{'权重':>6s}{'v7 acc':>9s}{'v7 th':>7s}{'seq acc':>9s}{'seq th':>8s}"
          f"{'seq@v7th':>10s}{'融合AUC':>9s}{'Δ贡献':>9s}")
    for c in TARGETS:
        r0, r1 = res_v7[c], cand[c]
        print(f"{c:6s}{WEIGHTS[c]:6.2f}{r0['acc']:9.4f}{r0['th']:7.3f}"
              f"{r1['acc']:9.4f}{r1['th']:8.3f}{r1['acc_v7th']:10.4f}"
              f"{r1['auc']:9.4f}{WEIGHTS[c]*(r1['acc']-r0['acc'])*100:+9.3f}")
    dv = tot_seq - tot_v7
    print(f"\n  Total: v7 = {tot_v7:.3f}   BiLSTM = {tot_seq:.3f}   Δ = {dv:+.3f}")
    print(f"  真值分段上界（不可达）= {TRUTH_SCREEN}；screen 噪声 ≈ ±0.3")
    print(f"  |Δ| = {abs(dv):.3f} → {'超出噪声带' if abs(dv) >= 0.3 else '在噪声带内'}")

    # ── λ 融合：BiLSTM 作为**第三路**加入 v7 集成，而不是替换树 ──
    # 上面的 drop-in 替换只测了「用 BiLSTM 换掉树的 p_def」。但两路信息不完全重叠
    #（树看 225 个上下文特征，BiLSTM 看 20 个基础特征 + 整井感受野），
    # 集成理论上应严格不劣。这里测 pm = (1-λ)·pm_v7 + λ·p_BiLSTM。
    print("\n" + "=" * 100)
    print("λ 融合扫描：pm = (1-λ)·pm_v7 + λ·p_BiLSTM（替换假设之外的另一种用法）")
    print("=" * 100)
    print(f"{'目标':6s}{'λ*':>7s}{'acc@λ*':>10s}{'acc@λ=0':>10s}{'最优th':>9s}{'Δ贡献':>10s}")
    dblend = 0.0
    for c in TARGETS:
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        pm0 = (1 - cw) * p_tree[c] + cw * pdf_t[c]
        rt = r_tree[c].astype(np.float64)      # 缓存已 log10（eval_v4.py:128），勿再取 log
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        m = valid[c]
        best_a, best_l, best_th = -1.0, 0.0, 0.0
        for lam in np.linspace(0.0, 1.0, 21):
            pm = (1 - lam) * pm0 + lam * pdf_va[:, i_of(c)].astype(np.float32)
            a, th = sweep(pm, regb, c, yva[c], m)
            if a > best_a:
                best_a, best_l, best_th = a, float(lam), th
        a0, _ = sweep(pm0, regb, c, yva[c], m)
        d = (best_a - a0) * WEIGHTS[c] * 100
        dblend += d
        print(f"{c:6s}{best_l:>7.2f}{best_a:>10.4f}{a0:>10.4f}{best_th:>9.3f}{d:>+10.3f}")
    print(f"  Σ Δ贡献（各目标各自最优 λ）= {dblend:+.3f}"
          f"   →  {'超出噪声带' if abs(dblend) >= 0.3 else '在噪声带内'}")

    # ── 留一井 jackknife（8 口井才是实验单元）──
    print("\n留一井 jackknife（逐井 Total：v7 vs BiLSTM）")
    print(f"{'井':32s}{'v7':>8s}{'seq':>8s}{'Δ':>9s}")
    well_rows, deltas = [], []
    off = 0
    for k, x in enumerate(va_Xs):
        s, off = off, off + len(x)
        row = {"v7": 0.0, "seq": 0.0}
        for c in TARGETS:
            cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
            rt = r_tree[c].astype(np.float64)      # 缓存已 log10（eval_v4.py:128），勿再取 log
            regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])[s:off]
            m = valid[c][s:off]
            y = yva[c][s:off]
            pm0 = ((1 - cw) * p_tree[c] + cw * pdf_t[c])[s:off]
            pm1 = (1 - cw) * pdf_va[s:off, i_of(c)] + cw * pdf_t[c][s:off]
            a0 = score_at(pm0, regb, c, y, m, V7["thresholds"][c])
            a1 = score_at(pm1, regb, c, y, m, V7["thresholds"][c])
            row["v7"] += WEIGHTS[c] * a0 * 100
            row["seq"] += WEIGHTS[c] * a1 * 100
        d = row["seq"] - row["v7"]
        deltas.append(d)
        print(f"well{k:02d} {'  '}{row['v7']:8.2f}{row['seq']:8.2f}{d:+9.3f}")
    deltas = np.array(deltas)
    np_pos = int((deltas > 0).sum())
    print(f"\n  逐井 Δ ∈ [{deltas.min():+.3f}, {deltas.max():+.3f}]，"
          f"{np_pos}/8 井为正")
    # 留一井 jackknife：每次去掉一口井，用剩 7 口的阈值
    loo = []
    for drop in range(len(va_Xs)):
        tot0 = tot1 = 0.0
        for c in TARGETS:
            cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
            rt = r_tree[c].astype(np.float64)      # 缓存已 log10（eval_v4.py:128），勿再取 log
            regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
            pm0 = (1 - cw) * p_tree[c] + cw * pdf_t[c]
            pm1 = (1 - cw) * pdf_va[:, i_of(c)].astype(np.float32) + cw * pdf_t[c]
            m_all = valid[c]
            # 去掉第 drop 口井，在剩 7 口上调阈值
            keep = np.ones(len(m_all), bool)
            keep[sum(len(x) for x in va_Xs[:drop]):sum(len(x) for x in va_Xs[:drop + 1])] = False
            mk = m_all & keep
            _, th0 = sweep(pm0, regb, c, yva[c], mk)
            _, th1 = sweep(pm1, regb, c, yva[c], mk)
            a0 = score_at(pm0, regb, c, yva[c], m_all, th0)
            a1 = score_at(pm1, regb, c, yva[c], m_all, th1)
            tot0 += WEIGHTS[c] * a0 * 100
            tot1 += WEIGHTS[c] * a1 * 100
        loo.append(tot1 - tot0)
    loo = np.array(loo)
    print(f"  留一井 jackknife Δ ∈ [{loo.min():+.3f}, {loo.max():+.3f}] "
          f"（跨 0 = {'是' if loo.min() < 0 < loo.max() else '否'}）")

    print(f"\n峰值 RSS = {rss_gb():.2f} GB   总耗时 = {elapsed():.0f}s")
    print(f"BiLSTM 参数 {npar:,}；hidden={args.hidden} layers={args.layers} "
          f"dropout={args.dropout} batch={args.batch} lr={args.lr}")

    # ── 分解：def / real（用真值 default_mask）──
    yv3 = default_mask(make_targets(va_df))
    decomp = {}
    for c in TARGETS:
        cw, bw = V7["clf_tcn_w"][c], V7["blend_w"][c]
        rt = r_tree[c].astype(np.float64)      # 缓存已 log10（eval_v4.py:128），勿再取 log
        regb = np.power(10.0, (1 - bw) * rt + bw * reg_t[c])
        pm = (1 - cw) * pdf_va[:, i_of(c)].astype(np.float32) + cw * pdf_t[c]
        m = valid[c]
        _, th = sweep(pm, regb, c, yva[c], m)
        pred = np.where(pm >= th, DEFAULT_VALUES[c], regb)
        # 勿命名 dv：外层 `dv = tot_seq - tot_v7` 是分数增量，被 bool 掩码覆盖后
        # CSV 打印 {dv:+.3f} 会抛 TypeError: unsupported format string passed to ndarray.__format__
        dvm = yv3[c].to_numpy() & m
        decomp[c] = (float(acc_vec(c, pred[m & dvm], yva[c][m & dvm]).mean()) if (m & dvm).sum() else float("nan"),
                     float(acc_vec(c, pred[m & ~dvm], yva[c][m & ~dvm]).mean()) if (m & ~dvm).sum() else float("nan"))

    # ── 追加 results.csv（表头固定，>> 追加）──
    csv_path = os.path.join(ROOT, "experiments", "results.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            raw = f.read()
        nl = b"\r\n" if raw.endswith(b"\r\n") else b"\n"
        cols_csv = [h.strip() for h in raw.split(b"\n")[0].decode().strip().split(",")]
        row = {
            "tag": args.tag,
            "mode": "screen",
            "folds": 1,
            "context": 0,
            "priors": 1,
            "ttransform": "perm_log",
            "weights": 0,
            "mono": 0,
            "total": round(tot_seq, 3),
            "std": 0.0,
            "POR_acc": round(cand["POR"]["acc"], 4),
            "PERM_acc": round(cand["PERM"]["acc"], 4),
            "SW_acc": round(cand["SW"]["acc"], 4),
            "POR_real": round(decomp["POR"][1], 4),
            "PERM_real": round(decomp["PERM"][1], 4),
            "SW_real": round(decomp["SW"][1], 4),
            "POR_def": round(decomp["POR"][0], 4),
            "PERM_def": round(decomp["PERM"][0], 4),
            "SW_def": round(decomp["SW"][0], 4),
        }
        with open(csv_path, "ab") as f:
            f.write(nl + (",".join(str(row.get(c, "")) for c in cols_csv) + "\n").encode())
        print(f"\n已追加 1 行 → {csv_path}（Δ vs v7 = {dv:+.3f}，"
              f"loo[{loo.min():+.3f},{loo.max():+.3f}]，{np_pos}/8 正，"
              f"AUC {'突破' if any_broke else '未突破'}）")

    return {
        "baseline_selfcheck_total": round(tot_v7, 3),
        "selfcheck_ok": bool(ok),
        "screen_total": round(tot_seq, 3),
        "por_auc": round(auc_seq["POR"], 4),
        "perm_auc": round(auc_seq["PERM"], 4),
        "sw_auc": round(auc_seq["SW"], 4),
        "por_acc": round(cand["POR"]["acc"], 4),
        "perm_acc": round(cand["PERM"]["acc"], 4),
        "sw_acc": round(cand["SW"]["acc"], 4),
        "loo_min": round(float(loo.min()), 3),
        "loo_max": round(float(loo.max()), 3),
        "n_positive_wells": np_pos,
        "runtime_min": round(elapsed() / 60, 1),
        "max_rss_gb": round(rss_gb(), 2),
        "per_epoch_sec": round(per_ep, 1),
        "delta_vs_v7": round(dv, 3),
        "auc_break": bool(any_broke),
        "episodes": hist,
    }


if __name__ == "__main__":
    main()
