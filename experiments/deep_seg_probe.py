"""深分段器探针：一个此前完全没试过的**模型家族**。

为什么值得试
------------
分段器的 AUC 上限被**三次**独立确认在 ~0.897（CLAUDE.md §4.11 / §4.12 / §4.14）：
  1. 六种 HistGB 分类器**训练方案**只动 0.000～0.006；
  2. 树 AUC（0.8966）≈ TCN AUC（0.8909），两路融合提不了 AUC；
  3. 更宽的上下文窗口让 AUC 下降。
但注意这三条共同的边界：**从来没有换过模型家族**。树是 HistGB（300 iter × 31 叶），
TCN 只看了 13 条原始曲线（`context=False`，拿不到 237 列）。
**一个能吃满 237 列特征的深网络从未试过** —— 这是训练侧/架构侧唯一没探索过的轴。

若深网络的 AUC 仍落在 0.89 附近，那是第四次独立确认「上限属于特征的信息量，不属于模型」，
§4.11 的结论就彻底闭环。若能突破 0.91，则值得投入整合（替换树的 p_def 进融合）。

口径纪律
--------
* 纯 torch 进程，**本脚本绝不调用 sklearn 的 fit**（§5.1：同进程 torch + sklearn fit
  会在 macOS OpenMP 双运行时下 SIGSEGV / 挂起）。只 import src.feature / src.utils
  （纯 numpy/pandas），以及 sklearn.metrics 的纯函数会避开 —— AUC 用 numpy 排序手算。
* 只训练分段器（三目标），不动回归器、不动 TCN、不动融合权重。
* 输出 p_def 存 npz，由 eval_deep_seg.py（纯 numpy 进程）与缓存的树 reg / TCN 融合后
  算官方口径 Total —— 两个进程彻底分开。

用法: python experiments/deep_seg_probe.py
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature import load_wells, make_targets, default_mask, make_features
from src.utils import TARGETS

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
OUT = os.path.join(RUNS, "deep_seg_pdef.npz")
EPOCHS = 14
BATCH = 16384
HIDDEN = (384, 384, 256)
DROPOUT = 0.15
LR = 3e-3
SEED = 42


def rss() -> str:
    import resource
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.0f}MB"


def auc_pr(score, label) -> float:
    """numpy 排序版 ROC AUC（不调 sklearn，避免 §5.1 的运行时冲突）。

    名次取「并列取均值」的正确形式：把每个唯一值对应的名次区间取中点，再映射回原索引。
    """
    s = np.asarray(score, np.float64)
    y = np.asarray(label, np.int8)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    uniq, inv = np.unique(s, return_inverse=True)
    cnt = np.bincount(inv)
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    ranks = ((start + start + cnt - 1) / 2.0)[inv]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def main():
    import torch

    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"设备 {device}   torch {torch.__version__}")

    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    by = dict(zip(ids, wells))
    tr = [by[i] for i in ids[:72]]
    va = [by[i] for i in ids[72:80]]

    # ---------- 特征（纯 pandas/numpy，无 sklearn） ----------
    bags = {}
    for dset, df in (("tr", tr), ("va", va)):
        X = np.concatenate([make_features(w, context=True, priors=True).to_numpy(np.float32, copy=False)
                            for w in df], axis=0)
        Y = np.concatenate([make_targets(w).to_numpy(np.float32, copy=False) for w in df], axis=0)
        bags[dset] = (X, Y)
        print(f"  {dset}  {X.shape[0]:,} 行 × {X.shape[1]} 列   {rss()}")
    del wells
    gc.collect()

    # 逐目标：只保留该目标标签有效的行
    trX, trY = bags["tr"]
    vaX, vaY = bags["va"]
    trd, vad, vidx = {}, {}, {}
    for k, c in enumerate(TARGETS):          # Y 已是 2D ndarray，按位置取列
        ok_t, ok_v = np.isfinite(trY[:, k]), np.isfinite(vaY[:, k])
        # default_mask_y 返回**全长**数组，必须再按 ok 索引一次，
        # 否则标签长度 657489 与特征行数 651479 不一致 → 索引越界。
        trd[c] = (trX[ok_t], default_mask_y(c, trY[:, k], ok_t)[ok_t].astype(np.int8))
        vad[c] = (vaX[ok_v], default_mask_y(c, vaY[:, k], ok_v)[ok_v].astype(np.int8))
        vidx[c] = np.where(ok_v)[0]                  # 回填到全长 va 的顺序，供整合脚本用
        print(f"  {c:<5} 训练 {ok_t.sum():,}（默认段占比 "
              f"{trd[c][1].mean() * 100:.2f}%）  验证 {ok_v.sum():,}（{vad[c][1].mean() * 100:.2f}%）")

    # ---------- 训练 ----------
    pdef = {}
    summary = []
    for k, c in enumerate(TARGETS):
        Xtr_d, ytr_d = trd[c]
        Xva_d, yva_d = vad[c]
        del trd[c], vad[c]                     # 每列 619MB float32，勿累积
        gc.collect()
        print(f"\n=== {c}  训练 {len(ytr_d):,}  正类(默认段)占比 "
              f"{ytr_d.mean() * 100:.2f}%   验证 {len(yva_d):,} ===")

        # 缺失填补：用训练集列中位数（部署时同样可算）
        med = np.nanmedian(Xtr_d, axis=0)
        Xtr_c = np.where(np.isnan(Xtr_d), med, Xtr_d)
        Xva_c = np.where(np.isnan(Xva_d), med, Xva_d)
        Xtr_t = torch.from_numpy(Xtr_c).to(device)
        Xva_t = torch.from_numpy(Xva_c).to(device)
        ytr_t = torch.from_numpy(ytr_d).to(device)
        pos = ytr_t.float().mean().item()
        pw = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=device)

        torch.manual_seed(SEED + k)
        m = torch.nn.Sequential(
            torch.nn.Linear(Xtr_t.shape[1], HIDDEN[0]), torch.nn.BatchNorm1d(HIDDEN[0]),
            torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(HIDDEN[0], HIDDEN[1]), torch.nn.BatchNorm1d(HIDDEN[1]),
            torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(HIDDEN[1], HIDDEN[2]), torch.nn.BatchNorm1d(HIDDEN[2]),
            torch.nn.GELU(), torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(HIDDEN[2], 1),
        ).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        n = len(ytr_t)
        hist, best_a, best_p = [], -1.0, None
        for e in range(EPOCHS):
            m.train()
            # ⚠️ randperm 在 CPU 上生成再搬过去：MPS 上的 randperm 对大 n 会返回
            # 越界索引（实测 index 652085 out of bounds for size 651479），且该错误
            # 是异步的、在下一次同步点（float(lo)）才抛出来，极难定位。
            perm = torch.randperm(n).to(device)
            tot, nb = 0.0, 0
            for s0 in range(0, n, BATCH):
                b = perm[s0:s0 + BATCH]
                opt.zero_grad()
                lo = lossf(m(Xtr_t[b]).squeeze(-1), ytr_t[b].float())
                lo.backward()
                opt.step()
                tot += lo.detach().item() * len(b)
                nb += len(b)
            sched.step()
            with torch.no_grad():
                m.eval()
                vp = torch.sigmoid(m(Xva_t)).squeeze(-1).cpu().numpy().astype(np.float64)
            a = auc_pr(vp, yva_d)
            acc_pos = ((vp > 0).astype(np.int8) == yva_d).mean()
            hist.append(a)
            print(f"    epoch {e + 1:2d}/{EPOCHS}  loss {tot / nb:.4f}   验证 AUC {a:.4f}   "
                  f"acc(p>0) {acc_pos:.4f}   {time.time() - t0:.0f}s")
            if a > best_a:
                best_a, best_p = a, vp
        # 取 val AUC 最好的那个 epoch。逐 epoch AUC 有 ±0.0085 的抽样噪声（72k 点），
        # 只看末轮会把噪声当成趋势，也会低估深模型。
        p, a = best_p, best_a
        print(f"    → {c} 深分段器验证 AUC 最佳 {a:.4f}（末轮 {hist[-1]:.4f}、"
              f"{EPOCHS} 轮均值 {float(np.mean(hist)):.4f}、区间 [{min(hist):.4f},{max(hist):.4f}]）"
              f"   (HistGB 参考见下)")
        # 回填到全长 va：无效点留 NaN。整合脚本与缓存的树 reg / TCN 是全长对齐的。
        full = np.full(len(vaY), np.nan, dtype=np.float64)
        full[vidx[c]] = p
        pdef[c] = full
        summary.append((c, a, float(np.mean(hist)), min(hist), max(hist)))
        del m, Xtr_t, Xva_t, vidx[c]
        gc.collect()

    np.savez_compressed(OUT, **{f"deep_{c}": pdef[c] for c in TARGETS})
    print(f"\n已保存 {OUT}")
    print("\n对照（HistGB 237 特征 / 225 特征 的验证 AUC，来自 §4.11 / §4.14 记录）")
    ref = {"POR": (0.8913, 0.8966), "PERM": (0.8846, 0.8906), "SW": (0.9742, 0.9701)}
    for c, best, mean, lo, hi in summary:
        a237, a225 = ref[c]
        print(f"  {c:<5} 深网络 最佳 {best:.4f} / 均值 {mean:.4f}   vs  HistGB237 {a237:.4f}"
              f"   HistGB225 {a225:.4f}   Δ237 最佳 {best - a237:+.4f} / 均值 {mean - a237:+.4f}")
    print(f"\n总用时 {time.time() - t0:.0f}s   峰值 {rss()}")


def default_mask_y(c: str, Ycol: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """默认段标签：与 src.feature.default_mask 同一判据（isclose 到本目标的 DEFAULT_VALUES）。"""
    from src.utils import DEFAULT_VALUES
    out = np.zeros(len(Ycol), np.int8)
    out[valid & np.isclose(Ycol, DEFAULT_VALUES[c])] = 1
    return out


if __name__ == "__main__":
    main()
