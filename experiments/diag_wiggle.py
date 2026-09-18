"""段内 wiggle 可预测性诊断 + 自回归追踪对水平偏置的敏感性."""
from __future__ import annotations
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import load_wells, make_features, make_targets, default_mask, transform_target
from src.utils import TARGETS, TOLERANCE, EPS

def pw(c, p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def segments(mask):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2]))

LOGCOLS = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "logRT", "logRXO"]

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr_w = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va_w = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]

    def collect(wells_list):
        out = {c: {"dy": [], "dx": [], "y": [], "segmean": []} for c in TARGETS}
        for w in wells_list:
            Y = make_targets(w); isd = default_mask(Y)
            X = make_features(w, context=False, priors=True)
            for c in TARGETS:
                y = Y[c].to_numpy(); valid = ~np.isnan(y)
                real = valid & ~isd[c].to_numpy()
                t = transform_target(c, pd.Series(np.where(valid, y, np.nan))).to_numpy()
                xl = X[LOGCOLS].to_numpy()
                for (s, e) in segments(real):
                    if e - s < 5:
                        continue
                    mt = t[s:e].mean()
                    mx = np.nanmean(xl[s:e], axis=0)
                    out[c]["dy"].append(t[s:e] - mt)
                    out[c]["dx"].append(np.nan_to_num(xl[s:e] - mx))
                    out[c]["y"].append(y[s:e])
                    out[c]["segmean"].append(np.full(e - s, mt))
        return out

    tr = collect(tr_w); va = collect(va_w)
    for c in TARGETS:
        dy = np.concatenate(tr[c]["dy"]); dx = np.concatenate(tr[c]["dx"])
        # 段内(去段均值)相关
        cors = {col: float(np.corrcoef(dx[:, i], dy)[0, 1]) for i, col in enumerate(LOGCOLS)}
        top = sorted(cors.items(), key=lambda kv: -abs(kv[1]))[:4]
        # 线性回归 wiggle~logs 的全局系数
        A = np.hstack([dx, np.ones((len(dx), 1))])
        coef, *_ = np.linalg.lstsq(A, dy, rcond=None)
        r2 = 1 - np.sum((A @ coef - dy) ** 2) / np.sum(dy ** 2)
        # 验证集 oracle: 真段均值 + 线性 wiggle
        dva = np.concatenate(va[c]["dy"]); dxa = np.concatenate(va[c]["dx"])
        pred_wig = np.hstack([dxa, np.ones((len(dxa), 1))]) @ coef
        yv = np.concatenate(va[c]["y"]); sm = np.concatenate(va[c]["segmean"])
        # 在物理空间合成: transform 空间相加后反变换
        from src.feature import inverse_transform_target
        pred_t = sm_trans = None
        # segmean 是 transform 空间; y 是物理空间; 直接对 transform 空间算 pw 需物理值: 反变换
        pred_phys = inverse_transform_target(c, pd.Series(sm + pred_wig)).to_numpy()
        acc_lw = pw(c, pred_phys, yv).mean()
        acc_sm = pw(c, inverse_transform_target(c, pd.Series(sm)).to_numpy(), yv).mean()
        # 自回归追踪敏感性: ŷ[t]=y[t-1] 但段首注入偏置 b (transform 空间)
        print(f"{c}: 段内wiggle R2(logs线性)={r2:.3f} top相关={top}")
        print(f"   oracle: 段均值={acc_sm:.4f}  段均值+线性wiggle={acc_lw:.4f}")
        for b in (0.0, 0.25, 0.5, 1.0):
            accs = []
            for yseg, tseg in zip(va[c]["y"], []):
                pass
            # 逐段模拟: 需要 transform 空间真值序列
            break
        # 逐段 AR 敏感性（用 transform 空间真值序列）
        tot = {b: [] for b in (0.0, 0.1, 0.2, 0.4)}
        # 重新收集 transform 真值序列
        for w in va_w:
            Y = make_targets(w); isd = default_mask(Y)
            y = Y[c].to_numpy(); valid = ~np.isnan(y)
            real = valid & ~isd[c].to_numpy()
            t = transform_target(c, pd.Series(np.where(valid, y, np.nan))).to_numpy()
            for (s, e) in segments(real):
                if e - s < 5:
                    continue
                ts = t[s:e]; ys = y[s:e]
                for b in tot:
                    pred = np.empty_like(ts)
                    pred[0] = ts[0] + b * np.sign(np.random.default_rng(0).standard_normal())
                    pred[1:] = ts[:-1] + b * 0.0  # 段内纯持久性, 仅段首偏置 b
                    # 偏置持续: pred[t] = pred[t-1] + (ts[t]-ts[t-1]) 等价 ts[t]+b
                    pred = ts + b
                    tot[b].append(pw(c, inverse_transform_target(c, pd.Series(pred)).to_numpy(), ys).mean() * (e - s))
        line = " ".join(f"b={b}: {sum(v) / sum(1 for _ in v) if v else 0:.3f}" for b, v in tot.items())
        ns = sum(len(v) for v in tot.values()) // len(tot)
        line = " ".join(f"b={b}: {sum(v) / max(1, ns):.4f}" for b, v in tot.items())
        print(f"   持久性+段首偏置b(transform): " + line)

if __name__ == "__main__":
    main()
