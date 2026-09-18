"""验证：逐目标阈值的单调校准（isotonic/temperature）是否等价于不改动。

若决策是 "p>=t -> 默认"，则任何严格单调变换 g 都只是重参数化阈值。
本脚本用 isotonic 回归 + 温度缩放实测 screen 分数，验证是否无增益。
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import joblib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS

def pointwise(c, pred, true):
    p = np.asarray(pred, float); t = np.asarray(true, float)
    if c == "PERM":
        p = np.log10(np.maximum(p, EPS)); t = np.log10(np.maximum(t, EPS))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOLERANCE[c] * (np.abs(t) + EPS)))

def main():
    wells = load_wells("./train", has_labels=True)
    ids = [w["logId"].iloc[0] for w in wells]
    tr = [w for w in wells if w["logId"].iloc[0] in set(ids[:72])]
    va = [w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    tr_df = __import__("pandas").concat(tr, ignore_index=True)
    va_df = __import__("pandas").concat(va, ignore_index=True)
    Ytr, Yva = make_targets(tr_df), make_targets(va_df)
    cache = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "screen_tree_cache.joblib"))
    reg, p_def = cache["reg"], cache["p_def"]
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    grid = np.arange(0.01, 0.91, 0.01)
    def evalp(pm):
        tot = 0.0; info = []
        for c in TARGETS:
            y = Yva[c].to_numpy(); m = ~np.isnan(y)
            bt, ba = 0.5, -1
            for t in grid:
                pred = np.where(pm[c] >= t, DEFAULT_VALUES[c], np.power(10.0, reg[c]))
                a = pointwise(c, pred[m], y[m]).mean()
                if a > ba: ba, bt = a, float(t)
            tot += WEIGHTS[c]*ba; info.append((c, ba, bt))
        return tot*100, info
    t0, i0 = evalp(p_def)
    print(f"raw               : {t0:.4f} " + " ".join(f"{c}:{a:.4f}@t={t:.2f}" for c, a, t in i0))
    # 温度缩放：在训练集上拟合
    Xtrp = __import__("pandas").concat(tr, ignore_index=True)
    from src.feature import make_features
    Xt = make_features(tr_df, context=True, priors=True); Xv = make_features(va_df, context=True, priors=True)
    from src.model import TwoStageModel
    # 用缓存中的 p_def 无法拿到训练概率；改为直接读阈值敏感性：对 p 做幂变换
    for a in (0.5, 0.7, 1.5, 2.0):
        pm = {c: np.clip(p_def[c], 1e-6, 1-1e-6) ** a for c in TARGETS}
        t, i = evalp(pm)
        print(f"p^{a:<4}            : {t:.4f} " + " ".join(f"{c}:{aa:.4f}@t={tt:.2f}" for c, aa, tt in i))
    for a in (2.0, 5.0, 10.0):
        pm = {c: 1/(1+np.exp(-a*(np.clip(p_def[c],1e-6,1-1e-6)*2-1))) for c in TARGETS}
        t, i = evalp(pm)
        print(f"temp T=1/{a:<4}       : {t:.4f} " + " ".join(f"{c}:{aa:.4f}@t={tt:.2f}" for c, aa, tt in i))

if __name__ == "__main__":
    main()
