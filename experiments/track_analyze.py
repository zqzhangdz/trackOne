"""阶段 2: 截断非凸指标下的集成决策分析 (读 track_members.npz).

A 聚合规则  B 仿射校准 (LOO)  C 系统性偏差  D 代理路由  E 成员分歧/神谕上限
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

RUNS = os.path.dirname(os.path.abspath(__file__))
D = np.load(os.path.join(RUNS, "runs", "track_members.npz"), allow_pickle=True)
EPS = 1e-5
TARGETS = ["POR", "PERM", "SW"]
TOL = {"POR": 0.08, "PERM": None, "SW": 0.05}
WTS = {"POR": [0.5, 1/6, 1/6, 1/6], "PERM": [0.5, 1/6, 1/6, 1/6], "SW": [0.25]*4}
MEMN = ["tree", "T15s42", "T16s43", "T16s44"]
DMV = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}
va_ids = np.unique(D["well_of"])

M = {c: D[f"M__{c}"] for c in TARGETS}
y = {c: D[f"y__{c}"] for c in TARGETS}
res = {c: D[f"res__{c}"] for c in TARGETS}
valid = {c: ~np.isnan(y[c]) for c in TARGETS}
well_of = D["well_of"]


def acc_vec(c, pred):
    p, t = pred, y[c]
    if c == "PERM":
        return np.maximum(0.0, 1.0 - np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS))))
    return np.maximum(0.0, 1.0 - np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS)))


def rel_err(c, pred):
    p, t = pred, y[c]
    if c == "PERM":
        return np.abs(np.log10(np.maximum(p / np.maximum(t, EPS), EPS)))
    return np.abs(p - t) / (TOL[c] * (np.abs(t) + EPS))


def tot(c, pred_phys):
    a = np.where(valid[c], acc_vec(c, np.where(valid[c], pred_phys, 1.0)), np.nan)
    m = valid[c]
    return np.nanmean(a[m])


def rep(c, pred_phys, label):
    a = acc_vec(c, np.where(valid[c], pred_phys, 1.0))
    m, r = valid[c], res[c]
    return f"  {c:5s} {label:26s} total={np.nanmean(a[m]):.4f}  reservoir={a[r].mean():.4f}"


print("=" * 100)
print("A. 聚合规则 (无拟合参数; 储层段用真值掩码 → 隔离回归器, 与分段器无关)")
print("=" * 100)
base = {}
for c in TARGETS:
    w = np.array(WTS[c]); bl = w @ M[c]
    base[c] = bl
    b = np.where(res[c], np.power(10.0, bl), DMV[c])
    a = acc_vec(c, b); print(rep(c, b, "凸平均(部署权重)"))
    e = rel_err(c, np.power(10.0, bl))[res[c]]
    print(f"        带内率={float((e<1).mean()):.4f}  带内均分={acc_vec(c,np.power(10.0,bl))[res[c]][e<1].mean():.4f}")
    for name, agg in [("等权平均4", lambda X: X.mean(0)), ("中位数4", lambda X: np.median(X, 0)),
                      ("上界(max)4", lambda X: np.max(X, 0)), ("下界(min)4", lambda X: np.min(X, 0)),
                      ("截尾均值(p10-p90)", lambda X: np.mean([np.clip(X[:, i], np.quantile(X[:, i], .1, 0), np.quantile(X[:, i], .9, 0)) for i in range(X.shape[0])], 0))]:
        print(rep(c, np.where(res[c], np.power(10.0, agg(M[c])), DMV[c]), name))

print("\n" + "=" * 100)
print("B. 逐目标 log10 仿射校准  pred_cal = 10^(a + b*blend)   [留一井拟合, 测试井绝不参与拟合]")
print("=" * 100)
for c in TARGETS:
    bl = base[c]
    b0 = np.where(res[c], np.power(10.0, bl), DMV[c])
    a0 = acc_vec(c, b0); r0, t0 = a0[res[c]].mean(), np.nanmean(a0[valid[c]])
    ty = np.log10(np.maximum(y[c], EPS))
    print(f"  {c:5s} 基线 total={t0:.4f}  reservoir={r0:.4f}   带半宽(log10)={np.log10(1+TOL[c]) if TOL[c] else 1.0:.4f}")
    for vname, sel, use_robust in [
        ("全储层 OLS", res[c], False),
        ("全储层 截尾20%", res[c], True),
        ("带内(<1.0) OLS", res[c] & (rel_err(c, np.power(10.0, bl)) < 1.0), False),
        ("|比值|<10 稳健", res[c] & (rel_err(c, np.power(10.0, bl)) < 2.0), False),
        ("b=1 中位比值", res[c], "b1"),
    ]:
        per_well, folds = [], []
        for iid in va_ids:
            mw = well_of == iid
            fi = sel & ~mw
            if vname == "b=1 中位比值":
                rr = np.power(10.0, bl)[fi] / np.maximum(y[c][fi], EPS)
                a, b = np.log10(np.median(rr)), 1.0
            else:
                ff = fi
                if use_robust:
                    er = rel_err(c, np.power(10.0, bl))[ff]
                    ff = ff & (er < np.quantile(er, 0.8))
                if ff.sum() < 30: continue
                a, b = np.polyfit(bl[ff], ty[ff], 1)
            pred = np.where(res[c], np.power(10.0, a + b * bl), DMV[c])
            aa = acc_vec(c, pred)
            per_well.append((aa[mw].mean() - a0[mw].mean(), aa[res[c] & mw].mean() - a0[res[c] & mw].mean()))
            folds.append((a, b))
        if not per_well: continue
        dw = np.array(per_well)
        avg_r = np.mean([p[1] for p in per_well]); avg_t = np.mean([p[0] for p in per_well])
        print(f"      {vname:16s} Δtotal={avg_t:+.4f}  Δreservoir={avg_r:+.4f}"
              f"  逐井Δtotal=[{dw[:,0].min():+.4f},{dw[:,0].max():+.4f}] 正号{int((dw[:,0]>0).sum())}/8"
              f"  a=[{min(f[0] for f in folds):+.4f},{max(f[0] for f in folds):+.4f}]"
              f"  b=[{min(f[1] for f in folds):.4f},{max(f[1] for f in folds):.4f}]")

print("\n" + "=" * 100)
print("C. 系统性乘性偏差 (预测/真值, 真值储层段)")
print("=" * 100)
for c in TARGETS:
    bl = base[c]
    rr = np.power(10.0, bl)[res[c]] / np.maximum(y[c][res[c]], EPS)
    e = rel_err(c, np.power(10.0, bl))[res[c]]
    print(f"  {c:5s} pred/true 中位={np.median(rr):.4f} p25/p75={np.quantile(rr,.25):.3f}/{np.quantile(rr,.75):.3f}"
          f" p10/p90={np.quantile(rr,.1):.3f}/{np.quantile(rr,.9):.3f}")
    print(f"        |归一化误差| 中位={np.median(e):.4f} 均值={e.mean():.4f}"
          f"  带内(<1)占比={float((e<1).mean()):.4f}"
          f"  带内均分={acc_vec(c,np.power(10.0,bl))[res[c]][e<1].mean():.4f}")
    # 若把 |log10 ratio| 全部压到中位以下 (不可达参考), reservoir acc 上限
    print(f"        [参考] 若所有点 |误差|<=中位值 {np.median(e):.3f} → reservoir acc>={1-np.median(e):.4f}")

print("\n" + "=" * 100)
print("D. 代理中心 best-of-N 路由 (无拟合参数)")
print("=" * 100)
for c in TARGETS:
    ref = acc_vec(c, np.where(res[c], np.power(10.0, base[c]), DMV[c]))
    rref = ref[res[c]].mean()
    for pn, pr in {"tree": M[c][0], "TCN3均值": M[c][1:].mean(0), "中位4": np.median(M[c], 0),
                   "凸平均": base[c], "等权均值4": M[c].mean(0)}.items():
        j = np.argmin(np.abs(M[c] - pr[None, :]), 0)
        pred = np.where(res[c], np.power(10.0, M[c][j]), DMV[c])
        aa = acc_vec(c, pred)
        print(f"  {c:5s} proxy={pn:8s} reservoir={aa[res[c]].mean():.4f} Δtotal={np.nanmean(aa[valid[c]])-np.nanmean(ref[valid[c]]):+.4f}"
              f"  选中频率={[(float((j==k).mean())) for k in range(4)]}")

print("\n" + "=" * 100)
print("E. 成员分歧度 与 成员池内神谕上限")
print("=" * 100)
for c in TARGETS:
    e = rel_err(c, np.power(10.0, M[c]))          # (4,N)
    sp = (e.max(0) - e.min(0))[res[c]]
    sc = np.maximum(0.0, 1.0 - e)
    oracle = sc.max(0)
    bacc = acc_vec(c, np.power(10.0, base[c]))
    print(f"  {c:5s} 成员误差 spread: 中位={np.median(sp):.4f} p90={np.quantile(sp,.9):.4f} p99={np.quantile(sp,.99):.4f}")
    print(f"        混叠点(spread>1,至少1成员可进带)={float((sp>1).mean()):.4f}"
          f"  4成员全带外={float(np.all(e>=1,0)[res[c]].mean()):.4f}"
          f"  至少1成员带内={float((~np.all(e>=1,0))[res[c]].mean()):.4f}")
    print(f"        [神谕成员选择] reservoir acc={oracle[res[c]].mean():.4f}  vs 凸平均={bacc[res[c]].mean():.4f}"
          f"  → 成员池内理论上限 Δ={oracle[res[c]].mean()-bacc[res[c]].mean():+.4f}"
          f"  (=+{(oracle[res[c]].mean()-bacc[res[c]].mean())*res[c].mean()* (0.30 if c=='POR' else 0.35)*100:+.4f} 总分)")
