"""检验逐井(transductive) p_def 归一/校准是否提升决策（复用缓存, 纯 numpy）.

动机: 若 p_def 存在井级偏移, 逐井 z-score/分位归一后应改善阈值决策。
只使用无标签输入信息(井内分布), 属允许的 transductive 变换。
"""
from __future__ import annotations
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask, load_wells, make_features, make_targets
from src.utils import DEFAULT_VALUES, EPS, TARGETS, TOLERANCE, WEIGHTS
from solution.tcn_model import numpy_forward
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

def pointwise(c, pred, true):
    p=np.asarray(pred,float); t=np.asarray(true,float)
    if c=="PERM": p=np.log10(np.maximum(p,EPS)); t=np.log10(np.maximum(t,EPS))
    return np.maximum(0.0,1.0-np.abs(p-t)/(TOLERANCE[c]*(np.abs(t)+EPS)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tags",default="T6_long,T8_seed43,T9_seed44")
    ap.add_argument("--wtcn",default="0.60,0.95,0.75")
    ap.add_argument("--clf_w",type=float,default=0.5)
    a=ap.parse_args()
    tags=[t.strip() for t in a.tags.split(",") if t.strip()]
    wmap=dict(zip(TARGETS,[float(x) for x in a.wtcn.split(",")]))
    wells=load_wells("./train",has_labels=True); ids=[w["logId"].iloc[0] for w in wells]
    va=[w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df=pd.concat(va,ignore_index=True); Yva=make_targets(va_df); tdef=default_mask(Yva)
    cache=joblib.load(os.path.join(RUNS,"screen_tree_cache.joblib"))
    reg_tree,p_tree=cache["reg"],cache["p_def"]
    rby,pby={},{}
    for tag in tags:
        art=joblib.load(os.path.join(RUNS,f"tcn_{tag}.joblib")); r,pp={},{}
        for w in va:
            Xw=make_features(w,context=False,priors=True).fillna(art["medians"])
            Xw=((Xw-art["stds"].replace(0,1.0))/art["stds"].replace(0,1.0)).to_numpy(np.float32)
            o=numpy_forward(Xw,art["sd"],art["dilations"])
            for c in TARGETS:
                r.setdefault(c,[]).append(o[c][:,1]); pp.setdefault(c,[]).append(1/(1+np.exp(-o[c][:,0])))
        rby[tag]={c:np.concatenate(v) for c,v in r.items()}; pby[tag]={c:np.concatenate(v) for c,v in pp.items()}
    ens={c:np.mean([rby[t][c] for t in tags],axis=0) for c in TARGETS}
    regs={c:(1-wmap[c])*reg_tree[c]+wmap[c]*ens[c] for c in TARGETS}
    wit=va_df["logId"].to_numpy()
    th_grid=np.arange(0.02,0.71,0.02)

    def pmix(c):
        return (1-a.clf_w)*p_tree[c]+a.clf_w*np.mean([pby[t][c] for t in tags],axis=0)

    def ev(name,pmap):
        tot=0.0; parts={}
        for c in TARGETS:
            y=Yva[c].to_numpy(); m=~np.isnan(y); ba,bt=-1,0.5
            for t in th_grid:
                pred=np.where(pmap[c]>=t,DEFAULT_VALUES[c],np.power(10.0,regs[c]))
                acc=pointwise(c,pred[m],y[m]).mean()
                if acc>ba: ba,bt=acc,float(t)
            pred=np.where(pmap[c]>=bt,DEFAULT_VALUES[c],np.power(10.0,regs[c]))
            d=pointwise(c,pred[m&tdef[c].to_numpy()],y[m&tdef[c].to_numpy()]).mean()
            r=pointwise(c,pred[m&~tdef[c].to_numpy()],y[m&~tdef[c].to_numpy()]).mean()
            parts[c]=(ba,r,d,bt); tot+=WEIGHTS[c]*ba
        print(f"{name:26s}: {tot*100:.3f} acc=(" + ",".join(f"{parts[c][0]:.4f}" for c in TARGETS)
              + ") real=(" + ",".join(f"{parts[c][1]:.4f}" for c in TARGETS)
              + ") def=(" + ",".join(f"{parts[c][2]:.4f}" for c in TARGETS)
              + ") th=(" + ",".join(f"{parts[c][3]:.2f}" for c in TARGETS) + ")")
        return tot*100

    base={c:pmix(c) for c in TARGETS}
    ev("raw prob-mean", base)

    # 变体 1: 逐井 z-score
    z={}
    for c in TARGETS:
        v=base[c].copy()
        for lid in pd.unique(wit):
            m=wit==lid; mu=v[m].mean(); sd=v[m].std()+1e-9; v[m]=(v[m]-mu)/sd
        z[c]=v
    ev("per-well z-score", z)

    # 变体 2: 逐井 rank(经验分位)
    for c in TARGETS:
        v=base[c].copy()
        for lid in pd.unique(wit):
            m=wit==lid; s=pd.Series(v[m]); v[m]=s.rank(pct=True).to_numpy()
        z[c]=v
    ev("per-well pct-rank", z)

    # 变体 3: 逐井减去中位数
    for c in TARGETS:
        v=base[c].copy()
        for lid in pd.unique(wit):
            m=wit==lid; v[m]=v[m]-np.median(v[m])
        z[c]=v
    ev("per-well median-shift", z)

if __name__=="__main__":
    main()
