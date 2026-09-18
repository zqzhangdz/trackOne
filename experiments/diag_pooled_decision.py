"""段级池化决策规则：在"预测调用"形成的连续段上池化 p_def 后重新决策.

可在测试时计算的决策层规则（不依赖真值分段）：
 1) 由 point-wise 阈值调用得到 0/1 序列
 2) 取连通段，段内用 p_def 的均值/中位数重新判定该段是否为 default
 3) 参数: 池化统计量 + 段级阈值
对比 point-wise 基线，看决策层 headroom。
纯 numpy + joblib。
"""
from __future__ import annotations
import os,sys,warnings
warnings.filterwarnings("ignore")
import joblib,numpy as np,pandas as pd
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature import default_mask,load_wells,make_features,make_targets
from src.utils import DEFAULT_VALUES,EPS,TARGETS,TOLERANCE,WEIGHTS
from solution.tcn_model import numpy_forward
RUNS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"runs")

def pw(c,pred,true):
    p=np.asarray(pred,float);t=np.asarray(true,float)
    if c=="PERM": p=np.log10(np.maximum(p,EPS));t=np.log10(np.maximum(t,EPS))
    return np.maximum(0.,1-np.abs(p-t)/(TOLERANCE[c]*(np.abs(t)+EPS)))

def runs_of(mask):
    d=np.diff(np.concatenate([[0],mask.astype(np.int8),[0]]))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))

def main():
    wells=load_wells("./train",has_labels=True);ids=[w["logId"].iloc[0] for w in wells]
    va=[w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df=pd.concat(va,ignore_index=True);Yva=make_targets(va_df);tdef=default_mask(Yva)
    lid=va_df["logId"].to_numpy()
    cache=joblib.load(os.path.join(RUNS,"screen_tree_cache.joblib"));reg_tree,p_tree=cache["reg"],cache["p_def"]
    tags=["T6_long","T8_seed43","T9_seed44"];rby,pby={},{}
    for tag in tags:
        art=joblib.load(os.path.join(RUNS,f"tcn_{tag}.joblib"));r,pp={},{}
        for w in va:
            Xw=make_features(w,context=False,priors=True).fillna(art["medians"])
            Xw=((Xw-art["stds"].replace(0,1.0))/art["stds"].replace(0,1.0)).to_numpy(np.float32)
            o=numpy_forward(Xw,art["sd"],art["dilations"])
            for c in TARGETS:
                r.setdefault(c,[]).append(o[c][:,1]);pp.setdefault(c,[]).append(1/(1+np.exp(-o[c][:,0])))
        rby[tag]={c:np.concatenate(v) for c,v in r.items()};pby[tag]={c:np.concatenate(v) for c,v in pp.items()}
    ens={c:np.mean([rby[t][c] for t in tags],axis=0) for c in TARGETS}
    wmap=dict(zip(TARGETS,[.60,.95,.75]))
    regs={c:(1-wmap[c])*reg_tree[c]+wmap[c]*ens[c] for c in TARGETS}
    pm={c:.5*p_tree[c]+.5*np.mean([pby[t][c] for t in tags],axis=0) for c in TARGETS}
    BASE_TH={"POR":.10,"PERM":.05,"SW":.05}

    def score(c, calls):
        y=Yva[c].to_numpy();m=~np.isnan(y)
        pred=np.where(calls,DEFAULT_VALUES[c],np.power(10.,regs[c]))
        return pw(c,pred[m],y[m]).mean()

    base_tot=sum(WEIGHTS[c]*score(c,pm[c]>=BASE_TH[c]) for c in TARGETS)*100
    print(f"point-wise 基线 = {base_tot:.3f}")

    def pooled(c, base_th, pool_th, how="mean", min_len=1, max_len=10**9):
        calls=pm[c]>=base_th
        out=calls.copy()
        for w in pd.unique(lid):
            wm=lid==w
            idx=np.flatnonzero(wm)
            for s,e in runs_of(calls[wm]):
                if e-s<min_len or e-s>max_len: continue
                v=pm[c][idx[s:e]]
                stat=v.mean() if how=="mean" else np.median(v)
                out[idx[s:e]]=stat>=pool_th
        return out

    print("\n=== 池化决策：在 calls 连续段上重判 ===")
    for how in ["mean","median"]:
        best=(-1,None)
        for bt in np.arange(.05,.51,.05):
            for pt in np.arange(.05,.71,.05):
                t=sum(WEIGHTS[c]*score(c,pooled(c,bt,pt,how)) for c in TARGETS)*100
                if t>best[0]: best=(t,(round(bt,2),round(pt,2)))
        print(f"  {how}: best={best[0]:.3f} @(base_th,pool_th)={best[1]}")
    # 只看长段（避免碎片抖动）
    print("\n=== 仅对长度>=L 的段池化 (mean) ===")
    for L in [3,5,10,20,40]:
        best=(-1,None)
        for bt in np.arange(.05,.51,.05):
            for pt in np.arange(.05,.71,.05):
                t=sum(WEIGHTS[c]*score(c,pooled(c,bt,pt,"mean",min_len=L)) for c in TARGETS)*100
                if t>best[0]: best=(t,(round(bt,2),round(pt,2)))
        print(f"  L>={L}: best={best[0]:.3f} @{best[1]}")

if __name__=="__main__": main()
