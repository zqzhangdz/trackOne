"""分段级决策诊断：真段整体被误判的比例，以及"段级池化 p_def"能否修复.

复用 screen 缓存与 TCN 工件（纯 numpy）。不动模型，只看决策层 headroom。
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
    """返回 [(start,end)) 的 True 连续段."""
    d=np.diff(np.concatenate([[0],mask.astype(np.int8),[0]]))
    st=np.flatnonzero(d==1); en=np.flatnonzero(d==-1)
    return list(zip(st,en))

def main():
    wells=load_wells("./train",has_labels=True);ids=[w["logId"].iloc[0] for w in wells]
    va=[w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df=pd.concat(va,ignore_index=True);Yva=make_targets(va_df);tdef=default_mask(Yva)
    lid=va_df["logId"].to_numpy()
    cache=joblib.load(os.path.join(RUNS,"screen_tree_cache.joblib"));p_tree=cache["p_def"]
    tags=["T6_long","T8_seed43","T9_seed44"];pby={}
    for tag in tags:
        art=joblib.load(os.path.join(RUNS,f"tcn_{tag}.joblib"));pp={}
        for w in va:
            Xw=make_features(w,context=False,priors=True).fillna(art["medians"])
            Xw=((Xw-art["stds"].replace(0,1.0))/art["stds"].replace(0,1.0)).to_numpy(np.float32)
            o=numpy_forward(Xw,art["sd"],art["dilations"])
            for c in TARGETS: pp.setdefault(c,[]).append(1/(1+np.exp(-o[c][:,0])))
        pby[tag]={c:np.concatenate(v) for c,v in pp.items()}
    pm={c:.5*p_tree[c]+.5*np.mean([pby[t][c] for t in tags],axis=0) for c in TARGETS}
    TH={"POR":.10,"PERM":.05,"SW":.05}
    print("=== 真段整体误判统计（真 real 连续段, 按井）===")
    for c in TARGETS:
        y=Yva[c].to_numpy(); m=~np.isnan(y); isreal=(~tdef[c].to_numpy())&m
        nseg=ntot=0; lens=[]; pool_p=[]; called_def=0
        for w in pd.unique(lid):
            wm=lid==w
            for s,e in runs_of(isreal&wm):
                nseg+=1; lens.append(e-s)
                pp=pm[c][s:e]
                mean_p=pp.mean()
                pool_p.append(mean_p)
                if mean_p>=TH[c]: called_def+=1
                ntot+=1
        lens=np.array(lens); pool_p=np.array(pool_p)
        print(f"  {c}: real段数={ntot} 段长中位={np.median(lens):.0f} 点 | 段级池化 p_def>=th 的段数={called_def} ({called_def/ntot*100:.1f}%)")
        if ntot:
            print(f"       段级 p_def 分位: p10={np.percentile(pool_p,10):.3f} p25={np.percentile(pool_p,25):.3f} p50={np.percentile(pool_p,50):.3f} p75={np.percentile(pool_p,75):.3f} p90={np.percentile(pool_p,90):.3f}")
    print("\n=== 默认段整体误判统计 ===")
    for c in TARGETS:
        y=Yva[c].to_numpy(); m=~np.isnan(y); isdef_=(tdef[c].to_numpy())&m
        ntot=0; called_real=0; pool=[]
        for w in pd.unique(lid):
            wm=lid==w
            for s,e in runs_of(isdef_&wm):
                ntot+=1; mp=pm[c][s:e].mean(); pool.append(mp)
                if mp<TH[c]: called_real+=1
        pool=np.array(pool)
        print(f"  {c}: default段数={ntot} 池化 p_def<th 的段数={called_real} ({called_real/max(1,ntot)*100:.1f}%) | 中位p={np.median(pool):.3f}")
    print("\n=== 当前 point-wise 决策的 far-error 段（真 real 段内 >50% 点判 default）===")
    for c in TARGETS:
        y=Yva[c].to_numpy(); m=~np.isnan(y); isreal=(~tdef[c].to_numpy())&m
        called_def=pm[c]>=TH[c]
        nfar=0; nseg=0
        for w in pd.unique(lid):
            wm=lid==w
            for s,e in runs_of(isreal&wm):
                nseg+=1
                if called_def[s:e].mean()>0.5: nfar+=1
        print(f"  {c}: {nfar}/{nseg} 段整体被误判为 default")

if __name__=="__main__": main()
