"""三目标共识分段决策：默认段三目标高度重叠(Jaccard>0.85)，但当前各目标独立决策.

检验：用三目标 p_def 的共识(均值/最小/乘积)作为分段统计量，能否改善决策。
纯 numpy + joblib，不重训。
"""
from __future__ import annotations
import os,sys,warnings,itertools
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

def main():
    wells=load_wells("./train",has_labels=True);ids=[w["logId"].iloc[0] for w in wells]
    va=[w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df=pd.concat(va,ignore_index=True);Yva=make_targets(va_df);tdef=default_mask(Yva)
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
    th_grid=np.arange(.02,.71,.02)
    lgt=lambda p: np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))

    def evalmap(name, mapper):
        tot=0;parts={}
        for c in TARGETS:
            y=Yva[c].to_numpy();m=~np.isnan(y)
            pv=mapper(c)
            ba,bt=-1,.5
            for t in th_grid:
                pred=np.where(pv>=t,DEFAULT_VALUES[c],np.power(10.,regs[c]))
                a=pw(c,pred[m],y[m]).mean()
                if a>ba: ba,bt=a,float(t)
            pred=np.where(pv>=bt,DEFAULT_VALUES[c],np.power(10.,regs[c]))
            d=pw(c,pred[m&tdef[c].to_numpy()],y[m&tdef[c].to_numpy()]).mean()
            r=pw(c,pred[m&~tdef[c].to_numpy()],y[m&~tdef[c].to_numpy()]).mean()
            parts[c]=(ba,r,d,bt);tot+=WEIGHTS[c]*ba
        print(f"{name:30s}: {tot*100:.3f} real=(" + ",".join(f"{parts[c][1]:.4f}" for c in TARGETS)
              + ") def=(" + ",".join(f"{parts[c][2]:.4f}" for c in TARGETS)
              + ") th=(" + ",".join(f"{parts[c][3]:.2f}" for c in TARGETS) + ")")
        return tot*100

    evalmap("per-target (baseline)", lambda c: pm[c])
    # 共识：三目标均值/最小/几何均值，用于所有目标
    cons_mean={c:np.mean([pm[x] for x in TARGETS],axis=0) for c in TARGETS}
    evalmap("consensus mean (all targets)", lambda c: cons_mean[c])
    cons_min={c:np.min([pm[x] for x in TARGETS],axis=0) for c in TARGETS}
    evalmap("consensus min", lambda c: cons_min[c])
    cons_geo={c:np.exp(np.mean([np.log(np.clip(pm[x],1e-6,1)) for x in TARGETS],axis=0)) for c in TARGETS}
    evalmap("consensus geo-mean", lambda c: cons_geo[c])
    cons_logit={c:1/(1+np.exp(-np.mean([lgt(pm[x]) for x in TARGETS],axis=0))) for c in TARGETS}
    evalmap("consensus logit-mean", lambda c: cons_logit[c])
    # 混合：自己的 p 与共识加权
    for a in (0.3,0.5,0.7):
        mix={c:(1-a)*pm[c]+a*cons_logit[c] for c in TARGETS}
        evalmap(f"blend self/consensus {a:.1f}", lambda c,mix=mix: mix[c])

if __name__=="__main__": main()
