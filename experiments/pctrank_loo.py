"""LOO 验证逐井 pct-rank 决策：阈值在 7 井调优、留出 1 井评估（纯 numpy）."""
from __future__ import annotations
import os,sys,warnings,argparse
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
    return np.maximum(0.0,1.0-np.abs(p-t)/(TOLERANCE[c]*(np.abs(t)+EPS)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tags",default="T6_long,T8_seed43,T9_seed44")
    ap.add_argument("--wtcn",default="0.60,0.95,0.75");ap.add_argument("--clf_w",type=float,default=0.5)
    a=ap.parse_args()
    tags=[t.strip() for t in a.tags.split(",") if t.strip()]
    wmap=dict(zip(TARGETS,[float(x) for x in a.wtcn.split(",")]))
    wells=load_wells("./train",has_labels=True);ids=[w["logId"].iloc[0] for w in wells]
    va=[w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df=pd.concat(va,ignore_index=True);Yva=make_targets(va_df);tdef=default_mask(Yva)
    cache=joblib.load(os.path.join(RUNS,"screen_tree_cache.joblib"));reg_tree,p_tree=cache["reg"],cache["p_def"]
    rby,pby={},{}
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
    regs={c:(1-wmap[c])*reg_tree[c]+wmap[c]*ens[c] for c in TARGETS}
    wit=va_df["logId"].to_numpy();wl=list(dict.fromkeys(wit.tolist()))
    base={c:(1-a.clf_w)*p_tree[c]+a.clf_w*np.mean([pby[t][c] for t in tags],axis=0) for c in TARGETS}
    # pct-rank 变换
    pr={}
    for c in TARGETS:
        v=base[c].copy()
        for lid in wl:
            m=wit==lid; v[m]=pd.Series(v[m]).rank(pct=True).to_numpy()
        pr[c]=v
    th_grid=np.arange(0.05,0.96,0.05)

    def acc_per_well(pmap,c):
        y=Yva[c].to_numpy();m=~np.isnan(y)
        pred=np.where(pmap[c]>=0.5,DEFAULT_VALUES[c],np.power(10.0,regs[c]))
        a_=pw(c,pred[m],y[m]);w=wit[m]
        return {lid:a_[w==lid].mean() for lid in wl if (w==lid).any()}

    def score_cfg(pmap,c,t):
        y=Yva[c].to_numpy();m=~np.isnan(y)
        pred=np.where(pmap[c]>=t,DEFAULT_VALUES[c],np.power(10.0,regs[c]))
        a_=pw(c,pred[m],y[m]);w=wit[m]
        return {lid:a_[w==lid].mean() for lid in wl if (w==lid).any()}

    for name,pmap in [("raw prob-mean",base),("per-well pct-rank",pr)]:
        best={}
        for c in TARGETS:
            y=Yva[c].to_numpy();m=~np.isnan(y);ba,bt=-1,0.5
            for t in th_grid:
                v=score_cfg(pmap,c,t); s=np.mean(list(v.values()))
                if s>ba: ba,bt=s,float(t)
            best[c]=(bt,ba)
        tot=0
        for c in TARGETS:
            y=Yva[c].to_numpy();m=~np.isnan(y)
            pred=np.where(pmap[c]>=best[c][0],DEFAULT_VALUES[c],np.power(10.0,regs[c]))
            tot+=WEIGHTS[c]*pw(c,pred[m],y[m]).mean()
        # LOO
        loo=[]
        for h in wl:
            rest=[x for x in wl if x!=h]; tsum=0
            for c in TARGETS:
                y=Yva[c].to_numpy();m=~np.isnan(y);ba,bt=-1,0.5
                for t in th_grid:
                    pred=np.where(pmap[c]>=t,DEFAULT_VALUES[c],np.power(10.0,regs[c]))
                    a_=pw(c,pred[m],y[m]);w=wit[m]
                    vv={lid:a_[w==lid].mean() for lid in wl if (w==lid).any()}
                    s=np.mean([vv[x] for x in rest if x in vv])
                    if s>ba: ba,bt=s,float(t)
                pred=np.where(pmap[c]>=bt,DEFAULT_VALUES[c],np.power(10.0,regs[c]))
                a_=pw(c,pred[m],y[m]);w=wit[m]
                tsum+=WEIGHTS[c]*(a_[w==h].mean() if (w==h).any() else np.nan)
            loo.append(tsum*100)
        loo=np.array(loo)
        print(f"{name:20s}: insample={tot*100:.3f} LOO={np.nanmean(loo):.3f} ± {np.nanstd(loo):.3f} th={ {c:round(best[c][0],2) for c in TARGETS} }")
        for i,h in enumerate(wl):
            print(f"      {h[:8]}: {loo[i]:.3f}")

if __name__=="__main__": main()
