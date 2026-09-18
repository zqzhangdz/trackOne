"""对比不同 seed 组合在 v4 融合协议(逐目标权重 + clf 概率融合)下的 screen 总分."""
from __future__ import annotations
import os,sys,warnings,argparse,itertools
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

ALL=["T6_long","T8_seed43","T9_seed44","T10_ln10"]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--combos",default="T6_long,T8_seed43|T6_long,T10_ln10|T8_seed43,T10_ln10|T6_long,T8_seed43,T9_seed44|T6_long,T8_seed43,T10_ln10|T6_long,T9_seed44,T10_ln10|T6_long,T8_seed43,T9_seed44,T10_ln10")
    ap.add_argument("--clf_w",type=float,default=0.5)
    a=ap.parse_args()
    wells=load_wells("./train",has_labels=True);ids=[w["logId"].iloc[0] for w in wells]
    va=[w for w in wells if w["logId"].iloc[0] in set(ids[72:])]
    va_df=pd.concat(va,ignore_index=True);Yva=make_targets(va_df);tdef=default_mask(Yva)
    cache=joblib.load(os.path.join(RUNS,"screen_tree_cache.joblib"));reg_tree,p_tree=cache["reg"],cache["p_def"]
    rby,pby={},{}
    for tag in ALL:
        fp=os.path.join(RUNS,f"tcn_{tag}.joblib")
        if not os.path.exists(fp): continue
        art=joblib.load(fp);r,pp={},{}
        for w in va:
            Xw=make_features(w,context=False,priors=True).fillna(art["medians"])
            Xw=((Xw-art["stds"].replace(0,1.0))/art["stds"].replace(0,1.0)).to_numpy(np.float32)
            o=numpy_forward(Xw,art["sd"],art["dilations"])
            for c in TARGETS:
                r.setdefault(c,[]).append(o[c][:,1]);pp.setdefault(c,[]).append(1/(1+np.exp(-o[c][:,0])))
        rby[tag]={c:np.concatenate(v) for c,v in r.items()};pby[tag]={c:np.concatenate(v) for c,v in pp.items()}
    th_grid=np.arange(.02,.71,.02);w_grid=np.arange(.0,1.01,.05)

    def eval_combo(tags):
        ens={c:np.mean([rby[t][c] for t in tags],axis=0) for c in TARGETS}
        pens={c:np.mean([pby[t][c] for t in tags],axis=0) for c in TARGETS}
        tot=0.0;parts={};cfg={}
        for c in TARGETS:
            y=Yva[c].to_numpy();m=~np.isnan(y)
            ba,bt,bw=-1,.5,.7
            for w in w_grid:
                reg=(1-w)*reg_tree[c]+w*ens[c]
                for cw in ([0.0,0.3,0.5,0.7] if a.clf_w<0 else [a.clf_w]):
                    pm=(1-cw)*p_tree[c]+cw*pens[c]
                    for t in th_grid:
                        pred=np.where(pm>=t,DEFAULT_VALUES[c],np.power(10.,reg))
                        acc=pw(c,pred[m],y[m]).mean()
                        if acc>ba: ba,bt,bw=acc,float(t),float(w)
            y2=Yva[c].to_numpy()
            reg=(1-bw)*reg_tree[c]+bw*ens[c]
            cw=a.clf_w if a.clf_w>=0 else 0.5
            pm=(1-cw)*p_tree[c]+cw*pens[c]
            pred=np.where(pm>=bt,DEFAULT_VALUES[c],np.power(10.,reg))
            d=pw(c,pred[m&tdef[c].to_numpy()],y2[m&tdef[c].to_numpy()]).mean()
            r=pw(c,pred[m&~tdef[c].to_numpy()],y2[m&~tdef[c].to_numpy()]).mean()
            parts[c]=(ba,r,d);cfg[c]=(bw,bt);tot+=WEIGHTS[c]*ba
        return tot*100,parts,cfg

    for combo in a.combos.split("|"):
        tags=[t.strip() for t in combo.split(",") if t.strip()]
        tags=[t for t in tags if t in rby]
        if not tags: continue
        t,parts,cfg=eval_combo(tags)
        print(f"{'+'.join(t.replace('_long','').replace('_seed','').replace('_ln10','') for t in tags):28s}: {t:.3f} "
              f"real=(" + ",".join(f"{parts[c][1]:.4f}" for c in TARGETS) + ") "
              f"def=(" + ",".join(f"{parts[c][2]:.4f}" for c in TARGETS) + ") "
              f"w=(" + ",".join(f"{cfg[c][0]:.2f}" for c in TARGETS) + ")")

if __name__=="__main__": main()
