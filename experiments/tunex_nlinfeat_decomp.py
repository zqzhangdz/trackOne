"""locminmax101 两因子分解：Δ 来自分类器还是回归器。重拟合 clf300+reg（npz 检查点被覆盖了）。"""
import sys, os, time, gc, json, numpy as np, pandas as pd, joblib, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,'.')
from src.feature import CURVE_COLS, load_wells, make_targets, default_mask, transform_target
from src.model import POR_MONOTONIC, _clf, _reg
from src.utils import DEFAULT_VALUES, EPS, TARGETS, WEIGHTS
import src.feature as F
T0=time.time(); RUNS='experiments/runs'; BND=657489
V7={"blend_w":{"POR":0.50,"PERM":0.50,"SW":0.75},"thresholds":{"POR":0.13,"PERM":0.27,"SW":0.05},
    "clf_tcn_w":{"POR":0.60,"PERM":0.70,"SW":0.30}}
GRID=np.arange(0.01,0.991,0.01)
F.LAG_OFFSETS=[-10,-5,-3,-2,-1,1,2,3,5,10]; F.WIN_SIZES=[5,11,21]
WELLS=load_wells("./train",has_labels=True); ids=[w["logId"].iloc[0] for w in WELLS]
va=set(ids[72:])
tr=pd.concat([w for w in WELLS if w["logId"].iloc[0] not in va],ignore_index=True)
va_d=pd.concat([w for w in WELLS if w["logId"].iloc[0] in va],ignore_index=True)
yid=va_d["logId"].to_numpy(); Ytr=make_targets(tr); Yva=make_targets(va_d)
Dtr,default_Yva=default_mask(Ytr),default_mask(Yva)
ytr={c:Ytr[c].to_numpy() for c in TARGETS}; yva={c:Yva[c].to_numpy() for c in TARGETS}
m_va={c:~np.isnan(yva[c]) for c in TARGETS}
Z=np.load(f'{RUNS}/tcn_all80_parts.npz')
ens={c:np.float32(Z[f"reg_{c}"][BND:]) for c in TARGETS}; pdf={c:np.float32(Z[f"pdf_{c}"][BND:]) for c in TARGETS}
st=joblib.load(f'{RUNS}/screen_tree_cache.joblib')
base_p={c:st['p_def'][c] for c in TARGETS}; base_r={c:np.asarray(st['reg'][c],np.float64) for c in TARGETS}
RAW13=["DEPTH","GR","PE","SP","CAL","AC","DEN","CNL","RXO","RT","DEVI","AZIM","BIT"]
def av(c,p,t):
    p=np.asarray(p,np.float64); t=np.asarray(t,np.float64)
    if c=="PERM": return np.maximum(0,1-np.abs(np.log10(np.maximum(p/np.maximum(t,EPS),EPS))))
    tol=0.08 if c=="POR" else 0.05
    return np.maximum(0,1-np.abs(p-t)/(tol*(np.abs(t)+EPS)))
def sweep(pm,c,rt):
    m=m_va[c]; cw=V7["clf_tcn_w"][c]; bw=V7["blend_w"][c]
    pmv=((1-cw)*pm+cw*pdf[c])[m]; regb=np.power(10.0,(1-bw)*rt+bw*ens[c])[m]
    fd=av(c,np.full(int(m.sum()),DEFAULT_VALUES[c]),yva[c][m]); fr=av(c,regb,yva[c][m])
    o=np.argsort(pmv,kind="stable"); pos=np.searchsorted(pmv[o],GRID,side="left")
    Fd=np.concatenate([[0.0],np.cumsum(fd[o])]); Fr=np.concatenate([[0.0],np.cumsum(fr[o])])
    n=len(pmv); a=(Fr[pos]+(Fd[n]-Fd[pos]))/n; j=int(np.argmax(a))
    return float(GRID[j]), float(a[j])
def total(pm,r,ths):
    s=0.0
    for c in TARGETS:
        m=m_va[c]; cw=V7["clf_tcn_w"][c]; bw=V7["blend_w"][c]
        pmv=(1-cw)*pm[c]+cw*pdf[c]; regb=np.power(10.0,(1-bw)*r[c]+bw*ens[c])
        row=np.nan_to_num(100.0*WEIGHTS[c]*av(c,np.where(pmv>=ths[c],DEFAULT_VALUES[c],regb),yva[c]))
        nv=m.astype(int); s+=float((np.asarray(row,np.float64)*nv/float(nv.sum())).sum())
    return s
Xtr=F.make_features(tr,context=True,priors=True); Xva=F.make_features(va_d,context=True,priors=True)
def mm(raw,g,w):
    out={}
    for c in CURVE_COLS:
        s=raw[c]
        mn=s.groupby(g,group_keys=False).transform(lambda s:s.rolling(w,center=True,min_periods=1).min())
        mx=s.groupby(g,group_keys=False).transform(lambda s:s.rolling(w,center=True,min_periods=1).max())
        out[f"lo{w}_{c}"]=(s-mn).to_numpy(); out[f"hi{w}_{c}"]=(mx-s).to_numpy()
    return pd.DataFrame(out).astype(np.float32)
Xtr_c=pd.concat([Xtr,mm(Xtr[RAW13].copy(),tr["logId"].to_numpy(),101)],axis=1)
Xva_c=pd.concat([Xva,mm(Xva[RAW13].copy(),va_d["logId"].to_numpy(),101)],axis=1)
print(f"{Xtr_c.shape[1]} 特征  [{time.time()-T0:.0f}s]",flush=True)
res={}
for c in TARGETS:
    v=~np.isnan(ytr[c]); clf=_clf(); clf.fit(Xtr_c.loc[v],Dtr.loc[v,c].astype(int).to_numpy())
    p=clf.predict_proba(Xva_c)[:,1].astype(np.float32); del clf; gc.collect()
    real=v & (~Dtr[c].to_numpy().astype(bool))
    reg=_reg(monotonic=POR_MONOTONIC if c=="POR" else None, columns=list(Xtr_c.columns))
    yt=transform_target(c,Ytr[c].loc[real]).to_numpy(); yraw=Ytr[c].loc[real].to_numpy().astype(np.float64)
    w=np.abs(np.log10(np.maximum(yraw,EPS))) if c=="PERM" else np.abs(yraw)
    w=np.clip(w,np.percentile(w,5),np.percentile(w,95)); w=(w/w.mean()).astype(np.float32)
    reg.fit(Xtr_c.loc[real],yt,sample_weight=w)
    rraw=reg.predict(Xva_c); r=np.asarray(rraw,np.float64)
    r = r if c=="PERM" else np.log10(np.maximum(r,EPS))
    r=r.astype(np.float32); del reg; gc.collect()
    res[c]=(p,r); print(f"  {c} fit done [{time.time()-T0:.0f}s]",flush=True)
print("fits saved to res",flush=True)
print("saved npz-alt json",flush=True)
cp={c:res[c][0].astype(np.float32) for c in TARGETS}; cr={c:res[c][1].astype(np.float64) for c in TARGETS}
TH0={c:sweep(base_p[c],c,base_r[c])[0] for c in TARGETS}; base_tot=total(base_p,base_r,TH0)
print("\n"+"="*96); print("两因子分解（基线 = screen_tree_cache 部署同源，各自重扫阈值）"); print("="*96)
print(f"基线 Total = {base_tot:.3f}  (最优阈值 "+" ".join(f"{c} {TH0[c]:.3f}" for c in TH0)+")")
tR={c:sweep(base_p[c],c,cr[c]) for c in TARGETS}
tC={c:sweep(cp[c],c,base_r[c]) for c in TARGETS}
tH={c:sweep(cp[c],c,cr[c]) for c in TARGETS}
a=total(base_p,cr,{c:tR[c][0] for c in TARGETS}); b=total(cp,base_r,{c:tC[c][0] for c in TARGETS})
c2=total(cp,cr,{c:tH[c][0] for c in TARGETS})
print(f"  只换回归器   {a:.3f}   Δ {a-base_tot:+.3f}")
print(f"  只换分类器   {b:.3f}   Δ {b-base_tot:+.3f}")
print(f"  两者都换     {c2:.3f}   Δ {c2-base_tot:+.3f}    交互项 {c2-a-b+base_tot:+.3f}")
print("  逐目标最优 acc（两者都换）: "+"  ".join(f"{c} {tH[c][1]:.4f}@th{tH[c][0]:.3f}" for c in TARGETS))
print(f"  逐目标最优 acc（只换分类器）: "+"  ".join(f"{c} {tC[c][1]:.4f}" for c in TARGETS))
print(f"  逐目标最优 acc（只换回归器）: "+"  ".join(f"{c} {tR[c][1]:.4f}" for c in TARGETS))
print(f"[{time.time()-T0:.0f}s]")
