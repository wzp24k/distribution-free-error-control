"""Figure 8, rebuilt from the validated runs.
(a) acceptance vs k, (b) willingness to answer vs k, (c) label-free triage.
Cross-tissue rows are from r14 (the run of Table 3); random-split and
cross-field-of-view rows are from r20 (grid 'abs', min_accept 25), whose k = 0
rows reproduce Table 1."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP=os.path.join(ROOT,"experiments"); FIG=os.path.join(ROOT,"figures"); WORK=os.path.join(ROOT,"work")
AL=0.10
FAMS=["cellsplit","within","transfer"]
LBL={"cellsplit":"random split (exchangeable)","within":"cross-field-of-view (same cohort)",
     "transfer":"cross-tissue transfer"}
COL={"cellsplit":"#2b6cb0","within":"#dd6b20","transfer":"#c53030"}

def grp(df, by):
    g=df.groupby(by).agg(acc=("accept_rate","mean"),nvf=("nonvac","mean"),
        ok=("covered_nonvac","sum"),nv=("nonvac","sum")).reset_index()
    g["covnv"]=g.ok/g.nv.replace(0,np.nan); return g

# ---- r20: cellsplit / within -------------------------------------------------
u=pd.read_csv(os.path.join(EXP,"r20_units.csv"))
u=u[(u.min_accept==25)&(u.grid=="abs")&(np.isclose(u.alpha,AL))].copy()
u["nonvac"]=(u["empty"]==0).astype(int)
T=grp(u[u.arm=="tgt_k"],["family","gkey"]).rename(columns={"gkey":"k"})
O=grp(u[u.arm=="oracle"],["family","gkey"]).rename(columns={"gkey":"k"})
S=grp(u[u.arm=="src"],["family","gkey"]).rename(columns={"gkey":"k"})
KMAX={f:int(T[T.family==f].k.max()) for f in ("cellsplit","within")}
# ---- r14: transfer -----------------------------------------------------------
r=pd.read_csv(os.path.join(EXP,"r14_targetcal.csv"),
   usecols=["unit","alpha","k","arm","accept_rate","empty","nonvac_covered"])
r=r[r.unit.str.startswith("transfer:")].copy(); r=r[np.isclose(r.alpha,AL)]
r["nonvac"]=(r["empty"]==0).astype(int); r["covered_nonvac"]=r["nonvac_covered"]
Tr=grp(r[r.arm=="tgt_k"],["k"]); Tr["family"]="transfer"
Or=grp(r[r.arm=="oracle"],["k"]); Or["family"]="transfer"
Sr=grp(r[r.arm=="src"],["k"]);   Sr["family"]="transfer"
T=pd.concat([T[["family","k","acc","nvf","covnv"]],Tr[["family","k","acc","nvf","covnv"]]],ignore_index=True)
O=pd.concat([O[["family","k","acc"]],Or[["family","k","acc"]]],ignore_index=True)
S=pd.concat([S[["family","k","acc"]],Sr[["family","k","acc"]]],ignore_index=True)
KMAX["transfer"]=int(T[T.family=="transfer"].k.max())

fig=plt.figure(figsize=(13.6,4.3))
ax=fig.add_subplot(1,3,1)
for f in FAMS:
    a=T[T.family==f].sort_values("k"); b=O[O.family==f].sort_values("k")
    ax.plot(a.k,a.acc,"-o",ms=3.5,color=COL[f],label=LBL[f])
    ax.plot(b.k,b.acc,"--",lw=1.1,color=COL[f],alpha=0.7)
    s0=S[(S.family==f)&(S.k==0)]
    if len(s0): ax.plot([0.55],[s0.acc.iloc[0]],marker="s",ms=6.5,color=COL[f],mec="k",mew=0.6)
ax.set_xscale("symlog",linthresh=10); ax.set_xlim(-0.35,12000)
ax.set_xticks([0,10,100,1000,8000]); ax.set_xticklabels(["0","10","100","1000","8000"],fontsize=8)
ax.set_xlabel("target-domain labels $k$"); ax.set_ylabel("acceptance rate")
ax.set_ylim(-0.02,1.0)
ax.set_title("(a) acceptance bought with labels\nlines: target-calibrated;  dashed: oracle;  squares: $k=0$",fontsize=9)
ax.legend(fontsize=7.5,loc="upper left"); ax.grid(alpha=0.25)

ax2=fig.add_subplot(1,3,2)
for f in FAMS:
    a=T[T.family==f].sort_values("k")
    ax2.plot(a.k,a.nvf,"-o",ms=3.5,color=COL[f])
ax2.axhline(0.05,color="k",ls=":",lw=1); ax2.text(1400,0.062,"usefulness floor 5%",fontsize=7)
ax2.set_xscale("symlog",linthresh=10); ax2.set_xlim(-0.35,12000); ax2.set_ylim(-0.02,1.0)
ax2.set_xticks([0,10,100,1000,8000]); ax2.set_xticklabels(["0","10","100","1000","8000"],fontsize=8)
ax2.set_xlabel("target-domain labels $k$")
ax2.set_ylabel("fraction of evaluations that answer at all")
ax2.set_title("(b) what labels actually buy\nnon-vacuous coverage is 1.000 wherever the\ncertificate answers",fontsize=9)
ax2.grid(alpha=0.25)

ax3=fig.add_subplot(1,3,3)
M=pd.read_csv(os.path.join(WORK,"triage_reach.csv"))
M=M[np.isclose(M.alpha,AL)].copy()
M["cens"]=M.censored==1
cap=1.6*float(M.k_max.max())
M["y"]=np.where(M.cens,cap,M.k_reach)
KMAXF=float(M.k_max.max())
for f in FAMS:
    a=M[(M.family==f)&(~M.cens)]
    b=M[(M.family==f)&(M.cens)]
    ax3.scatter(a.ks_scores,a.y,s=36,color=COL[f],edgecolor="k",linewidth=0.4,label=LBL[f])
    ax3.scatter(b.ks_scores,b.y,s=42,facecolor="none",edgecolor=COL[f],linewidth=1.5)
ax3.axhline(cap,color="k",ls=":",lw=1)
ax3.text(0.015,0.965,"never reached inside the grid",transform=ax3.transAxes,fontsize=7,va="top")
x=M.ks_scores.to_numpy(float)
impy=np.where(M.cens.to_numpy(),M.k_max.to_numpy(float),M.k_reach.to_numpy(float))
rho,pv=stats.spearmanr(x,impy)
z=int(((M.accept_src==0)&M.cens).sum()); zn=int((M.accept_src==0).sum())
ax3.text(0.015,0.04,"$k=0$ acceptance zero in %d of %d units;\n%d of them never reach validity"%(zn,len(M),z),
         transform=ax3.transAxes,fontsize=7)
ax3.set_yscale("log"); ax3.set_ylim(18.0,cap*1.8)
ax3.set_yticks([25,100,500,2000,cap]); ax3.set_yticklabels(["25","100","500","2000","none"],fontsize=8)
ax3.set_xlabel("label-free score shift\n(KS distance, source calibration vs deployment)")
ax3.set_ylabel("labels $k$ needed to reach validity")
ax3.set_title("(c) attainability, not price, is predictable\nSpearman $\\rho$ = %+.2f (p = %.1g, n = %d) against $k$;\nopen symbols: never reached (%d of %d)"
              %(rho,pv,len(M),int(M.cens.sum()),len(M)),fontsize=8.5)
ax3.legend(fontsize=7,loc="lower right"); ax3.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG,"fig8_label_ladder.pdf")); fig.savefig(os.path.join(FIG,"fig8_label_ladder.png"),dpi=200)

L=["# The label ladder (figure 8 source table)",""]
L.append("| family | k | acceptance | non-vacuous fraction | non-vacuous coverage | oracle acceptance |")
L.append("|---|---|---|---|---|---|")
for f in FAMS:
    a=T[T.family==f].sort_values("k")
    for _,rw in a.iterrows():
        o=O[(O.family==f)&np.isclose(O.k,rw.k)]
        L.append("| %s | %d | %.3f | %.3f | %s | %s |"%(f,rw.k,rw.acc,rw.nvf,
            ("%.3f"%rw.covnv) if np.isfinite(rw.covnv) else "n/a",
            ("%.3f"%o.acc.iloc[0]) if len(o) else "-"))
    L.append("")
open(os.path.join(WORK,"tab_ladder.md"),"w",encoding="utf-8").write("\n".join(L))
print("wrote fig8 + work/tab_ladder.md ; panel c spearman %+.3f (p=%.3g, n=%d)"%(rho,pv,len(M)))
