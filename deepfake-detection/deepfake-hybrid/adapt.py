import numpy as np, os, json, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
L=lambda n: np.load('feats/'+n+'.npz')
ffo,ffl,fa1,fa2,dco,dcf=[L(n) for n in ('ff_orig','ff_flip','ff_aug1','ff_aug2','dfdc_orig','dfdc_flip')]
P='gend'
Xs=np.vstack([d[P] for d in (ffo,ffl,fa1,fa2)]); ys=np.concatenate([ffo['y']]*4)
Xt=np.vstack([dco[P],dcf[P]]); n=len(dco['y']); yt=np.concatenate([dco['y']]*2)  # labels: diagnostics only
vid=np.array([os.path.dirname(p) for p in dco['path']]); vid2=np.concatenate([vid,vid])
u,inv=np.unique(vid2,return_inverse=True); cnt=np.bincount(inv)
sc=StandardScaler().fit(Xs); Zs=sc.transform(Xs); Zt=sc.transform(Xt)
def vscore(p): return (np.bincount(inv,p)/cnt)[inv]
def evalp(p,tag):
    pv=vscore(p)[:n]; y=yt[:n]  # one entry per frame of the orig view
    pv=(vscore(p)[:n]+vscore(p)[n:])/2
    ths=np.unique(np.quantile(pv,np.linspace(0.02,0.98,97))); best=max(accuracy_score(y,pv>t) for t in ths)
    r=dict(stage=tag,video_auc=roc_auc_score(y,pv),video_acc_at_0_5=accuracy_score(y,pv>0.5),oracle_thr_acc_DIAGNOSTIC_ONLY=best); print(r,flush=True); return r
m=LogisticRegression(C=0.01,max_iter=3000).fit(Zs,ys); p=m.predict_proba(Zt)[:,1]
res=[evalp(p,'baseline FF++ only')]
for rnd,frac in enumerate((0.2,0.3,0.4,0.5,0.6),1):
    vs=np.array([ (p[inv==k]).mean() for k in range(len(u))])
    order=np.argsort(vs); k=int(len(u)*frac)//2
    realv=set(order[:k]); fakev=set(order[-k:])
    sel=np.array([ (i in realv) or (i in fakev) for i in inv]); pl=np.array([1 if i in fakev else 0 for i in inv])
    m=LogisticRegression(C=0.01,max_iter=3000).fit(np.vstack([Zs,Zt[sel]]),np.concatenate([ys,pl[sel]]))
    p=m.predict_proba(Zt)[:,1]
    r=evalp(p,'self-train round %d (%d%% of videos pseudo-labelled)'%(rnd,int(frac*100)))
    r['pseudo_label_acc_DIAGNOSTIC']=float((pl[sel]==yt[sel]).mean()); res.append(r); print('   pseudo-label accuracy (diagnostic):',r['pseudo_label_acc_DIAGNOSTIC'],flush=True)
json.dump(res,open('adapt_results.json','w'),indent=1)
