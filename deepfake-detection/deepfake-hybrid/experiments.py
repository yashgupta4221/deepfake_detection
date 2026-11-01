# CPU-only. Needs the six feature files in ./feats (see extract_features.py), then: python experiments.py
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neural_network import MLPClassifier
L=lambda n: np.load('feats/'+n+'.npz')
def X(d,parts=('gend','moe')): return np.hstack([d[p] for p in parts])
ffo,ffl,fa1,fa2,dco,dcf=[L(n) for n in ('ff_orig','ff_flip','ff_aug1','ff_aug2','dfdc_orig','dfdc_flip')]
import os
ytr=ffo['y']; yte=dco['y']; vid=np.array([os.path.dirname(p) for p in dco['path']])  # class+video folder
# hold-out FF++ by video for in-dataset number + threshold choice (never touches DFDC)
vf=ffo['vid']; uv=np.unique(vf); rng=np.random.RandomState(0); rng.shuffle(uv); hv=set(uv[:len(uv)//5]); hold=np.array([v in hv for v in vf])
def vagg(p,v):
    u,inv=np.unique(v,return_inverse=True); return (np.bincount(inv,p)/np.bincount(inv))[inv]
def rep(name,p,thr=0.5):
    r=dict(name=name,frame_auc=roc_auc_score(yte,p),frame_acc=accuracy_score(yte,p>thr))
    pv=vagg(p,vid); r.update(video_auc=roc_auc_score(yte,pv),video_acc=accuracy_score(yte,pv>thr)); return r
def fit(parts,views,clf='lr',C=0.01,align=False):
    Xtr=np.vstack([X(v,parts)[~hold] for v in views]); Ytr=np.concatenate([ytr[~hold]]*len(views))
    sc=StandardScaler().fit(Xtr); m=(LogisticRegression(C=C,max_iter=3000) if clf=='lr' else MLPClassifier((256,),alpha=1e-2,early_stopping=True,max_iter=200,random_state=0)).fit(sc.transform(Xtr),Ytr)
    ff_hold=m.predict_proba(sc.transform(X(ffo,parts)[hold]))[:,1]
    def pred(a,b):
        out=[]
        for d in (a,b):
            Z=X(d,parts)
            # unlabeled test-domain standardization (no DFDC labels used)
            Z=(Z-Z.mean(0))/(Z.std(0)+1e-6) if align else sc.transform(Z)
            out.append(m.predict_proba(Z)[:,1])
        return out
    return m,ff_hold,pred
res=[]
for parts in (('gend',),('gend','moe')):
    for views_name,views in (('orig',[ffo]),('orig+flip+aug',[ffo,ffl,fa1,fa2])):
        for align in (False,True):
            for C in (0.01,):
                m,fh,pred=fit(parts,views,'lr',C,align); p1,p2=pred(dco,dcf)
                r=rep('+'.join(parts)+'|'+views_name+'|align=%s|C=%s'%(align,C),(p1+p2)/2)
                r['ff_holdout_acc']=accuracy_score(ytr[hold],fh>0.5); res.append(r); print(r,flush=True)
m,fh,pred=fit(('gend','moe'),[ffo,ffl,fa1,fa2],'mlp',align=True); p1,p2=pred(dco,dcf)
r=rep('gend+moe|MLP|aug|align',(p1+p2)/2); r['ff_holdout_acc']=accuracy_score(ytr[hold],fh>0.5); res.append(r); print(r)
json.dump(res,open('results.json','w'),indent=1)
# Headline = pick config by FF++ hold-out only is uninformative for transfer; report ALL rows, plus a clearly-labelled oracle.
b=max(res,key=lambda r:r['video_auc']); print('BEST video AUC row (selected using DFDC => optimistic):',b)
