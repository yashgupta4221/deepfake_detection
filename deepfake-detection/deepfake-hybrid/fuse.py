import numpy as np, os, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import rankdata
F='/content/feats/'; N=int(os.environ.get('N','10000'))
z=np.load(F+'fa_scores_N%d.npz'%N); idx=z['idx']; pfa=z['prob']; y=z['y']
ffo,ffl,fa1,fa2,dco,dcf=[np.load(F+n+'.npz') for n in ('ff_orig','ff_flip','ff_aug1','ff_aug2','dfdc_orig','dfdc_flip')]
Xs=np.vstack([d['gend'] for d in (ffo,ffl,fa1,fa2)]); ys=np.concatenate([ffo['y']]*4)
sc=StandardScaler().fit(Xs); m=LogisticRegression(C=0.01,max_iter=3000).fit(sc.transform(Xs),ys)
pg=(m.predict_proba(sc.transform(dco['gend'][idx]))[:,1]+m.predict_proba(sc.transform(dcf['gend'][idx]))[:,1])/2
assert (dco['y'][idx]==y).all()
vid=np.array([os.path.dirname(p) for p in dco['path'][idx]]); u,inv=np.unique(vid,return_inverse=True)
def vagg(p): return (np.bincount(inv,p)/np.bincount(inv))[inv]
def rep(name,p,acc=True):
    pv=vagg(p); s='%-34s frame AUROC %.4f | video AUROC %.4f'%(name,roc_auc_score(y,p),roc_auc_score(y,pv))
    if acc: s+=' | frame acc %.4f | video acc %.4f'%(accuracy_score(y,p>0.5),accuracy_score(y,pv>0.5))
    print(s)
print('frames',len(y),'videos',len(u),'| correlation GenD vs ForAda scores: %.3f'%np.corrcoef(pg,pfa)[0,1])
rep('GenD alone (frozen LR)',pg); rep('ForensicsAdapter alone',pfa)
rep('FUSION: mean of probabilities',(pg+pfa)/2)
rk=lambda p: rankdata(p)/len(p)
rep('FUSION: mean of ranks (AUROC only)',(rk(pg)+rk(pfa))/2,acc=False)
