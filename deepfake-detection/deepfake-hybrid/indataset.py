import numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
L=lambda n: np.load('feats/'+n+'.npz')
ffo,ffl,fa1,fa2=[L(n) for n in ('ff_orig','ff_flip','ff_aug1','ff_aug2')]
y=ffo['y']; vids=ffo['vid']
ids=[set(v.split('_')) for v in vids]            # real: '000'  fake: '000_003' (two people)
people=sorted(set().union(*ids)); rng=np.random.RandomState(0); rng.shuffle(people)
test_people=set(people[:len(people)//5])
is_test=np.array([s<=test_people for s in ids])        # every person in the video is held out
is_train=np.array([len(s&test_people)==0 for s in ids]) # nobody in the video is held out
print('train frames',is_train.sum(),'test frames',is_test.sum(),'(frames mixing held-out and training people are dropped)')
def agg(p,v):
    u,inv=np.unique(v,return_inverse=True); return (np.bincount(inv,p)/np.bincount(inv))[inv]
for name,parts in (('GenD only',('gend',)),('GenD + MoE-FFD hybrid',('gend','moe'))):
    X=lambda d: np.hstack([d[p] for p in parts])
    Xtr=np.vstack([X(d)[is_train] for d in (ffo,ffl,fa1,fa2)]); ytr=np.concatenate([y[is_train]]*4)
    sc=StandardScaler().fit(Xtr); m=LogisticRegression(C=0.01,max_iter=3000).fit(sc.transform(Xtr),ytr)
    ptr=m.predict_proba(sc.transform(X(ffo)[is_train]))[:,1]; pte=m.predict_proba(sc.transform(X(ffo)[is_test]))[:,1]
    pv=agg(pte,vids[is_test])
    print(name)
    print('   TRAINING-set accuracy (not a valid performance measure):',round(accuracy_score(y[is_train],ptr>0.5),4))
    print('   IN-DATASET test, person-disjoint, frame-level : acc',round(accuracy_score(y[is_test],pte>0.5),4),'AUROC',round(roc_auc_score(y[is_test],pte),4))
    print('   IN-DATASET test, person-disjoint, video-level : acc',round(accuracy_score(y[is_test],pv>0.5),4),'AUROC',round(roc_auc_score(y[is_test],pv),4))
