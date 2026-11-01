"""Grid of real/fake DFDC test frames with the hybrid's prediction. Run in the Colab session where `model` (HybridDetector) exists.
Frames are a fixed-seed random draw, NOT hand-picked, so wrong predictions appear too."""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
def make_grid(model, feats='/content/feats/', out='/content/prediction_examples.png', per_class=4, seed=3):
    d=np.load(feats+'dfdc_orig.npz'); y=d['y']; paths=d['path']; rng=np.random.RandomState(seed)
    pick=[('real',rng.choice(np.where(y==0)[0],per_class,replace=False)),('fake',rng.choice(np.where(y==1)[0],per_class,replace=False))]
    idx=np.concatenate([p for _,p in pick]); ims=[Image.open(paths[i]).convert('RGB') for i in idx]
    ph,pg,pf=[a.cpu().numpy() for a in model.predict_proba(ims)]
    fig,axs=plt.subplots(2,per_class,figsize=(3.1*per_class,7.4),dpi=170)
    for k,ax in enumerate(axs.ravel()):
        truth='REAL' if y[idx[k]]==0 else 'FAKE'; verdict='FAKE' if ph[k]>0.5 else 'REAL'; ok=(truth==verdict)
        ax.imshow(ims[k]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(3); s.set_color('#1baf7a' if ok else '#e34948')
        ax.set_title('True: %s\nPredicted: %s  %s'%(truth,verdict,'(correct)' if ok else '(WRONG)'),fontsize=10.5,loc='left')
        ax.set_xlabel('hybrid P(fake) = %.2f\nGenD %.2f   ForAda %.2f'%(ph[k],pg[k],pf[k]),fontsize=9.5)
    fig.suptitle('Hybrid predictions on unseen DFDC frames (random draw, not hand-picked)',fontsize=12.5,fontweight='bold',x=0.02,ha='left')
    fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(out); print('saved',out,'| correct:',int(sum((ph>0.5)==(y[idx]==1))),'of',len(idx))
