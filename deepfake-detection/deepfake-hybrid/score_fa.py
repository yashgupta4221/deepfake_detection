import os, sys, re, numpy as np, torch, traceback
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
R='/content/ForensicsAdapter'; N=int(os.environ.get('N','1000')); OUTF='/content/feats/fa_scores_N%d.npz'%N
# the repo hard-codes the CLIP download folder of the authors' machine -> point it to a local folder
p=R+'/model/ds.py'; s=open(p).read(); s=s.replace('/data/cuixinjie/weights','/content/clip_weights'); open(p,'w').write(s)
sys.path.insert(0,R); os.chdir(R)
from model.ds import DS
dev='cuda'
m=DS(clip_name='ViT-L/14',adapter_vit_name='vit_tiny_patch16_224',num_quires=128,fusion_map={0:0,1:1,2:8,3:15},mlp_dim=256,mlp_out_dim=128,head_num=16,device=dev)
ck=torch.load('/content/fa_weights.pth',map_location=dev)
if isinstance(ck,dict) and 'state_dict' in ck: ck=ck['state_dict']
ck={re.sub(r'^module\.','',k):v for k,v in ck.items()}
r=m.load_state_dict(ck,strict=False)
print('checkpoint keys',len(ck),'| missing',len(r.missing_keys),'| unexpected',len(r.unexpected_keys))
print('sample missing:',r.missing_keys[:4],'| sample unexpected:',r.unexpected_keys[:4])
m=m.to(dev).eval()
import torch.nn as nn
class _Pass(nn.Module):
    def forward(self,x,b): return x
m.masked_xray_post_process=_Pass()   # only affects the X-ray map, not the real/fake probability
mean=torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1); std=torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)
def load_img(pth): 
    im=Image.open(pth).convert('RGB').resize((256,256),Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255.
d=np.load('/content/feats/dfdc_orig.npz'); paths=d['path']; y=d['y']; vid=np.array([os.path.dirname(x) for x in paths])
rng=np.random.RandomState(0); idx=np.sort(rng.choice(len(y),min(N,len(y)),replace=False)); print('scoring',len(idx),'frames')
probs=[]
with torch.no_grad():
    for i in range(0,len(idx),32):
        b=idx[i:i+32]; x=torch.stack([load_img(paths[j]) for j in b]); x=((x-mean)/std).to(dev)
        dd={'image':x,'if_boundary':torch.zeros(len(b),device=dev),'label':torch.tensor(y[b],device=dev),'mask':torch.zeros(len(b),1,256,256,device=dev),'landmark':None}
        try: out=m(dd,inference=True)
        except Exception: traceback.print_exc(); raise SystemExit('forward failed - send me the traceback above')
        probs.append(out['prob'].detach().float().cpu().numpy().reshape(-1))
        if (i//32)%10==0: print(i,'/',len(idx),flush=True)
p=np.concatenate(probs); yy=y[idx]; vv=vid[idx]
u,inv=np.unique(vv,return_inverse=True); pv=(np.bincount(inv,p)/np.bincount(inv))[inv]
print('ForAda on DFDC (%d frames): frame AUROC %.4f acc %.4f | video AUROC %.4f acc %.4f'%(len(idx),roc_auc_score(yy,p),accuracy_score(yy,p>0.5),roc_auc_score(yy,pv),accuracy_score(yy,pv>0.5)))
np.savez(OUTF,idx=idx,prob=p,y=yy,path=paths[idx])
