"""HybridDetector = GenD (CLIP ViT-L/14, LayerNorm-tuned) + ForensicsAdapter, fused by averaging fake probabilities.
GenD branch: GenD feature extractor -> standardise -> linear head (fitted on FF++ features) [+ horizontal-flip averaging]
ForAda branch: authors' released weights, CLIP-normalised 256px input."""
import os, sys, re, numpy as np, torch, torch.nn as nn
from PIL import Image
MEAN=[0.48145466,0.4578275,0.40821073]; STD=[0.26862954,0.26130258,0.27577711]
GEND_DIR='/content/GenD'; FA_DIR='/content/ForensicsAdapter'

class _Pass(nn.Module):                      # X-ray map post-processing does not affect the real/fake probability
    def forward(self,x,b): return x

class HybridDetector(nn.Module):
    def __init__(self,gend_fe,gend_pre,fa,h_mean,h_scale,h_w,h_b,tta_flip=True):
        super().__init__()
        self.gend_fe=gend_fe; self.gend_pre=gend_pre; self.fa=fa; self.tta_flip=tta_flip
        t=lambda a: torch.as_tensor(np.asarray(a),dtype=torch.float32)
        self.register_buffer('h_mean',t(h_mean)); self.register_buffer('h_scale',t(h_scale))
        self.register_buffer('h_w',t(h_w)); self.register_buffer('h_b',t(h_b).reshape(()))
        self.register_buffer('fa_mean',torch.tensor(MEAN).view(1,3,1,1)); self.register_buffer('fa_std',torch.tensor(STD).view(1,3,1,1))
    def _dev(self): return self.h_mean.device
    def _gend_run(self,ims):
        dev=self._dev(); x=torch.stack([self.gend_pre(im) for im in ims]).to(dev)
        with torch.autocast('cuda',dtype=torch.float16,enabled=(dev.type=='cuda')):
            f=self.gend_fe(x)
        z=(f.float()-self.h_mean)/self.h_scale
        return torch.sigmoid(z@self.h_w+self.h_b)
    def gend_prob(self,ims):
        p=self._gend_run(ims)
        if self.tta_flip: p=(p+self._gend_run([im.transpose(Image.FLIP_LEFT_RIGHT) for im in ims]))/2
        return p
    def fa_prob(self,ims):
        n0=len(ims)
        if n0==1: ims=list(ims)*2   # ForAda squeezes the batch dim, so a batch of 1 breaks
        dev=self._dev(); b=len(ims)
        x=torch.stack([torch.from_numpy(np.asarray(im.convert('RGB').resize((256,256),Image.BILINEAR)).copy()).permute(2,0,1).float()/255. for im in ims]).to(dev)
        x=(x-self.fa_mean)/self.fa_std
        dd={'image':x,'if_boundary':torch.zeros(b,device=dev),'label':torch.zeros(b,dtype=torch.long,device=dev),'mask':torch.zeros(b,1,256,256,device=dev),'landmark':None}
        out=self.fa(dd,inference=True)
        p=out['prob'].detach().float().reshape(-1).clone()
        self.fa.prob.clear(); self.fa.label.clear(); self.fa.correct=0; self.fa.total=0
        return p[:n0]
    @torch.no_grad()
    def predict_proba(self,ims):
        """ims: list of PIL images (aligned face crops). Returns (hybrid, gend, foradapter) fake-probabilities."""
        pg=self.gend_prob(ims); pf=self.fa_prob(ims)
        return (pg+pf)/2,pg,pf
    @torch.no_grad()
    def predict(self,path,thr=0.5):
        ph,pg,pf=self.predict_proba([Image.open(path).convert('RGB')])
        return dict(verdict='FAKE' if ph.item()>thr else 'REAL',p_fake=ph.item(),p_gend=pg.item(),p_foradapter=pf.item())

def build(head_arrays,fa_weights='/content/fa_weights.pth',dev='cuda'):
    """head_arrays = (mean,scale,w,b) of the GenD linear head."""
    sys.path.insert(0,GEND_DIR); os.chdir(GEND_DIR)
    from src.hf.modeling_gend import GenD
    g=GenD.from_pretrained("yermandy/GenD_CLIP_L_14").to(dev).eval()
    fe=g.feature_extractor; pre=fe.preprocess
    p=FA_DIR+'/model/ds.py'; s=open(p).read(); open(p,'w').write(s.replace('/data/cuixinjie/weights','/content/clip_weights'))
    sys.path.insert(0,FA_DIR); os.chdir(FA_DIR)
    from model.ds import DS
    fa=DS(clip_name='ViT-L/14',adapter_vit_name='vit_tiny_patch16_224',num_quires=128,fusion_map={0:0,1:1,2:8,3:15},mlp_dim=256,mlp_out_dim=128,head_num=16,device=dev)
    ck=torch.load(fa_weights,map_location=dev); ck=ck.get('state_dict',ck) if isinstance(ck,dict) else ck
    r=fa.load_state_dict({re.sub(r'^module\.','',k):v for k,v in ck.items()},strict=False)
    assert len(r.missing_keys)==0 and len(r.unexpected_keys)==0, 'ForAda weights did not load cleanly'
    fa=fa.to(dev).eval(); fa.masked_xray_post_process=_Pass()
    m=HybridDetector(fe,pre,fa,*head_arrays).to(dev).eval()
    return m

def save(model,path='/content/hybrid_detector.pt'):
    sd={k:(v.half() if v.is_floating_point() else v) for k,v in model.state_dict().items()}
    torch.save(sd,path); print('saved',path,round(os.path.getsize(path)/1e9,2),'GB')
def load(path,head_dims=1024,dev='cuda'):
    m=build((np.zeros(head_dims),np.ones(head_dims),np.zeros(head_dims),0.0),dev=dev)
    print(m.load_state_dict(torch.load(path,map_location=dev),strict=False)); return m.eval()
