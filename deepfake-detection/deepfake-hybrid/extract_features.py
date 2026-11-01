# Extracts frozen GenD (1024-d) and MoE-FFD (768-d) features for FF++ (train) and DFDC (test), with flip/augmented views.
# Run once on a GPU. Env: BASE=/content DATA=/content/data NTR=12000 NTE=5000
BASE=__import__('os').environ.get('BASE','/kaggle/working')
import os, sys, glob, random, io, re, numpy as np, torch
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,BASE+'/GenD'); sys.path.insert(0,BASE+'/MoE-FFD'); os.chdir(BASE+'/GenD')
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from huggingface_hub import hf_hub_download
from src.hf.modeling_gend import GenD
from ViT_MoE import vit_base_patch16_224_in21k

DATA=os.environ.get('DATA','/root/gend_data'); FF=DATA+'/FF-x1.3-th0.5-all'; DF=DATA+'/DFDC-pre-x1.3-th0.5-all'
OUT=BASE+'/feats'; os.makedirs(OUT,exist_ok=True)
N_TRAIN_PER_CLASS=int(os.environ.get('NTR','23000')); N_TEST_PER_CLASS=int(os.environ.get('NTE','10000'))
dev='cuda'; random.seed(0)

def list_imgs(root, real_dirs, fake_dirs, n):
    out=[]
    for lab,dirs in ((0,real_dirs),(1,fake_dirs)):
        fs=[]
        for d in dirs: fs+=glob.glob(os.path.join(root,d,'*','*.png'))
        random.shuffle(fs); fs=fs[:n]
        for f in fs: out.append((f,lab,os.path.basename(os.path.dirname(f))))
    return out

def augment(im):
    if random.random()<0.7:
        b=io.BytesIO(); im.save(b,'JPEG',quality=random.randint(35,90)); b.seek(0); im=Image.open(b).convert('RGB')
    if random.random()<0.4: im=im.filter(ImageFilter.GaussianBlur(random.uniform(0.3,1.5)))
    im=ImageEnhance.Brightness(im).enhance(random.uniform(0.75,1.25))
    im=ImageEnhance.Contrast(im).enhance(random.uniform(0.75,1.25))
    if random.random()<0.3:
        s=random.uniform(0.5,0.85); w,h=im.size; im=im.resize((int(w*s),int(h*s))).resize((w,h))
    return im

gend=GenD.from_pretrained("yermandy/GenD_CLIP_L_14").to(dev).eval()
moe=vit_base_patch16_224_in21k(pretrained=False,num_classes=2)
moe.load_state_dict(torch.load(hf_hub_download(repo_id="luobo91/MoE-FFD",filename="MoE-FFD.tar"),map_location='cpu')['model_state_dict'])
moe=moe.to(dev).eval()
mt=alb.Compose([alb.Resize(224,224),alb.Normalize(mean=[0.5]*3,std=[0.5]*3),ToTensorV2()])
gp=gend.feature_extractor.preprocess

class DS(Dataset):
    def __init__(s,items,view): s.items=items; s.view=view
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        im=Image.open(s.items[i][0]).convert('RGB')
        if s.view=='flip': im=im.transpose(Image.FLIP_LEFT_RIGHT)
        elif s.view=='aug': im=augment(im)
        return gp(im), mt(image=np.array(im))['image']

@torch.no_grad()
def run(items,view,name):
    f=os.path.join(OUT,name+'.npz')
    if os.path.exists(f): print('skip',name); return
    dl=DataLoader(DS(items,view),batch_size=64,num_workers=4)
    G=[];M=[]
    for a,b in dl:
        with torch.autocast('cuda',dtype=torch.float16):
            g=gend.feature_extractor(a.to(dev))
        m,_=moe.forward_features(b.to(dev))   # MoE-FFD gating breaks under fp16, so it runs in fp32
        G.append(g.float().cpu().numpy()); M.append(m.float().cpu().numpy())
        if len(G)%50==0: print(name,len(G)*64,'/',len(items),flush=True)
    np.savez(f,gend=np.concatenate(G),moe=np.concatenate(M),y=np.array([x[1] for x in items]),vid=np.array([x[2] for x in items]),path=np.array([x[0] for x in items]))
    print('saved',name,flush=True)

tr=list_imgs(FF,['real'],['DF','F2F','FS','NT'],N_TRAIN_PER_CLASS)
te=list_imgs(DF,['real'],['fake'],N_TEST_PER_CLASS)
print('train',len(tr),'test',len(te))
run(te,'orig','dfdc_orig'); run(te,'flip','dfdc_flip')
run(tr,'orig','ff_orig'); run(tr,'flip','ff_flip')
random.seed(1); run(tr,'aug','ff_aug1')   # same images, augmented copy (labels identical, same order)
random.seed(2); run(tr,'aug','ff_aug2')
print('ALL DONE')
