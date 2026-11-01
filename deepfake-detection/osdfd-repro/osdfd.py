"""OSDFD (Kong et al., IEEE TCSVT 2026, "Open-Set Deepfake Detection: A Parameter-Efficient Adaptation Method With
Forgery Style Mixture") on a frozen CLIP ViT-L/14 backbone.

 * LoRA (rank r=8, paper default) on the q/k/v projections of a block          -> h_out = W h + LoRA(h)          (Eq. 4-5)
 * CDC adapter in parallel with the MLP:  Conv1x1 down -> Central-Difference Conv 3x3 -> Conv1x1 up (Eq. 1-3)
 * Forgery Style Mixture (FSM): AdaIN-style mixing of per-sample token statistics across forgery domains (Eq. 6-9)
 * Head: MLP on the final CLS token; Single-Center Loss on the penultimate FC features (Eq. 10-13)
Only LoRA + adapters + head are trained; every CLIP weight is frozen.

`first_adapt_block=K` puts the trainable modules only in blocks K..23. K=0 is the paper's design (all blocks).
K>0 lets us cache the output of the frozen blocks 0..K-1 (exact, no augmentation) and train much faster."""
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import CLIPVisionModel

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class LoRA(nn.Module):
    def __init__(self, d, r):
        super().__init__()
        self.A = nn.Parameter(torch.empty(d, r)); self.B = nn.Parameter(torch.zeros(r, d))   # B=0 -> starts as identity
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        return (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype)


class CDC(nn.Module):
    """Central-difference conv:  out = sum_i w_i (x_i - x_c) = conv(x, W) - x_c * sum(W)   (theta = 1)."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        w = self.conv.weight
        return F.conv2d(x, w, self.conv.bias, padding=1) - F.conv2d(x, w.sum((2, 3), keepdim=True))


class Adapter(nn.Module):
    def __init__(self, d, m, grid=16):
        super().__init__()
        self.g = grid
        self.down = nn.Conv2d(d, m, 1); self.cdc = CDC(m); self.up = nn.Conv2d(m, d, 1)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)                          # starts as zero residual

    def forward(self, x):                                   # x [B,N,d]; the CLS token gets no adapter output
        B, N, d = x.shape
        p = x[:, 1:].transpose(1, 2).reshape(B, d, self.g, self.g)
        o = self.up(self.cdc(self.down(p))).reshape(B, d, N - 1).transpose(1, 2)
        return torch.cat([torch.zeros_like(x[:, :1]), o], 1)


class Block(nn.Module):
    def __init__(self, layer, heads, adapt, r=8, m=28, grid=16):
        super().__init__()
        self.ln1, self.ln2 = layer.layer_norm1, layer.layer_norm2
        a = layer.self_attn
        self.q, self.k, self.v, self.o = a.q_proj, a.k_proj, a.v_proj, a.out_proj
        self.mlp = layer.mlp; self.h = heads
        d = self.q.in_features
        self.adapt = adapt
        if adapt:
            self.lq, self.lk, self.lv = LoRA(d, r), LoRA(d, r), LoRA(d, r)
            self.adapter = Adapter(d, m, grid)

    def forward(self, x):
        B, N, d = x.shape
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        if self.adapt:
            q, k, v = q + self.lq(h), k + self.lk(h), v + self.lv(h)
        sp = lambda t: t.reshape(B, N, self.h, d // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(1, 2).reshape(B, N, d)
        x = x + self.o(o)
        h2 = self.ln2(x)
        out = self.mlp(h2)
        if self.adapt:
            out = out + self.adapter(h2)                    # h_out = MLP(h_in) + Adapter(h_in)   (Eq. 1)
        return x + out


class OSDFD(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-large-patch14", first_adapt_block=0, r=8, m=28, clip=None):
        super().__init__()
        if clip is None:
            dt = torch.float16 if torch.cuda.is_available() else torch.float32
            try:
                clip = CLIPVisionModel.from_pretrained(clip_name, dtype=dt)
            except TypeError:                               # older transformers
                clip = CLIPVisionModel.from_pretrained(clip_name, torch_dtype=dt)
        vm = getattr(clip, "vision_model", clip); cfg = clip.config      # transformers 5.x flattened the module
        self.K = first_adapt_block; self.d = cfg.hidden_size; self.grid = cfg.image_size // cfg.patch_size
        self.embeddings, self.pre = vm.embeddings, vm.pre_layrnorm
        self.blocks = nn.ModuleList([Block(l, cfg.num_attention_heads, i >= first_adapt_block, r, m, self.grid)
                                     for i, l in enumerate(vm.encoder.layers)])
        self.post = vm.post_layernorm
        for n, p in self.named_parameters():                # freeze everything that came from CLIP
            p.requires_grad = False
        for b in self.blocks:
            if b.adapt:
                for p in list(b.lq.parameters()) + list(b.lk.parameters()) + list(b.lv.parameters()) + list(b.adapter.parameters()):
                    p.requires_grad = True; p.data = p.data.float()
        d = self.d
        self.head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64))   # 64-d = penultimate features (SCL)
        self.cls = nn.Linear(64, 1)

    # ---- backbone pieces -------------------------------------------------------------------------------------
    @torch.no_grad()
    def frozen_prefix(self, pix):                           # images -> input tokens of block K   (no grad, frozen)
        x = self.pre(self.embeddings(pix))
        for b in self.blocks[: self.K]:
            x = b(x)
        return x

    def tokens_from(self, x):                               # input tokens of block K -> output tokens of block 23
        for b in self.blocks[self.K:]:
            x = b(x)
        return x

    # ---- Forgery Style Mixture (training only) -----------------------------------------------------------------
    @staticmethod
    def fsm(f, dom, is_fake, beta=(0.1, 0.1), eps=1e-6):
        """f [B,N,d] tokens, dom [B] forgery-domain id (real: -1). Fake features get statistics mixed with those of a
        fake sample from a DIFFERENT domain (Eq. 6-9). delta ~ Beta(0.1,0.1) per sample."""
        f32 = f.float(); idx = torch.nonzero(is_fake).squeeze(1)
        if idx.numel() < 2:
            return f
        fd = dom[idx]; perm = torch.randperm(idx.numel(), device=f.device)
        for _ in range(8):                                  # find a partner from a different domain
            bad = fd[perm] == fd
            if not bad.any(): break
            perm[bad] = torch.randint(0, idx.numel(), (int(bad.sum()),), device=f.device)
        ok = fd[perm] != fd
        x = f32[idx]; xs = x[perm]
        mu, sd = x.mean(1, keepdim=True), (x.var(1, keepdim=True) + eps).sqrt()
        mus, sds = xs.mean(1, keepdim=True), (xs.var(1, keepdim=True) + eps).sqrt()
        delta = torch.distributions.Beta(beta[0], beta[1]).sample((idx.numel(), 1, 1)).to(f.device)
        g_mix, e_mix = delta * sd + (1 - delta) * sds, delta * mu + (1 - delta) * mus
        mixed = g_mix * (x - mu) / sd + e_mix
        mixed = torch.where(ok.view(-1, 1, 1), mixed, x)
        out = f32.clone(); out[idx] = mixed
        return out.to(f.dtype)

    def head_forward(self, tok):
        c = self.post(tok[:, 0]).float()
        z = self.head(c)
        return self.cls(z).squeeze(1), z

    def forward_tokens(self, x, dom=None, is_fake=None, use_fsm=False):
        t = self.tokens_from(x)
        if use_fsm:
            t = self.fsm(t, dom, is_fake)
        return self.head_forward(t)

    @torch.no_grad()
    def predict_images(self, pix):
        return self.head_forward(self.tokens_from(self.frozen_prefix(pix)))[0]

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items()
                if any(s in k for s in (".lq.", ".lk.", ".lv.", ".adapter.", "head.", "cls."))}


def single_center_loss(z, y, margin=0.01):
    """L_SCL = Dist_R + max(Dist_R - Dist_F + margin, 0);  C = mean of the real features in the batch (Eq. 11-13)."""
    z = F.normalize(z, dim=1)   # deviation: unit-norm features so the loss cannot be 'solved' by shrinking features to 0
    zr, zf = z[y == 0], z[y == 1]
    if len(zr) == 0 or len(zf) == 0:
        return z.sum() * 0
    C = zr.mean(0, keepdim=True)
    dr = (zr - C).norm(dim=1).mean(); df = (zf - C).norm(dim=1).mean()
    return dr + F.relu(dr - df + margin)


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.float16 if DEV == "cuda" else torch.bfloat16


def autocast():
    return torch.autocast(DEV, dtype=AMP)


def wdtype(model):
    return model.embeddings.patch_embedding.weight.dtype
