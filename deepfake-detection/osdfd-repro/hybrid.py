"""HYBRID = OSDFD (training recipe: FSM + Single-Center Loss, frozen CLIP ViT-L/14) x MoE-FFD (architecture: routed
mixture-of-experts LoRA and routed mixture-of-experts conv adapters).  ONE network, ONE loss, no score averaging.

What is taken from where
  OSDFD   : frozen CLIP ViT-L/14 backbone, parallel CDC adapter, Forgery Style Mixture (FSM), Single-Center Loss (SCL),
            forgery-domain balanced batches.
  MoE-FFD : several LoRA experts of different rank on q/k/v, several convolution experts (vanilla / central-difference /
            angular-difference) in the adapter, a router that picks one expert, and a load-balancing loss.
  NEW     : FSM is applied to the hidden states BEFORE the routed blocks (`fsm_layers`), so every router/expert is
            trained on forgery-style-mixed features and cannot route by forgery fingerprint; one joint loss
            BCE + lam*SCL + moe_coef*balance trains experts, routers and head together.

Round 2 option `lora_route='image'`: the LoRA router picks ONE expert per image from the mean patch token (like the
adapter router and like MoE-FFD's image-level routing) instead of one per token.

Reductions:  1 LoRA expert (rank 8) + 1 conv expert ('cd') + FSM + SCL  ~= OSDFD ;   many experts + fsm_p=0 + lam=0 ~= MoE-FFD.

Deviations from MoE-FFD (kept on purpose, listed in README): (1) router output is used as a straight-through gate
(forward value 1, gradient flows) so the task loss trains the router; (2) Switch-style balance loss E*sum(f_e*P_e)
instead of the CV^2 loss; (3) 3 conv experts and 4 LoRA experts (not 5 / 7) to fit a T4 time budget; (4) CLIP ViT-L/14
backbone and 16x16 token grid (their code assumes ImageNet ViT-B, 14x14)."""
import torch, torch.nn as nn, torch.nn.functional as F
from osdfd import OSDFD, count_trainable

RING = [0, 1, 2, 5, 8, 7, 6, 3]                              # outer positions of a 3x3 kernel, clockwise


class Router(nn.Module):
    """Noisy top-1 gate (Shazeer et al.).  Returns expert index, a straight-through gate (value 1.0, gradient to the
    router), the soft probabilities and the Switch-style balance loss."""
    def __init__(self, d, n_exp):
        super().__init__()
        self.n = n_exp
        self.w = nn.Parameter(torch.randn(d, n_exp) * 0.02); self.wn = nn.Parameter(torch.zeros(d, n_exp))

    def forward(self, h):                                    # h [..., d]  ->  idx [...], gate [...], aux, usage [E]
        h = h.float(); lead = h.shape[:-1]; h = h.reshape(-1, h.shape[-1])
        logits = h @ self.w
        if self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(h @ self.wn)
        p = F.softmax(logits, -1); idx = p.argmax(-1)
        pt = p.gather(1, idx[:, None]).squeeze(1)
        gate = pt / pt.detach()                              # == 1 in the forward pass, d/dp = 1/p in the backward pass
        f = F.one_hot(idx, self.n).float().mean(0); P = p.mean(0)
        aux = self.n * (f * P).sum()
        return idx.reshape(lead), gate.reshape(lead), aux, f.detach()


class MoELoRA(nn.Module):
    """Experts of different rank, one per token.  All experts are stored as ONE A_cat [d,R] / B_cat [R,3d]; a rank mask
    selects the chosen expert's columns, so the whole layer is two matmuls.  Mathematically identical to separate experts."""
    def __init__(self, d, ranks, route="token"):
        super().__init__()
        assert route in ("token", "image")
        self.route = route                                   # 'token': one expert per token; 'image': one expert per image
        self.ranks = list(ranks); R = sum(ranks)
        self.A = nn.Parameter(torch.empty(d, R)); self.B = nn.Parameter(torch.zeros(R, 3 * d))     # B=0: starts as identity
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.register_buffer("seg", torch.cat([torch.full((r,), i) for i, r in enumerate(ranks)]), persistent=False)
        self.router = Router(d, len(ranks))
        self.aux = None; self.usage = None

    def forward(self, h):                                    # h [B,N,d] -> (dq, dk, dv)
        if self.route == "image":                            # router sees the mean patch token (as MoEAdapter / MoE-FFD)
            idx, gate, self.aux, self.usage = self.router(h[:, 1:].mean(1))           # [B]
            mask = ((self.seg.view(1, -1) == idx.unsqueeze(-1)).to(h.dtype) * gate.unsqueeze(-1).to(h.dtype)).unsqueeze(1)   # [B,1,R]
        else:
            idx, gate, self.aux, self.usage = self.router(h)                          # [B,N]
            mask = (self.seg.view(1, 1, -1) == idx.unsqueeze(-1)).to(h.dtype) * gate.unsqueeze(-1).to(h.dtype)
        hid = (h @ self.A.to(h.dtype)) * mask
        return (hid @ self.B.to(h.dtype)).chunk(3, dim=-1)


class ConvExpert(nn.Module):
    """1x1 down -> 3x3 conv of a given type -> 1x1 up (zero-init).  kind: cv vanilla, cd central-difference
    (= OSDFD's CDC), ad angular-difference (PiDiNet-style: weights act on x_i - x_{i-1} around the ring)."""
    def __init__(self, d, m, kind):
        super().__init__()
        self.kind = kind
        self.down = nn.Conv2d(d, m, 1); self.up = nn.Conv2d(m, d, 1)
        self.w = nn.Parameter(torch.empty(m, m, 3, 3)); self.b = nn.Parameter(torch.zeros(m))
        nn.init.kaiming_uniform_(self.w, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def kernel(self):
        w = self.w
        if self.kind == "cv":
            return w
        if self.kind == "cd":
            k = w.clone(); k[:, :, 1, 1] = w[:, :, 1, 1] - w.sum((2, 3)); return k
        if self.kind == "ad":
            wf = w.flatten(2); ring = torch.tensor(RING, device=w.device)
            wr = wf[..., ring]; k = torch.zeros_like(wf); k[..., ring] = wr - wr.roll(-1, -1)      # sum_i w_i(x_i - x_{i-1}) = sum_j x_j (w_j - w_{j+1})
            return k.view_as(w)
        raise ValueError(self.kind)

    def forward(self, p):
        return self.up(F.conv2d(self.down(p), self.kernel().to(p.dtype), self.b.to(p.dtype), padding=1))


class MoEAdapter(nn.Module):
    """Parallel to the MLP.  One expert per image (router sees the mean patch token, as in MoE-FFD)."""
    def __init__(self, d, m, kinds, grid=16):
        super().__init__()
        self.g = grid; self.kinds = list(kinds)
        self.experts = nn.ModuleList([ConvExpert(d, m, k) for k in self.kinds])
        self.router = Router(d, len(self.kinds))
        self.aux = None; self.usage = None

    def forward(self, h):                                    # h [B,N,d] (ln2 output); CLS token gets zero
        B, N, d = h.shape
        idx, gate, self.aux, self.usage = self.router(h[:, 1:].mean(1))
        p = h[:, 1:].transpose(1, 2).reshape(B, d, self.g, self.g)
        out = h.new_zeros(B, d, self.g, self.g)
        for e, ex in enumerate(self.experts):
            sel = (idx == e).nonzero().squeeze(1)
            if sel.numel():
                out = out.index_copy(0, sel, (ex(p[sel]) * gate[sel].view(-1, 1, 1, 1).to(h.dtype)).to(h.dtype))
        out = out.reshape(B, d, N - 1).transpose(1, 2)
        return torch.cat([torch.zeros_like(h[:, :1]), out], 1)


class HBlock(nn.Module):
    """CLIP block (frozen weights taken from an OSDFD Block) + MoE-LoRA on q/k/v + MoE adapter parallel to the MLP."""
    def __init__(self, b, ranks, kinds, m, grid, lora_route="token"):
        super().__init__()
        self.ln1, self.ln2, self.q, self.k, self.v, self.o, self.mlp, self.h = b.ln1, b.ln2, b.q, b.k, b.v, b.o, b.mlp, b.h
        d = self.q.in_features; self.adapt = True
        self.lora = MoELoRA(d, ranks, lora_route); self.adp = MoEAdapter(d, m, kinds, grid)

    def forward(self, x):
        B, N, d = x.shape
        h = self.ln1(x)
        dq, dk, dv = self.lora(h)
        q, k, v = self.q(h) + dq, self.k(h) + dk, self.v(h) + dv
        sp = lambda t: t.reshape(B, N, self.h, d // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(1, 2).reshape(B, N, d)
        x = x + self.o(o)
        h2 = self.ln2(x)
        return x + self.mlp(h2) + self.adp(h2)

    @property
    def aux(self):
        return self.lora.aux + self.adp.aux


class HybridOSDFD(OSDFD):
    def __init__(self, clip_name="openai/clip-vit-large-patch14", first_adapt_block=12, lora_ranks=(4, 8, 16, 32),
                 adapter_experts=("cv", "cd", "ad"), m=28, fsm_layers=None, lora_route="token", clip=None):
        super().__init__(clip_name, first_adapt_block=first_adapt_block, r=1, m=1, clip=clip)
        for i, b in enumerate(self.blocks):
            if b.adapt:
                self.blocks[i] = HBlock(b, lora_ranks, adapter_experts, m, self.grid, lora_route)
        for n, p in self.named_parameters():                  # freeze everything except the new trainable parts
            p.requires_grad = any(s in n for s in (".lora.", ".adp.", "head.", "cls."))
            if p.requires_grad: p.data = p.data.float()
        L = len(self.blocks)
        self.fsm_layers = set(fsm_layers if fsm_layers is not None else [first_adapt_block, L])
        self.cfg = dict(arch="hybrid", lora_ranks=list(lora_ranks), adapter_experts=list(adapter_experts), m=m,
                        fsm_layers=sorted(self.fsm_layers), lora_route=lora_route)

    def forward_tokens(self, x, dom=None, is_fake=None, use_fsm=False):
        L = len(self.blocks)
        for i in range(self.K, L):
            if use_fsm and i in self.fsm_layers: x = self.fsm(x, dom, is_fake)       # BEFORE the routed block i
            x = self.blocks[i](x)
        if use_fsm and L in self.fsm_layers: x = self.fsm(x, dom, is_fake)           # after the last block (OSDFD placement)
        return self.head_forward(x)

    def aux_loss(self):
        a = [b.aux for b in self.blocks if b.adapt and getattr(b.lora, "aux", None) is not None]
        return torch.stack(a).mean() if a else torch.zeros((), device=next(self.parameters()).device)

    def usage(self):                                         # fraction of tokens/images sent to each expert, per block
        return {"lora": [b.lora.usage.tolist() for b in self.blocks if b.adapt and b.lora.usage is not None],
                "adapter": [b.adp.usage.tolist() for b in self.blocks if b.adapt and b.adp.usage is not None]}

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items() if any(s in k for s in (".lora.", ".adp.", "head.", "cls."))}


def build_model(cfg, clip_name, K, clip=None):
    """cfg=None or arch=='osdfd' -> plain OSDFD, else HybridOSDFD."""
    if not cfg or cfg.get("arch", "osdfd") == "osdfd":
        return OSDFD(clip_name, first_adapt_block=K, clip=clip)
    return HybridOSDFD(clip_name, first_adapt_block=K, lora_ranks=cfg["lora_ranks"], adapter_experts=cfg["adapter_experts"],
                       m=cfg["m"], fsm_layers=cfg["fsm_layers"], lora_route=cfg.get("lora_route", "token"), clip=clip)
