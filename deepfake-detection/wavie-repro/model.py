"""WAVIE (Pulkit et al., arXiv 2609.14437) - model, written from the paper's Section III.
Trainable: Q1, layer-aggregation logits, low-band linear (137x137), Q2, classifier (~2.14M params).
Fixed: frozen CLIP ViT-L/14 (features cached separately) and the db6 wavelet filters."""
import numpy as np, pywt, torch, torch.nn as nn, torch.nn.functional as F


class DWT1D(nn.Module):
    """3-level db6 DWT/IDWT of a length-n vector as FIXED linear maps (differentiable).
    Built by pushing unit vectors through pywt, so it is exactly pywt.wavedec / waverec
    (default 'symmetric' mode; n=1024 -> low band 137, highs 137/264/517)."""
    def __init__(self, n=1024, wavelet="db6", level=3):
        super().__init__()
        eye = np.eye(n)
        an = pywt.wavedec(eye, wavelet, level=level, axis=-1)          # list of [n, len_k]
        self.lens = [a.shape[1] for a in an]                            # [137,137,264,517]
        T = sum(self.lens)
        offs = np.cumsum([0] + self.lens)
        eyeT = np.eye(T)
        bands = [eyeT[:, offs[k]:offs[k + 1]] for k in range(len(self.lens))]
        syn = pywt.waverec(bands, wavelet, axis=-1)                     # [T, n]
        self.register_buffer("A", torch.tensor(np.concatenate(an, 1), dtype=torch.float32))  # [n,T]
        self.register_buffer("S", torch.tensor(syn, dtype=torch.float32))                    # [T,n]
        self.low_len = self.lens[0]

    def analyze(self, f):
        c = f @ self.A
        return c[:, : self.low_len], c[:, self.low_len:]

    def synthesize(self, low, high):
        return torch.cat([low, high], 1) @ self.S


def mlp(d, p):  # Q1 / Q2: Dropout, Linear, ReLU, Dropout
    return nn.Sequential(nn.Dropout(p), nn.Linear(d, d), nn.ReLU(), nn.Dropout(p))


class WAVIE(nn.Module):
    def __init__(self, n_layers=24, d=1024, p=0.1, mode="full"):
        """mode: full | no_fusion (final-layer CLS only) | no_wavelet | avg_agg  (Table III ablations)"""
        super().__init__()
        self.mode, self.n = mode, n_layers
        self.q1 = mlp(d, p)
        self.alpha = nn.Parameter(torch.zeros(n_layers, d))    # per-layer, per-dim logits (softmax over layers)
        self.dwt = DWT1D(d)
        self.low = nn.Linear(self.dwt.low_len, self.dwt.low_len)
        self.q2 = mlp(d, p)
        self.cls = nn.Linear(d, 1)

    def fuse(self, G):                                    # G: [B, n, d] raw CLS tokens
        if self.mode == "no_fusion":
            return self.q1(G[:, -1])
        h = self.q1(G)                                    # shared Q1 applied per layer
        if self.mode == "avg_agg":
            return h.mean(1)
        w = torch.softmax(self.alpha, dim=0)              # [n, d]
        return (w.unsqueeze(0) * h).sum(1)

    def forward(self, G):
        f = self.fuse(G.float())
        if self.mode == "no_wavelet":
            fhat = f
        else:
            low, high = self.dwt.analyze(f)
            fhat = self.dwt.synthesize(self.low(low), high)   # refine low band only, keep highs
        z = self.q2(fhat)
        return self.cls(z).squeeze(-1), z, fhat


def sup_con(z, y, tau=0.07):
    """Supervised contrastive loss as in the paper: positives = other same-label samples in batch."""
    with torch.autocast(device_type=z.device.type, enabled=False):  # fp32 loss even under AMP
        return _sup_con(z.float(), y, tau)


def _sup_con(z, y, tau):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & ~eye
    npos = pos.sum(1)
    ok = npos > 0
    if ok.sum() == 0:
        return z.sum() * 0
    loss = -(logp * pos).sum(1)[ok] / npos[ok]
    return loss.mean()
