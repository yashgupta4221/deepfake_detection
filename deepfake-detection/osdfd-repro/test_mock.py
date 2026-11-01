"""Sanity checks on a tiny random CLIP (no download): (1) my block-by-block forward == HF CLIPVisionModel when the
adapters are zero-initialised, (2) param count with real ViT-L/14 dims, (3) FSM / SCL / one training step run."""
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel
from osdfd import OSDFD, single_center_loss, count_trainable

torch.manual_seed(0)
cfg = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                       image_size=224, patch_size=14, hidden_act="quick_gelu")
clip = CLIPVisionModel(cfg).eval()
x = torch.randn(3, 3, 224, 224)
ref = clip(pixel_values=x).last_hidden_state
for K in (0, 2):
    m = OSDFD(first_adapt_block=K, r=4, m=8, clip=clip).eval()
    for b in m.blocks: b.float()
    with torch.no_grad():
        t = m.tokens_from(m.frozen_prefix(x))
    print(f"K={K}  max|mine - HF| = {(t - ref).abs().max().item():.2e}  (adapters zero-init => must be ~0)")
    assert (t - ref).abs().max() < 1e-4

# one training step on cached-style tokens with FSM + SCL
m = OSDFD(first_adapt_block=2, r=4, m=8, clip=clip); m.train()
tok = m.frozen_prefix(x.repeat(4, 1, 1, 1))[:12]
y = torch.tensor([0] * 6 + [1] * 6).float(); dom = torch.tensor([-1] * 6 + [0, 0, 1, 1, 2, 2]); fake = y == 1
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 1e-3)
for use in (False, True):
    logit, z = m.forward_tokens(tok, dom, fake, use_fsm=use)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y) + single_center_loss(z, y.long())
    opt.zero_grad(); loss.backward(); opt.step(); print("fsm", use, "loss", float(loss))
g = [p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if p.requires_grad]
print("trainable tensors with gradient:", sum(map(bool, g)), "/", len(g))

# parameter count at real ViT-L/14 dims (random weights)
big = CLIPVisionModel(CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                                       num_attention_heads=16, image_size=224, patch_size=14, hidden_act="quick_gelu"))
for K in (0, 12):
    mm = OSDFD(first_adapt_block=K, r=8, m=28, clip=big)
    print(f"ViT-L/14, adapt from block {K}: trainable params = {count_trainable(mm):,}  (paper: 2.89M at K=0)")
