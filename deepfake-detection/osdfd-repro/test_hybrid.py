"""Tests for hybrid.py on a tiny random CLIP (no downloads)."""
import torch, torch.nn.functional as F
from transformers import CLIPVisionConfig, CLIPVisionModel
from osdfd import OSDFD, CDC, single_center_loss, count_trainable
from hybrid import HybridOSDFD, MoELoRA, ConvExpert, MoEAdapter, Router, RING

torch.manual_seed(0)
cfg = CLIPVisionConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                       image_size=224, patch_size=14, hidden_act="quick_gelu")
clip = CLIPVisionModel(cfg).eval()
x = torch.randn(3, 3, 224, 224)
ref = clip(pixel_values=x).last_hidden_state

# 1. zero-initialised experts => hybrid == frozen CLIP exactly
for K in (0, 3):
    m = HybridOSDFD(first_adapt_block=K, lora_ranks=(2, 4), adapter_experts=("cv", "cd", "ad"), m=8, clip=clip).eval()
    for b in m.blocks: b.float()
    with torch.no_grad(): t = m.tokens_from(m.frozen_prefix(x))
    d = (t - ref).abs().max().item(); print(f"[1] K={K} max|hybrid - HF| = {d:.2e}"); assert d < 1e-4

# 2. CDC expert kernel == OSDFD's CDC module ; AD kernel == naive angular-difference conv
ce = ConvExpert(16, 6, "cd"); cdc = CDC(6)
with torch.no_grad(): cdc.conv.weight.copy_(ce.w); cdc.conv.bias.copy_(ce.b)
p = torch.randn(2, 6, 8, 8)
a = F.conv2d(p, ce.kernel(), ce.b, padding=1); b_ = cdc(p)
print("[2] CD kernel vs OSDFD CDC:", (a - b_).abs().max().item()); assert (a - b_).abs().max() < 1e-5
ae = ConvExpert(16, 6, "ad"); w = ae.w.detach()
pp = F.pad(p, (1, 1, 1, 1)); naive = torch.zeros(2, 6, 8, 8)
pos = [(i // 3, i % 3) for i in RING]
for i, (r, c) in enumerate(pos):
    pr, pc = pos[i - 1]
    for y in range(8):
        for xx in range(8):
            naive[:, :, y, xx] += torch.einsum("oc,bc->bo", w[:, :, r, c], pp[:, :, y + r, xx + c] - pp[:, :, y + pr, xx + pc])
got = F.conv2d(p, ae.kernel(), None, padding=1)
print("[2] AD kernel vs naive angular-difference conv:", (got - naive).abs().max().item()); assert (got - naive).abs().max() < 1e-4

# 3. masked single-matmul MoE-LoRA == explicit per-expert LoRA
lo = MoELoRA(32, (2, 4, 8)).eval(); torch.nn.init.normal_(lo.B, std=0.1)
h = torch.randn(2, 11, 32)
dq, dk, dv = lo(h)
idx, gate, _, _ = lo.router(h)
off = [0, 2, 6, 14]; ref_out = torch.zeros(2, 11, 96)
for e in range(3):
    A, Bm = lo.A[:, off[e]:off[e + 1]], lo.B[off[e]:off[e + 1]]
    ref_out += ((h @ A) @ Bm) * (idx == e).unsqueeze(-1)
print("[3] masked LoRA vs per-expert loop:", (torch.cat([dq, dk, dv], -1) - ref_out).abs().max().item())
assert (torch.cat([dq, dk, dv], -1) - ref_out).abs().max() < 1e-5

# 3b. per-image routing: ONE expert per image (same for all its tokens); masked matmul == explicit per-expert loop;
#     old checkpoints without 'lora_route' still build as token routing
li = MoELoRA(32, (2, 4, 8), route="image").eval(); torch.nn.init.normal_(li.B, std=0.1)
dq, dk, dv = li(h)
idx_i, gate_i, _, _ = li.router(h[:, 1:].mean(1))
ref_i = torch.zeros(2, 11, 96)
for e in range(3):
    A, Bm = li.A[:, off[e]:off[e + 1]], li.B[off[e]:off[e + 1]]
    ref_i += ((h @ A) @ Bm) * (idx_i == e).view(-1, 1, 1)
print("[3b] image-routed LoRA vs per-expert loop:", (torch.cat([dq, dk, dv], -1) - ref_i).abs().max().item(), "| experts per image:", idx_i.tolist())
assert (torch.cat([dq, dk, dv], -1) - ref_i).abs().max() < 1e-5
from hybrid import build_model
mi = HybridOSDFD(first_adapt_block=3, lora_ranks=(2, 4), adapter_experts=("cv", "cd"), m=8, lora_route="image", clip=clip)
assert all(b.lora.route == "image" for b in mi.blocks[3:]) and mi.cfg["lora_route"] == "image"
old_cfg = {k: v for k, v in mi.cfg.items() if k != "lora_route"}
assert all(b.lora.route == "token" for b in build_model(old_cfg, "unused", 3, clip=clip).blocks[3:])
mi.train(); tk = mi.frozen_prefix(x.repeat(4, 1, 1, 1))[:12]
for b in mi.blocks[3:]: torch.nn.init.normal_(b.lora.B, std=0.05)
lg, zz = mi.forward_tokens(tk, torch.tensor([-1] * 6 + [0, 0, 1, 1, 0, 1]), torch.tensor([0] * 6 + [1] * 6).bool(), use_fsm=True)
(lg.sum() + zz.sum()).backward()
g = [float(b.lora.router.w.grad.abs().sum()) for b in mi.blocks[3:]]
print("[3b] image-route router grad per block:", [f"{v:.2e}" for v in g]); assert all(v > 0 for v in g)
print("[3b] image routing + old-checkpoint default OK")

# 4. routers get gradient from the TASK loss (not only the balance loss); full train step with FSM at two places
m = HybridOSDFD(first_adapt_block=2, lora_ranks=(2, 4), adapter_experts=("cv", "cd", "ad"), m=8, fsm_layers=[2, 6], clip=clip).train()
with torch.no_grad():
    for b in m.blocks[2:]:
        torch.nn.init.normal_(b.lora.B, std=0.05)
        for ex in b.adp.experts: torch.nn.init.normal_(ex.up.weight, std=0.05)
tok = m.frozen_prefix(x.repeat(4, 1, 1, 1))[:12]
y = torch.tensor([0] * 6 + [1] * 6).float(); dom = torch.tensor([-1] * 6 + [0, 0, 1, 1, 2, 2]); fake = y == 1
opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 1e-3)
logit, z = m.forward_tokens(tok, dom, fake, use_fsm=True)
task = F.binary_cross_entropy_with_logits(logit, y) + single_center_loss(z, y.long())      # NO balance loss here
opt.zero_grad(); task.backward(retain_graph=True)
rg = [float(b.lora.router.w.grad.abs().sum() + b.adp.router.w.grad.abs().sum()) for b in m.blocks[2:]]
print("[4] router grad from task loss only, per block:", [f"{g:.2e}" for g in rg]); assert all(g > 0 for g in rg)
tot = task + 0.01 * m.aux_loss(); opt.zero_grad(); tot.backward(); opt.step()
print("[4] usage lora/adapter (block 0):", m.usage()["lora"][0], m.usage()["adapter"][0])
ng = sum(1 for n, p_ in m.named_parameters() if p_.requires_grad and (p_.grad is None or p_.grad.abs().sum() == 0))
print("[4] trainable tensors without gradient:", ng, "(last block's adapter/LoRA-v legitimately get none: only CLS is read)")

# 5. FSM placement actually changes what the routers see
m.eval(); torch.manual_seed(1)
captured = []
hook = m.blocks[2].lora.router.register_forward_hook(lambda mod, i, o: captured.append(i[0].detach().clone()))
m.forward_tokens(tok, dom, fake, use_fsm=False); m.forward_tokens(tok, dom, fake, use_fsm=True)
hook.remove()
print("[5] router input changes when FSM is on:", (captured[0] - captured[1]).abs().max().item() > 1e-4)
assert (captured[0] - captured[1]).abs().max() > 1e-4
assert (captured[0][:6] - captured[1][:6]).abs().max() < 1e-6, "real samples must be untouched"
print("[5] real samples untouched by FSM: True")

# 6. parameter count at real ViT-L/14 dims
big = CLIPVisionModel(CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                                       num_attention_heads=16, image_size=224, patch_size=14, hidden_act="quick_gelu"))
hb = HybridOSDFD(first_adapt_block=12, clip=big)
print(f"[6] ViT-L/14 hybrid K=12 trainable params: {count_trainable(hb):,}  (OSDFD K=12: 1,515,105; OSDFD paper K=0: 2.89M)")
print("ALL HYBRID TESTS PASSED")
