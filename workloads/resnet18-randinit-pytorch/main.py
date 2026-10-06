"""resnet18-randinit-pytorch: torchvision ResNet-18 architecture, seeded random weights, one forward pass.

No pretrained weights: torchvision's code is BSD-3-Clause but its ImageNet weights come with
dataset terms that were not verified, and a functional check only needs a deterministic network.
"""
import os
import sys

import torch
import torchvision

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_pytorch"))
import common  # noqa: E402

common.setup()
torch.manual_seed(0)
model = torchvision.models.resnet18(weights=None)        # weights=None: random init from the seed
g = torch.Generator().manual_seed(42)
x = torch.randn(4, 3, 64, 64, generator=g)
# Random-init BatchNorm running statistics are 0/1, which lets activations drift; one seeded
# forward pass in train mode, on the CPU, sets them from data, so the eval-mode logits are well scaled.
model.train()
with torch.no_grad():
    model(torch.randn(16, 3, 64, 64, generator=g))
model.eval()
model = model.to(common.DEVICE)
with torch.no_grad():
    logits = model(x.to(common.DEVICE)).float().cpu()
top = torch.topk(logits, 6, dim=1)   # the sixth only proves the fifth is not tied
gap = min(common.margin_ok(top.values[i], f"image {i}", floor=1e-3) for i in range(logits.shape[0]))
output = {
    "top5_ids_per_image": " | ".join(" ".join(map(str, r[:5])) for r in top.indices.tolist()),   # exact
    "logits_stats": [float(logits.abs().mean()), float(logits.pow(2).mean().sqrt()), float(logits[0, 0]),
                     float(logits[-1, -1])],
}
metrics = {}
if common.REAL and common.DEVICE != "cpu" and os.environ.get("PW_NO_BENCH") != "1":
    big = torch.randn(64, 3, 224, 224, device=common.DEVICE)
    with torch.no_grad():
        sec = common.bench(lambda: model(big), iters=5, rounds=3)
    metrics["images_per_s_b64_224"] = round(64 / sec, 1)
common.finish(output, f"{common.device_name()}, torch {torch.__version__} torchvision {torchvision.__version__}, "
              f"min top1-top2 gap {gap:.3g}", metrics)
