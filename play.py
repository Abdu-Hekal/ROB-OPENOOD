#!/usr/bin/env python3
import torch
import sys
from openood.networks.resnet18_32x32 import ResNet18_32x32

def main(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18_32x32(num_classes=10)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.", "", 1)
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    print("Model structure:")
    print(model)
    print("\nTop-level layers:")
    for name, layer in model.named_children():
        print(f"{name}: {layer}")
    print("\nParameter shapes:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    print("\nForward hook shapes for a dummy input:")
    hooks = []
    def hook_fn(name):
        def hook(mod, inp, out):
            print(f"{name}: {tuple(out.shape)}")
        return hook
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn(name)))
    dummy = torch.randn(1, 3, 32, 32).to(device)
    model(dummy)
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        ckpt_path = "results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"
    main(ckpt_path) 