"""
Merge LoRA weights trained on top of a base nanochat model into the base weights,
producing a standalone checkpoint without LoRA parameters.
"""

import os
import argparse
import torch

from nano_embed.gpt import GPT, GPTConfig
from nano_embed.lora import LoRALinear
from nano_embed.checkpoint_manager import load_checkpoint, save_checkpoint
from nano_embed.common import autodetect_device_type, compute_init, compute_cleanup


def get_latest_step(checkpoint_dir: str) -> int:
    steps = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model_") and f.endswith(".pt"):
            try:
                steps.append(int(f.split("_")[1].split(".")[0]))
            except Exception:
                pass
    if not steps:
        raise ValueError(f"No model_*.pt found in {checkpoint_dir}")
    return max(steps)


def get_module_by_path(root: torch.nn.Module, path: str) -> torch.nn.Module:
    mod = root
    for part in path.split('.'):
        mod = getattr(mod, part)
    return mod


def merge_lora_into_base(base_model: GPT, lora_model: GPT):
    for name, module in lora_model.named_modules():
        if isinstance(module, LoRALinear):
            # Find corresponding linear in base model (same name path)
            base_linear = get_module_by_path(base_model, name)
            assert isinstance(base_linear, torch.nn.Linear), f"Expected Linear at {name} in base model"
            # delta = (A @ B).T * alpha, shape (out, in)
            delta = (module.lora.lora_a @ module.lora.lora_b).T * module.lora.alpha
            base_linear.weight.data += delta.to(base_linear.weight.dtype)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA checkpoint into base model weights")
    parser.add_argument("--lora_checkpoint_path", type=str, required=True, help="Path to LoRA-only checkpoint file (model_*.pt)")
    parser.add_argument("--base_model_dir", type=str, required=True, help="Directory with base model checkpoints (model_*.pt)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write merged checkpoint into")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp_rank != 0:
        compute_cleanup()
        return

    # Load LoRA metadata (for model_config)
    lora_dir = os.path.dirname(args.lora_checkpoint_path)
    lora_step = int(os.path.basename(args.lora_checkpoint_path).split('_')[1].split('.')[0])
    lora_state, _, lora_meta = load_checkpoint(lora_dir, lora_step, device, load_optimizer=False, rank=ddp_rank)

    # Build base model (no LoRA) and load full base weights
    base_step = get_latest_step(args.base_model_dir)
    _, _, base_meta = load_checkpoint(args.base_model_dir, base_step, device, load_optimizer=False, rank=ddp_rank)
    model_config_kwargs = base_meta.get("model_config", lora_meta.get("model_config"))
    base_cfg = GPTConfig(**{**model_config_kwargs, "use_lora": False})
    with torch.device("meta"):
        base_model = GPT(base_cfg)
    base_model.to_empty(device=device)
    base_weights, _, _ = load_checkpoint(args.base_model_dir, base_step, device, load_optimizer=False, rank=ddp_rank)
    base_model.init_weights()  # initialize buffers
    base_model.load_state_dict(base_weights, strict=True, assign=True)
    del base_weights

    # Build a LoRA-enabled model to hold LoRA parameters, load only LoRA tensors
    lora_cfg = GPTConfig(**{**model_config_kwargs, "use_lora": True})
    with torch.device("meta"):
        lora_model = GPT(lora_cfg)
    lora_model.to_empty(device=device)
    lora_model.init_weights()
    lora_model.load_state_dict(lora_state, strict=False, assign=True)
    del lora_state

    # Merge LoRA deltas into base weights
    merge_lora_into_base(base_model, lora_model)

    # Save merged checkpoint (metadata based on base_meta, ensure use_lora=False)
    os.makedirs(args.output_dir, exist_ok=True)
    # Use the LoRA step so the merged checkpoint step reflects training stage
    merged_step = lora_step
    # Update meta to reflect no LoRA
    base_meta["model_config"]["use_lora"] = False
    save_checkpoint(
        args.output_dir,
        merged_step,
        base_model.state_dict(),
        [],
        base_meta,
        rank=ddp_rank,
    )
    print(f"Merged model saved to {os.path.join(args.output_dir, f'model_{merged_step:06d}.pt')}")
    compute_cleanup()


if __name__ == "__main__":
    main()
