"""
Script to merge LoRA weights into the base model.
"""

import os
import torch

from nano_embed.gpt import GPT, GPTConfig
from nano_embed.lora import merge_lora_weights_in_model
from nano_embed.checkpoint_manager import load_checkpoint, save_checkpoint
from nano_embed.common import get_base_dir, autodetect_device_type, compute_init, compute_cleanup

# -----------------------------------------------------------------------------
# User settings
checkpoint_path = "" # path to the LoRA checkpoint (e.g., 'base_checkpoints/d20_mntp/ckpt_XXXXX.pt')
output_path = "" # path to save the merged checkpoint
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

if not ddp_rank == 0:
    # Only master process merges weights
    compute_cleanup()
    exit()

if not checkpoint_path:
    raise ValueError("Please provide a checkpoint path.")

if not output_path:
    output_path = os.path.join(os.path.dirname(checkpoint_path), "merged_" + os.path.basename(checkpoint_path))

# Load model configuration from checkpoint
model_data, _, meta_data = load_checkpoint(os.path.dirname(checkpoint_path), int(os.path.basename(checkpoint_path).split('_')[1].split('.')[0]), device, load_optimizer=False, rank=ddp_rank)

model_config_kwargs = meta_data["model_config"]
model_config = GPTConfig(**model_config_kwargs)
model = GPT(model_config)
model.load_state_dict(model_data, strict=False, assign=True)

# Merge LoRA weights
model = merge_lora_weights_in_model(model)
print("LoRA weights merged.")

# Save the merged model
save_checkpoint(
    os.path.dirname(output_path),
    int(os.path.basename(output_path).split('_')[1].split('.')[0]),
    model.state_dict(),
    [], # no optimizer state
    meta_data,
    rank=ddp_rank,
)

print(f"Merged model saved to {output_path}")

compute_cleanup()
