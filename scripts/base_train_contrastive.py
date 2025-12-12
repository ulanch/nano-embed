"""
Train model for Contrastive Learning (SimCSE). Run as:

python base_train_contrastive.py

or distributed as:

torchrun --nproc_per_node=8 base_train_contrastive.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nano_embed.gpt import GPT, GPTConfig
from nano_embed.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nano_embed.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nano_embed.tokenizer import get_tokenizer, get_token_bytes
from nano_embed.checkpoint_manager import save_checkpoint, load_checkpoint
from nano_embed.loss_eval import evaluate_bpb
from nano_embed.engine import Engine
import torch
from nano_embed.contrastive_loss import ContrastiveLoss
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 34 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
bidirectional = True # enable bidirectional attention for embedding model
use_lora = True # enable LoRA
lora_rank = 4 # LoRA rank
lora_alpha = 32 # LoRA alpha
lora_dropout = 0.3 # LoRA dropout
# Contrastive learning specific
contrastive_temperature = 0.07 # temperature for contrastive loss
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = 1000 # explicit number of steps of the optimization (-1 = disable)
# load_from_mntp_ckpt:
load_from_base_dir = "" # path to the directory containing base model checkpoints
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nano_embed', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nano_embed_contrastive", name=run, config=user_config)
contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temperature)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    bidirectional=bidirectional,
    use_lora=use_lora,
    lora_rank=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# Load from base directory if provided
if load_from_base_dir:
    checkpoint_files = [f for f in os.listdir(load_from_base_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {load_from_base_dir}")

    # Find the checkpoint with the highest step number
    latest_checkpoint = ""
    max_step = -1
    for ckpt in checkpoint_files:
        try:
            step_num = int(ckpt.split('_')[1].split('.')[0])
            if step_num > max_step:
                max_step = step_num
                latest_checkpoint = ckpt
        except (ValueError, IndexError):
            continue

    if not latest_checkpoint:
        raise ValueError(f"Could not determine the latest checkpoint in {load_from_base_dir}")

    checkpoint_path = os.path.join(load_from_base_dir, latest_checkpoint)
    print0(f"Loading weights from latest checkpoint: {checkpoint_path}")
    model_data, _, _ = load_checkpoint(load_from_base_dir, max_step, device, load_optimizer=False, rank=ddp_rank)
    model.load_state_dict(model_data, strict=False, assign=True)
    del model_data

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
if not use_dummy_wandb:
    wandb.watch(model, log='all', log_freq=100)
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Output directories
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}_contrastive"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict, mask_ratio=0.0)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device, mask_ratio=0.0)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_loss = float("inf") # changed from min_val_bpb
    smooth_train_loss = 0 # EMA of training loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_loss = meta_data["val_loss"] # changed from val_bpb
    min_val_loss = loop_state["min_val_loss"] # changed from min_val_bpb
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
for step in tqdm(range(num_iterations), desc="Training"):
    last_step = step == num_iterations - 1
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val loss (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        val_losses = []
        for _ in range(eval_steps):
            x_val, y_val, _ = next(val_loader)
            with autocast_ctx:
                loss = model(x_val, y_val)
            val_losses.append(loss.item())
        val_loss = torch.tensor(val_losses).mean().item() # changed from val_bpb
        print0(f"Step {step:05d} | Validation loss: {val_loss:.4f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/loss": val_loss,
        })
        model.train()



def get_lora_state_dict(model):
    lora_state_dict = {}
    for n, p in model.named_parameters():
        if "lora_a" in n or "lora_b" in n:
            lora_state_dict[n] = p
    return lora_state_dict

# ... (rest of the code)

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        # Determine what to save based on use_lora
        if use_lora:
            model_to_save = get_lora_state_dict(orig_model)
        else:
            model_to_save = orig_model.state_dict()

        save_checkpoint(
            checkpoint_dir,
            step,
            model_to_save, # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_loss": val_loss, # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_loss": min_val_loss,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            # SimCSE: create two different views with independent dropout noise
            hidden = model(x, return_embeddings=True)
            # Masked mean pooling to ignore padding tokens (id == 0)
            attn = (x != 0).unsqueeze(-1).expand_as(hidden)
            # Apply dropout to create two different stochastic views
            hidden_a = F.dropout(hidden, p=0.1, training=True)
            hidden_b = F.dropout(hidden, p=0.1, training=True)
            sum_a = (hidden_a * attn).sum(dim=1)
            sum_b = (hidden_b * attn).sum(dim=1)
            denom = torch.clamp(attn.sum(dim=1), min=1e-9)
            embeddings_a = sum_a / denom
            embeddings_b = sum_b / denom
            loss = contrastive_loss_fn(embeddings_a, embeddings_b)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    # tqdm will handle the step printing, so we can simplify this
    # print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)


 # print a few more stats (safe)
peak_mem_mib = 0.0
if device_type == "cuda" and torch.cuda.is_available():
    try:
        peak_mem_mib = torch.cuda.max_memory_reserved() / (1024 * 1024)
    except Exception:
        peak_mem_mib = 0.0
print0(f"Peak memory usage: {peak_mem_mib:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation loss: {min_val_loss:.4f}") # changed from bpb

# Log to report
from nano_embed.report import get_report
results = {}
total_training_flops_est = num_flops_per_token * total_tokens
get_report().log(section="Contrastive model training", data=[ # changed section name
    user_config, # CLI args
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation loss": min_val_loss, # changed from bpb
        "Final validation loss": val_loss, # changed from bpb
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{total_training_flops_est:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{peak_mem_mib:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
