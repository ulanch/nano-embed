"""
Script to evaluate the quality of generated embeddings from nano-embed using the MTEB benchmark.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from mteb import MTEB, DRESModel

from nano_embed.gpt import GPT, GPTConfig
from nano_embed.tokenizer import get_tokenizer
from nano_embed.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nano_embed.checkpoint_manager import load_checkpoint

# -----------------------------------------------------------------------------
# User settings
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model checkpoint
checkpoint_path = "" # path to the model checkpoint directory (e.g., 'base_checkpoints/d20_merged')
# MTEB tasks
tasks = ["STSBenchmark", "STS17"] # a small subset of MTEB for quick evaluation
# -----------------------------------------------------------------------------

class NanoEmbedMTEB(DRESModel):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Computes sentence embeddings from a list of sentences.
        """
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding Sentences"):
            batch_sentences = sentences[i:i+batch_size]
            
            # Tokenize the sentences
            tokenized_batch = [self.tokenizer.encode(s, prepend=self.tokenizer.get_bos_token_id()) for s in batch_sentences]
            
            # Pad to the longest sequence in the batch
            max_len = max(len(tokens) for tokens in tokenized_batch)
            padded_batch = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_batch]
            
            input_ids = torch.tensor(padded_batch, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                embeddings = self.model(input_ids, return_embeddings=True) # (B, T, E)
            
            # Mean pooling over the sequence dimension
            # Note: we should ignore padding tokens in the mean calculation
            attention_mask = (input_ids != 0).unsqueeze(-1).expand(embeddings.size())
            sum_embeddings = torch.sum(embeddings * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_pooled_embeddings.cpu().numpy())
            
        return np.concatenate(all_embeddings, axis=0)

def main():
    # Compute init
    device_type_detected = autodetect_device_type() if device_type == "" else device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type_detected)
    master_process = ddp_rank == 0

    if not master_process:
        compute_cleanup()
        return

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found at {checkpoint_path}")

    # find the latest checkpoint in the directory
    checkpoint_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pt')]
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_path}")

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
        raise ValueError(f"Could not determine the latest checkpoint in {checkpoint_path}")

    full_checkpoint_path = os.path.join(checkpoint_path, latest_checkpoint)

    model_data, _, meta_data = load_checkpoint(checkpoint_path, max_step, device, load_optimizer=False, rank=ddp_rank)
    model_config_kwargs = meta_data["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
    model.load_state_dict(model_data, strict=False, assign=True)
    model.to(device)
    model.eval()

    print0(f"Loaded model from {full_checkpoint_path}")

    # Create MTEB-compatible model
    mteb_model = NanoEmbedMTEB(model=model, tokenizer=tokenizer, device=device)

    # Run MTEB evaluation
    print0(f"Running MTEB evaluation on tasks: {tasks}")
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(mteb_model, output_folder=f"mteb_results/{os.path.basename(checkpoint_path)}")

    print0("MTEB evaluation finished.")
    print0(results)

    compute_cleanup()

if __name__ == "__main__":
    main()