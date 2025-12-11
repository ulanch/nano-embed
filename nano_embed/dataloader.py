from collections import deque

import torch
import pyarrow.parquet as pq

from nano_embed.common import get_dist_info
from nano_embed.dataset import list_parquet_files
from nano_embed.tokenizer import get_tokenizer

def mask_tokens(tokens, mask_ratio, vocab_size, mask_token_id):
    """
    Masks tokens in a 1D tensor for Masked Next Token Prediction (MNTP).
    
    Args:
        tokens (torch.Tensor): A 1D tensor of token IDs.
        mask_ratio (float): The probability of masking a token.
        vocab_size (int): The size of the vocabulary, used for random token replacement.
        mask_token_id (int): The ID of the special MASK token.
        
    Returns:
        tuple: A tuple containing:
            - inputs (torch.Tensor): The token IDs with some tokens masked.
            - targets (torch.Tensor): The original token IDs for masked positions, and -100 elsewhere.
    """
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, mask_ratio)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    # 80% of the time, replace masked input tokens with [MASK] token
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_replaced] = mask_token_id
    
    # 10% of the time, replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long, device=tokens.device)
    tokens[indices_random] = random_words[indices_random]
    
    # The rest of the 10% of the time, keep the masked input tokens unchanged
    
    # Set labels to -100 (ignore_index) for all unmasked tokens
    labels[~masked_indices] = -100 # -100 is the default ignore_index for F.cross_entropy
    
    return tokens, labels

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None, mask_ratio=0.0):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    # The MNTP objective only requires a single special MASK token.
    # The default tokenizer already has a [MASK] token.
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    mask_token_id = tokenizer.get_mask_token_id()
    vocab_size = tokenizer.get_vocab_size()

    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token

    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        
        # Apply masking if mask_ratio is greater than 0
        if mask_ratio > 0.0:
            inputs_masked, targets_masked = mask_tokens(scratch[:-1].clone(), mask_ratio, vocab_size, mask_token_id)
            inputs_cpu = inputs_masked
            targets_cpu = targets_masked
        else:
            inputs_cpu = scratch[:-1]
            targets_cpu = scratch[1:]

        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, mask_ratio=0.0, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, mask_ratio=mask_ratio, **kwargs):
        yield inputs, targets

