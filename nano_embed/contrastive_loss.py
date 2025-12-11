import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_a, embeddings_b):
        """
        Computes the SimCSE-style contrastive loss.

        Args:
            embeddings_a (torch.Tensor): Embeddings of the first set of sentences (e.g., original).
                                         Shape: (batch_size, embedding_dim)
            embeddings_b (torch.Tensor): Embeddings of the second set of sentences (e.g., augmented or positive pair).
                                         Shape: (batch_size, embedding_dim)
        Returns:
            torch.Tensor: The computed contrastive loss.
        """
        # Normalize embeddings to unit vectors
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)

        # Compute cosine similarity between all pairs of embeddings
        # similarities[i, j] is cosine similarity between embeddings_a[i] and embeddings_b[j]
        cosine_sim_ab = torch.matmul(embeddings_a, embeddings_b.transpose(0, 1)) / self.temperature
        
        # Similarities within embeddings_a for negative pairs
        cosine_sim_aa = torch.matmul(embeddings_a, embeddings_a.transpose(0, 1)) / self.temperature
        
        # Similarities within embeddings_b for negative pairs
        cosine_sim_bb = torch.matmul(embeddings_b, embeddings_b.transpose(0, 1)) / self.temperature

        # Create positive labels (diagonal elements indicate positive pairs)
        batch_size = embeddings_a.size(0)
        labels = torch.arange(batch_size).to(embeddings_a.device)

        # Combine similarities for the numerator of the softmax
        # For each embeddings_a[i], its positive pair is embeddings_b[i].
        # Negative pairs are embeddings_b[j] where j != i, and all embeddings_a[k] (for in-batch negatives).
        # We also include similarities with other embeddings in 'a' and 'b' as negatives.
        
        # Original SimCSE loss only uses (a_i, a_j) and (a_i, a_i^+)
        # For simplicity, here we consider (a_i, b_i) as positive, and (a_i, b_j) for j!=i as negative,
        # plus (a_i, a_j) for j!=i as negative.

        # The logits for a_i to b_i (positive pair)
        pos_logits = torch.diag(cosine_sim_ab) # (batch_size)

        # All other (a_i, b_j) as negatives, plus (a_i, a_j) as negatives
        # The diagonal of cosine_sim_aa (self-similarity) should be removed or handled.
        # Let's construct the full matrix of logits for the denominator more cleanly.
        
        # Concatenate negative similarities
        # [a_i vs b_j (all), a_i vs a_k (excluding a_i itself)]
        
        # Simplified for now: just treating (a_i, b_i) as positive and everything else in the batch as negative.
        # This means, for each a_i, the batch has 1 positive (b_i) and (2*batch_size - 2) negatives.
        
        # Full similarity matrix for batch:
        # [a_0, b_0, a_1, b_1, ..., a_N, b_N]
        # Then, for a_0, its positive is b_0. All other are negatives.
        # This requires reshaping and careful indexing.

        # Following a common SimCSE implementation approach for in-batch negatives:
        # Create a concatenated matrix of all embeddings, [A; B]
        all_embeddings = torch.cat([embeddings_a, embeddings_b], dim=0) # (2*batch_size, embedding_dim)
        
        # Compute similarities between all concatenated embeddings
        # similarities[i, j] is cosine similarity between all_embeddings[i] and all_embeddings[j]
        full_similarity_matrix = torch.matmul(all_embeddings, all_embeddings.transpose(0, 1)) / self.temperature
        
        # Positive pairs: (a_i, b_i)
        # Indices for embeddings_a are 0 to batch_size - 1
        # Indices for embeddings_b are batch_size to 2*batch_size - 1
        
        # For embeddings_a[i], its positive pair is embeddings_b[i].
        # So, the positive pair is at full_similarity_matrix[i, batch_size + i]
        # And also embeddings_b[i] to embeddings_a[i] is positive: full_similarity_matrix[batch_size + i, i]

        # Targets for contrastive loss:
        # For a_i, its positive is b_i (index `i + batch_size`).
        # For b_i, its positive is a_i (index `i`).
        # So the labels are [batch_size, batch_size + 1, ..., 2*batch_size - 1, 0, 1, ..., batch_size - 1]
        
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size).to(embeddings_a.device), # targets for embeddings_a are corresponding embeddings_b
            torch.arange(batch_size).to(embeddings_a.device)                # targets for embeddings_b are corresponding embeddings_a
        ], dim=0) # (2*batch_size)

        # Mask out self-similarities
        # The diagonal elements are self-similarities and should be ignored as negatives.
        # We are interested in `full_similarity_matrix[i, labels[i]]` as positive.
        # For the negatives, we take all other entries *except* self-similarities and the positive.
        
        # Create a mask to set diagonal to a very small number (effectively -inf after / temperature)
        # This prevents an embedding from being its own negative.
        mask = torch.eye(2 * batch_size).bool().to(embeddings_a.device)
        full_similarity_matrix = full_similarity_matrix.masked_fill(mask, -1e9) # Fill diagonal with large negative value

        loss = F.cross_entropy(full_similarity_matrix, labels)
        
        return loss
