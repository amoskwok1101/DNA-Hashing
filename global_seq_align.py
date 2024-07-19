import torch

def global_alignment_batch_with_gap_penalty(seqs1, seqs2, device, match_score=1, mismatch_penalty=0, gap_penalty=-1):
    batch_size, max_len = seqs1.size()
    dtype = torch.float32  # smaller type, assuming scores fit into int16
    
    # Initialize the score matrix
    score_matrix = torch.zeros((batch_size, max_len + 1, max_len + 1), dtype=dtype, device=device)
 
    for i in range(1, max_len + 1):
        score_matrix[:, i, 0] = score_matrix[:, i-1, 0] + gap_penalty
        score_matrix[:, 0, i] = score_matrix[:, 0, i-1] + gap_penalty

    # Precompute match/mismatch results
    match_mask = seqs1[:, :, None] == seqs2[:, None, :]
    scores_update = torch.where(match_mask, match_score, mismatch_penalty)

    for i in range(1, max_len + 1):
        for j in range(1, max_len + 1):
            match = score_matrix[:, i-1, j-1] + scores_update[:, i-1, j-1]
            delete = score_matrix[:, i-1, j] + gap_penalty
            insert = score_matrix[:, i, j-1] + gap_penalty
            
            score_matrix[:, i, j] = torch.max(torch.max(match, delete), insert)
    
    final_scores = score_matrix[:, -1, -1]
    return final_scores