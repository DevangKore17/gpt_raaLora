import torch
from torch.nn import Parameter

def prune_gpt2_layers(ema_matrix, layers, threshold=0.5):
    """
    Specialized pruner for GPT-2 which has Q and V matrices in the same layer.
    """
    binary_mask = (ema_matrix >= threshold).int()

    for i, layer in enumerate(layers):
        keep_indices = torch.nonzero(binary_mask[i]).squeeze(-1)
        if len(keep_indices) == 0:
            keep_indices = torch.tensor([0], device=ema_matrix.device)
            
        new_rank = len(keep_indices)

        # Prune Q
        new_A_q = layer.MatA_q.data[:, keep_indices]
        new_B_q = layer.MatB_q.data[keep_indices, :]
        layer.MatA_q = Parameter(new_A_q.clone())
        layer.MatB_q = Parameter(new_B_q.clone())
        
        # Prune V
        new_A_v = layer.MatA_v.data[:, keep_indices]
        new_B_v = layer.MatB_v.data[keep_indices, :]
        layer.MatA_v = Parameter(new_A_v.clone())
        layer.MatB_v = Parameter(new_B_v.clone())
        
        # Update layer metadata
        layer.R_max = new_rank
