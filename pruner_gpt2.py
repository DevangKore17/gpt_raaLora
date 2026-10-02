import torch
from torch.nn import Parameter

def prune_gpt2_layers(ema_matrix, layers, threshold=0.5):
    """
    Specialized pruner for GPT-2 which has Q and V matrices in the same layer.
    """
    binary_mask = (ema_matrix >= threshold).int()

    for i, layer in enumerate(layers):
        new_rank = binary_mask[i].sum().item()
        new_rank = max(1, int(new_rank)) # prevent division by zero

        # Prune Q
        new_A_q = layer.MatA_q.data[:, :new_rank]
        new_B_q = layer.MatB_q.data[:new_rank, :]
        layer.MatA_q = Parameter(new_A_q.clone())
        layer.MatB_q = Parameter(new_B_q.clone())
        
        # Prune V
        new_A_v = layer.MatA_v.data[:, :new_rank]
        new_B_v = layer.MatB_v.data[:new_rank, :]
        layer.MatA_v = Parameter(new_A_v.clone())
        layer.MatB_v = Parameter(new_B_v.clone())
        
        # Update layer metadata
        layer.R_max = new_rank
