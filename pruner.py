import torch
from torch.nn import Parameter
    
def prune_layers(ema_matrix, layers, threshold=0.5):
    # ema_matrix: shape [num_layers, r_max] — from ema_tracker.get()
    # layers: a list of your RaaLoRa layer objects
    # threshold: values >= this are kept, below are pruned

    # For each layer:
    #   1. Get that layer's row from ema_matrix
    #   2. Binarize it (>= threshold → 1, else → 0)

    binary_mask = (ema_matrix >= threshold).int()

    for i, layer in enumerate(layers):
        keep_indices = torch.nonzero(binary_mask[i]).squeeze(-1)
        if len(keep_indices) == 0:
            keep_indices = torch.tensor([0], device=ema_matrix.device)
            
        new_rank = len(keep_indices)
        keep_scales = ema_matrix[i, keep_indices]

        # 2. Slice MatA and MatB using exact keep_indices and absorb scales
        new_A_data = layer.MatA.data[:, keep_indices]
        new_B_data = layer.MatB.data[keep_indices, :] * keep_scales.unsqueeze(1)

        # 3. Replace them with new Parameters
        layer.MatA = Parameter(new_A_data.clone())
        layer.MatB = Parameter(new_B_data.clone())
        
        # 4. Update layer.R_max = new_rank
        layer.R_max = new_rank
