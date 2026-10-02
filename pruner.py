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
            # 1. Count the 1s in this layer's row → new rank
            new_rank = binary_mask[i].sum().item()
            new_rank = max(1, int(new_rank)) # prevent division by zero in forward pass

            # 2. Slice MatA: keep first new_rank columns
            #    layer.MatA.data has shape (in_features, r_max)
            #    We want (in_features, new_rank)
            new_A_data = layer.MatA.data[:, :new_rank]    # All rows, first new_rank columns
        # First new_rank rows, all columns

    
            # 3. Slice MatB: keep first new_rank rows
            new_B_data = layer.MatB.data[:new_rank, :]
            #    layer.MatB.data has shape (r_max, out_features)
            #    We want (new_rank, out_features)

            # 4. Replace them with new Parameters
            layer.MatA = Parameter(new_A_data.clone())
            layer.MatB = Parameter(new_B_data.clone())
            
            # 5. Update layer.R_max = new_rank
            layer.R_max = new_rank
