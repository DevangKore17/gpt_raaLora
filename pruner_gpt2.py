import torch
from torch.nn import Parameter

def prune_gpt2_layers(ema_matrix, layers, threshold=0.5):
    """
    Specialized pruner for GPT-2 which has Q and V matrices in the same layer.
    
    This pruner:
      1. Uses the EMA routing matrix to decide which rank dimensions to keep.
      2. Selects the exact indices the router chose (not just first-N slicing).
      3. Bakes the learned EMA scales into MatB so inference can use scale=1.0.
      4. Applies a scale correction factor (new_rank / old_rank) to cancel
         the (alpha / R_max) shift that occurs when R_max shrinks.
    
    Args:
        ema_matrix: shape [num_layers, r_max] — averaged routing weights from EMA tracker.
        layers: list of RaaLoRA_GPT2_c_attn layer objects.
        threshold: values >= this are kept, below are pruned.
        
    Returns:
        dict with pruning statistics for logging.
    """
    binary_mask = (ema_matrix >= threshold).int()
    stats = {
        'per_layer': [],
        'total_original_params': 0,
        'total_pruned_params': 0,
    }

    for i, layer in enumerate(layers):
        # 1. Find the exact rank indices the router decided to keep
        keep_indices = torch.nonzero(binary_mask[i]).squeeze(-1)
        
        # Edge case: if squeeze reduces a single-element tensor to 0-dim, fix it
        if keep_indices.dim() == 0:
            keep_indices = keep_indices.unsqueeze(0)
        
        # Safety: keep at least 1 rank to avoid empty tensors
        if len(keep_indices) == 0:
            # Pick the rank with the highest EMA value
            best_idx = torch.argmax(ema_matrix[i])
            keep_indices = best_idx.unsqueeze(0)

        old_rank = layer.R_max
        new_rank = len(keep_indices)
        keep_scales = ema_matrix[i, keep_indices]
        
        # Scale correction: the forward pass computes (alpha / R_max).
        # When R_max shrinks from old_rank to new_rank, the denominator shrinks,
        # which would amplify the output. We pre-multiply MatB by (new/old) to cancel.
        scale_correction = new_rank / old_rank

        # Original param count for this layer: 2 * (in_features * old_rank + old_rank * in_features)
        in_features = layer.MatA_q.shape[0]
        original_params = 4 * in_features * old_rank  # Q and V, A and B each
        pruned_params = 4 * in_features * new_rank

        # 2. Prune Q — select correct indices and bake scales + correction into MatB
        new_A_q = layer.MatA_q.data[:, keep_indices]
        new_B_q = layer.MatB_q.data[keep_indices, :] * keep_scales.unsqueeze(1) * scale_correction
        layer.MatA_q = Parameter(new_A_q.clone())
        layer.MatB_q = Parameter(new_B_q.clone())
        
        # 3. Prune V — same treatment
        new_A_v = layer.MatA_v.data[:, keep_indices]
        new_B_v = layer.MatB_v.data[keep_indices, :] * keep_scales.unsqueeze(1) * scale_correction
        layer.MatA_v = Parameter(new_A_v.clone())
        layer.MatB_v = Parameter(new_B_v.clone())
        
        # 4. Update layer metadata
        layer.R_max = new_rank
        
        # Record stats
        stats['per_layer'].append({
            'layer_idx': i,
            'old_rank': old_rank,
            'new_rank': new_rank,
            'kept_indices': keep_indices.tolist(),
            'ema_values': ema_matrix[i].tolist(),
            'kept_scales': keep_scales.tolist(),
        })
        stats['total_original_params'] += original_params
        stats['total_pruned_params'] += pruned_params
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"  PRUNING SUMMARY")
    print(f"{'='*50}")
    for info in stats['per_layer']:
        print(f"  Layer {info['layer_idx']:2d}: rank {info['old_rank']} → {info['new_rank']}  "
              f"(kept indices: {info['kept_indices']}, "
              f"scales: [{', '.join(f'{s:.3f}' for s in info['kept_scales'])}])")
    
    avg_rank = sum(s['new_rank'] for s in stats['per_layer']) / len(stats['per_layer'])
    compression = 1.0 - (stats['total_pruned_params'] / stats['total_original_params']) if stats['total_original_params'] > 0 else 0
    print(f"\n  Average Rank: {avg_rank:.2f}")
    print(f"  LoRA Parameter Reduction: {compression*100:.1f}%")
    print(f"  Total LoRA Params: {stats['total_original_params']:,} → {stats['total_pruned_params']:,}")
    print(f"{'='*50}\n")
    
    return stats
