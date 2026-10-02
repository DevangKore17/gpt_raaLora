import torch
from torch.nn import Module, Parameter

class RaaLoRA_GPT2_c_attn(Module):
    """
    GPT-2 merges Q, K, and V into a single 'c_attn' Conv1D layer.
    The original LoRA paper explicitly states they adapt only W_q and W_v.
    This custom layer wraps the HuggingFace Conv1D layer and applies
    RaaLoRA scaling and deltas specifically to the Q and V output slices.
    """
    def __init__(self, base_conv1d, in_features, alpha, R_max):
        super().__init__()
        # Keep the original frozen base layer
        self.base_layer = base_conv1d
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        self.alpha = alpha
        self.R_max = R_max
        
        # MatA and MatB for Query (Q)
        self.MatA_q = Parameter(torch.randn(in_features, R_max) * 0.01)
        self.MatB_q = Parameter(torch.zeros(R_max, in_features))
        
        # MatA and MatB for Value (V)
        self.MatA_v = Parameter(torch.randn(in_features, R_max) * 0.01)
        self.MatB_v = Parameter(torch.zeros(R_max, in_features))
        
    def forward(self, x, router_scale=None):
        if router_scale is None:
            router_scale = getattr(self, "current_scale", None)
            
        if router_scale is None:
            raise ValueError("router_scale must be provided or self.current_scale must be set")
            
        if router_scale.dim() == 2:
            router_scale = router_scale.unsqueeze(1)
            
        # 1. Get the frozen base output [batch, seq_len, 3 * in_features]
        base_out = self.base_layer(x)
        
        # 2. Query LoRA path
        hq = x @ self.MatA_q
        hq_scaled = hq * router_scale
        delta_q = hq_scaled @ self.MatB_q * (self.alpha / self.R_max)
        
        # 3. Value LoRA path
        hv = x @ self.MatA_v
        hv_scaled = hv * router_scale
        delta_v = hv_scaled @ self.MatB_v * (self.alpha / self.R_max)
        
        # 4. Inject deltas into the specific Q and V slices of the base output
        d_model = x.size(-1)
        # base_out shape is [..., 3 * d_model]. 
        # Q is [..., :d_model], K is [..., d_model:2*d_model], V is [..., 2*d_model:]
        base_out[..., :d_model] += delta_q
        base_out[..., 2 * d_model:] += delta_v
        
        return base_out
