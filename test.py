import torch
class A:
    def forward(self, x, router_scale=None):
        if router_scale is None:
            router_scale = self.current_scale
        if router_scale.dim() == 2:
            router_scale = router_scale.unsqueeze(1)

