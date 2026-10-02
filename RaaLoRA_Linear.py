import torch
from torch.nn import Module, Linear, Parameter
from router import RaaLoRARouter 

#building the logic from the scratch
class RaaLoRA(Module):

    def __init__(self,in_feature,out_feature,alpha,R_max,bias=True): #constructor
        super().__init__()   #inhertits all the methods from Module

        self.base_layer=Linear(in_feature,out_feature,bias=bias)
        for param in self.base_layer.parameters():
            param.requires_grad=False
            

        self.alpha=alpha
        self.MatA= Parameter(torch.randn(in_feature,R_max)*0.01)
        self.MatB= Parameter(torch.zeros(R_max,out_feature))
        self.R_max=R_max

    def forward(self,x,router_scale=None):
        
        if router_scale is None:
            router_scale = getattr(self, "current_scale", None)

        if router_scale is None:
            raise ValueError("router_scale must be provided or self.current_scale must be set")

        if router_scale.dim() == 2:
            router_scale = router_scale.unsqueeze(1)
        base_out=self.base_layer(x)
        h= x @ self.MatA
        h_scaled= h* router_scale
        delta= h_scaled @ self.MatB

        return  base_out + delta*(self.alpha/self.R_max)
    

        
        
