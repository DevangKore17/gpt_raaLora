import torch
from torch.nn import Linear, Module

class RaaLoRARouter(Module):
    def __init__(self,d_module,num_layers,r_max,bottleneck_dim=256):
        super().__init__()

        self.d_module=d_module
        self.num_layers= num_layers
        self.r_max= r_max

        self.downstream=Linear(d_module,bottleneck_dim)
        self.upstream= Linear(bottleneck_dim,num_layers * r_max)
    
        
    def forward(self,x):
        input1= torch.mean(x,dim=1)
        h=self.downstream(input1)
        h=torch.relu(h)
        out=self.upstream(h)
        out=torch.sigmoid(out)

        out=out.reshape(-1,self.num_layers,self.r_max)
        return out

if __name__== "__main__":

    d_model=64
    num_layers=4
    r_max=8

    router= RaaLoRARouter(d_model,num_layers,r_max)
    x = torch.randn(2, 10, 64)
    out=router.forward(x)
    print(out)
    
    # Check 1: Shape
    print("Shape:", out.shape)          # Should print [2, 4, 8]

    # Check 2: Range
    print("Min:", out.min().item())     # Should be >= 0.0
    print("Max:", out.max().item())     # Should be <= 1.0

    # Check 3: Gradients flow
    out.sum().backward()
    print("Gradients OK!")   
        
    
