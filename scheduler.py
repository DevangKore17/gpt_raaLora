import torch
from torch.nn import Module

def get_penalty_weight(warmup_step, rampup_step,current_step,LambdaMax):
    if current_step<warmup_step:
        return 0.0
    elif current_step>=warmup_step and current_step<=(rampup_step+warmup_step):
        ratio = (current_step - warmup_step)/ rampup_step
        return LambdaMax * ratio
    else:
        return LambdaMax

class ema():
    def __init__(self,beta=0.99):
        self.beta=beta
        self.ema=None
    
    def update(self,current):
        if self.ema is None:
            self.ema=current.detach().clone()
        else:
            self.ema= self.beta* self.ema + (1-self.beta)*current.detach().clone()

    def get_value(self):
        return self.ema

