import torch 

def compute_loss_penalty(R,Target_budget,penalty):
    a=torch.mean(R)
    diff= a- Target_budget
    diff = torch.clamp(diff, min=0.0)        
    Lbudget=penalty*(diff**2)
    return Lbudget


if __name__ == "__main__":
        # Test 1: All values below budget → penalty should be 0
        R_low = torch.full((2, 4, 8), 0.3)   # mean = 0.3, budget = 0.5
        print("Below budget:", compute_loss_penalty(R_low, 0.5, 1.0))  # Should be 0.0

        # Test 2: All values above budget → penalty should be positive
        R_high = torch.full((2, 4, 8), 0.8)  # mean = 0.8, budget = 0.5
        print("Above budget:", compute_loss_penalty(R_high, 0.5, 1.0)) # Should be 0.09
