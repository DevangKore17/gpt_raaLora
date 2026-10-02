
import torch
from torch.nn import Module, Linear
from RaaLoRA_Linear import RaaLoRA
from router import RaaLoRARouter
from scheduler import ema , get_penalty_weight
from pruner import prune_layers

class ToyModel(Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.layer1 = Linear(d_model, d_model)
        self.layer2 = Linear(d_model, d_model)
        self.head = Linear(d_model, 10)  # output 10 classes

    def forward(self, x, router):
        # 1. Ask the router for the decisions for all layers
        # Shape: [batch, num_layers, r_max]
        routing_matrix = router(x)
        # 2. Extract the specific row for layer 1 (index 0)
        scale_1 = routing_matrix[:, 0, :]
        x = torch.relu(self.layer1(x, scale_1))

        # 3. Extract the specific row for layer 2 (index 1)
        scale_2 = routing_matrix[:, 1, :]
        x = torch.relu(self.layer2(x, scale_2))
        # 4. Final output layer (this one wasn't swapped, so it just takes x)
        x = self.head(x)

        # 5. Return both the final predictions AND the routing matrix
        # (we need the routing matrix to calculate the budget penalty later!)
        return x, routing_matrix

model = ToyModel(d_model=64)
original = model.layer1
in_f = original.in_features
out_f = original.out_features
# Create RaaLoRa replacement
new_layer1 = RaaLoRA(in_f, out_f, alpha=32, R_max=8)
# Copy pretrained weights into the frozen base layer
new_layer1.base_layer.weight.data = original.weight.data.clone()
# Put it back on the model
setattr(model, "layer1", new_layer1)

original = model.layer2
in_f = original.in_features
out_f = original.out_features
# Create RaaLoRa replacement
new_layer2= RaaLoRA(in_f, out_f, alpha=32, R_max=8)
# Copy pretrained weights into the frozen base layer
new_layer2.base_layer.weight.data = original.weight.data.clone()
# Put it back on the model
setattr(model, "layer2", new_layer2)


#Training loop

# Setup lists and EMA
lora_layers = [model.layer1, model.layer2]
router = RaaLoRARouter(64,2,8)
ema_tracker = ema(beta=0.99)

# Two separate optimizers because router needs a faster learning rate!
optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer_router = torch.optim.Adam(router.parameters(), lr=1e-3)
# Dummy dataset: 100 batches
num_batches = 100
batch_size = 2
seq_len = 10
d_model = 64

import torch.nn.functional as F
from loss import compute_loss_penalty # (Make sure you imported your loss function!)

for step in range(num_batches):
    # 1. Fake Data
    x = torch.randn(batch_size, seq_len, d_model)
    targets = torch.randint(0, 10, (batch_size, seq_len))

    # 2. Forward Pass
    predictions, routing_matrix = model(x, router)

    # 3. Task Loss
    # We flatten the predictions and targets to match what cross_entropy expects
    task_loss = F.cross_entropy(predictions.view(-1, 10), targets.view(-1))

    # 4. Budget Penalty
    # Figure out the lambda weight for this step (warmup=20, ramp=50, max=1.0)
    lambda_weight = get_penalty_weight(warmup_step=20, rampup_step=50, current_step=step, LambdaMax=1.0)

    # Calculate the budget loss using your compute_loss_penalty function
    budget_loss = compute_loss_penalty(R=routing_matrix, Target_budget=0.5, penalty=lambda_weight)

    # 5. Total Loss
    total_loss = task_loss + budget_loss

    # 6. Backpropagation
    optimizer_model.zero_grad()
    optimizer_router.zero_grad()

    total_loss.backward()

    optimizer_model.step()
    optimizer_router.step()

    # 7. Update EMA Tracker
    # The routing_matrix is [batch, num_layers, r_max].
    # We want to average it across the batch dimension (dim=0) before saving it.
    mean_route_for_step = torch.mean(routing_matrix, dim=0)
    ema_tracker.update(mean_route_for_step)

    if step % 10 == 0:
        print(f"Step {step} | Task Loss: {task_loss.item():.4f} | Budget Loss: {budget_loss.item():.4f} | Lambda: {lambda_weight:.2f}")

    
print("\n--- Starting Pruning Phase ---")

# 1. Get the final smoothed routing decisions from the EMA tracker
final_routing_matrix = ema_tracker.get_value()
print("Final EMA Routing Matrix shape:", final_routing_matrix.shape)

# 2. Before pruning, print the original ranks
print(f"Layer 1 rank before: {model.layer1.R_max}")
print(f"Layer 2 rank before: {model.layer2.R_max}")

# 3. Perform surgery!
# We pass the final matrix, our list of layers, and a threshold (e.g. 0.5)
prune_layers(final_routing_matrix, lora_layers, threshold=0.5)

# 4. Check the results
print(f"Layer 1 rank AFTER pruning: {model.layer1.R_max}")
print(f"Layer 2 rank AFTER pruning: {model.layer2.R_max}")

print("\nRaaLoRA Pipeline Completed Successfully!")
