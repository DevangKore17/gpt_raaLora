# ============================================================
# train_llama.py — RaaLoRA Training on a Real LLM
# ============================================================
# This script trains LLaMA-3.2-1B-Instruct using the RaaLoRA
# 2-stage pipeline: Unified Training + Short Polish.
# Compatible with both GPU and TPU (Colab).
# ============================================================

import torch
from torch.nn import Linear
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from router import RaaLoRARouter
from RaaLoRA_Linear import RaaLoRA
from loss import compute_loss_penalty
from scheduler import get_penalty_weight, ema
from pruner import prune_layers

# ============================================================
# CONFIGURATION — Tweak these values as needed
# ============================================================
USE_TPU = False                          # Set True on Colab TPU
MODEL_NAME = "openai-community/gpt2-medium"

# RaaLoRA Hyperparameters (from the paper)
R_MAX = 16                               # Maximum rank per layer
ALPHA = 32                               # LoRA scaling factor
BOTTLENECK_DIM = 256                     # Router bottleneck size
TARGET_MODULES = ["q_proj", "v_proj"]    # Which layers to apply RaaLoRA to

# Training Hyperparameters
MAX_SEQ_LEN = 512                        # Max tokens per sample
BATCH_SIZE = 1                           # Keep small for memory
NUM_TRAIN_SAMPLES = 1500                 # Total training samples
LEARNING_RATE_MODEL = 2e-4               # LR for LoRA A/B matrices
LEARNING_RATE_ROUTER = 1e-3              # LR for router (faster)

# Budget Penalty Schedule
WARMUP_STEPS = 50                        # Steps before penalty kicks in
RAMPUP_STEPS = 100                       # Steps to ramp penalty from 0 → max
LAMBDA_MAX = 1.0                         # Maximum penalty weight
TARGET_BUDGET = 0.5                      # Target average routing score

# EMA
EMA_BETA = 0.99

# Polish
POLISH_STEPS = 100                       # Short polish after pruning

# ============================================================
# DEVICE SETUP
# ============================================================
if USE_TPU:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

# ============================================================
# LOAD MODEL AND TOKENIZER
# ============================================================
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,           # float32 for TPU; use float16 for GPU if needed
)

# LLaMA doesn't have a pad token by default — use eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Freeze the ENTIRE base model — we only train LoRA A/B and the router
for param in model.parameters():
    param.requires_grad = False

print(f"Model loaded: {MODEL_NAME}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Num layers: {model.config.num_hidden_layers}")

# ============================================================
# SWAP TARGET LAYERS WITH RaaLoRA
# ============================================================
print("\nSwapping layers with RaaLoRA...")

lora_layers = []    # List of all RaaLoRA layer objects
layer_names = []    # Their names in the model (for logging)

# We need to collect all targets first, then swap
# (Can't modify the model while iterating over it)
targets_to_swap = []
for name, module in model.named_modules():
    if any(name.endswith(t) for t in TARGET_MODULES):
        if isinstance(module, Linear):
            targets_to_swap.append((name, module))

# Now swap each target
for name, module in targets_to_swap:
    in_f = module.in_features
    out_f = module.out_features

    # Create the RaaLoRA replacement
    new_layer = RaaLoRA(in_f, out_f, alpha=ALPHA, R_max=R_MAX, bias=(module.bias is not None))

    # Copy the pretrained weights into the frozen base layer
    new_layer.base_layer.weight.data = module.weight.data.clone()
    if module.bias is not None:
        new_layer.base_layer.bias.data = module.bias.data.clone()

    # Find the parent module and replace the attribute
    # e.g., for "model.layers.0.self_attn.q_proj":
    #   parent = model.layers.0.self_attn
    #   attr   = "q_proj"
    parent_name, attr_name = name.rsplit(".", 1)
    parent_module = dict(model.named_modules())[parent_name]
    setattr(parent_module, attr_name, new_layer)

    lora_layers.append(new_layer)
    layer_names.append(name)
    print(f"  ✓ {name} ({in_f} → {out_f})")

num_lora_layers = len(lora_layers)
print(f"\nTotal RaaLoRA layers swapped: {num_lora_layers}")

# ============================================================
# CREATE THE ROUTER
# ============================================================
d_model = model.config.hidden_size    # 2048 for LLaMA-1B

router = RaaLoRARouter(
    d_model=d_model,
    num_layers=num_lora_layers,
    r_max=R_MAX,
    bottleneck_dim=BOTTLENECK_DIM
)

print(f"Router created: d_model={d_model}, layers={num_lora_layers}, r_max={R_MAX}")

# ============================================================
# MOVE EVERYTHING TO DEVICE
# ============================================================
model = model.to(device)
router = router.to(device)

# ============================================================
# SETUP OPTIMIZERS
# ============================================================
# Only train: LoRA A/B matrices (inside the model) + router weights
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer_model = torch.optim.Adam(lora_params, lr=LEARNING_RATE_MODEL)
optimizer_router = torch.optim.Adam(router.parameters(), lr=LEARNING_RATE_ROUTER)

# EMA tracker for routing decisions
ema_tracker = ema(beta=EMA_BETA)

print(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")
print(f"Router parameters: {sum(p.numel() for p in router.parameters()):,}")

# ============================================================
# LOAD DATASET
# ============================================================
print("\nLoading dataset...")

# Using Alpaca dataset as an example.
# To use your own JSONL file instead, replace with:
#   dataset = load_dataset("json", data_files="your_data.jsonl", split="train")
dataset = load_dataset("tatsu-lab/alpaca", split="train")
dataset = dataset.shuffle(seed=42).select(range(NUM_TRAIN_SAMPLES))

print(f"Dataset: {len(dataset)} samples")


def format_sample(sample):
    """Format a single sample into an instruction-following prompt."""
    if sample.get("input", ""):
        text = (f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Input:\n{sample['input']}\n\n"
                f"### Response:\n{sample['output']}{tokenizer.eos_token}")
    else:
        text = (f"### Instruction:\n{sample['instruction']}\n\n"
                f"### Response:\n{sample['output']}{tokenizer.eos_token}")
    return text


def tokenize_text(text):
    """Tokenize a single text string and return tensors on device."""
    encodings = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # For causal LM: labels = input_ids, but ignore padding tokens
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100    # -100 tells PyTorch to skip these in loss

    return input_ids, attention_mask, labels


# ============================================================
# STAGE 1: UNIFIED TRAINING WITH DYNAMIC ROUTING
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 1: Unified Training with Dynamic Routing")
print("=" * 60)

model.train()
router.train()

total_steps = NUM_TRAIN_SAMPLES // BATCH_SIZE

for step in range(total_steps):
    # --- 1. Prepare a Single Sample ---
    sample = dataset[step % len(dataset)]
    text = format_sample(sample)
    input_ids, attention_mask, labels = tokenize_text(text)

    # --- 2. Get Input Embeddings for the Router ---
    # The router needs to see the raw text embeddings to make routing decisions.
    # We grab them from the model's embedding layer WITHOUT gradients
    # (the router will add its own gradient path).
    with torch.no_grad():
        embeddings = model.model.embed_tokens(input_ids)

    # --- 3. Compute Routing Matrix ---
    # Shape: [batch=1, num_lora_layers, r_max]
    routing_matrix = router(embeddings)

    # --- 4. Pre-load Each Layer's Scale ---
    # Since HuggingFace controls the forward pass internally,
    # we store each layer's routing scale on the layer itself.
    # The layer's forward() will read self.current_scale automatically.
    for layer_idx, layer in enumerate(lora_layers):
        layer.current_scale = routing_matrix[:, layer_idx, :]

    # --- 5. Forward Pass (HuggingFace handles this) ---
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    task_loss = outputs.loss

    # --- 6. Compute Budget Penalty ---
    lambda_weight = get_penalty_weight(
        warmup_step=WARMUP_STEPS,
        rampup_step=RAMPUP_STEPS,
        current_step=step,
        LambdaMax=LAMBDA_MAX
    )
    budget_loss = compute_loss_penalty(
        R=routing_matrix,
        Target_budget=TARGET_BUDGET,
        penalty=lambda_weight
    )

    # --- 7. Total Loss = Task Loss + Budget Penalty ---
    total_loss = task_loss + budget_loss

    # --- 8. Backpropagation ---
    optimizer_model.zero_grad()
    optimizer_router.zero_grad()

    total_loss.backward()

    # Gradient clipping (prevents exploding gradients on large models)
    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)

    if USE_TPU:
        xm.optimizer_step(optimizer_model)
        xm.optimizer_step(optimizer_router)
    else:
        optimizer_model.step()
        optimizer_router.step()

    # --- 9. Update EMA Tracker ---
    mean_routing = torch.mean(routing_matrix, dim=0)    # Average over batch
    ema_tracker.update(mean_routing)

    # --- 10. Logging ---
    if step % 50 == 0:
        avg_route = routing_matrix.mean().item()
        print(f"  Step {step:>4d}/{total_steps} | "
              f"Task: {task_loss.item():.4f} | "
              f"Budget: {budget_loss.item():.6f} | "
              f"λ: {lambda_weight:.2f} | "
              f"Avg Route: {avg_route:.3f}")

print("\nStage 1 complete.")

# ============================================================
# PRUNING PHASE
# ============================================================
print("\n" + "=" * 60)
print("  PRUNING PHASE")
print("=" * 60)

# Get the final smoothed routing decisions
final_routing_matrix = ema_tracker.get_value()
print(f"EMA Matrix shape: {final_routing_matrix.shape}")

# Show ranks before pruning
print("\nBefore pruning:")
for name, layer in zip(layer_names, lora_layers):
    print(f"  {name}: rank = {layer.R_max}")

# Perform the surgery — physically chop down A and B matrices
prune_layers(final_routing_matrix, lora_layers, threshold=TARGET_BUDGET)

# Show ranks after pruning
print("\nAfter pruning:")
total_params_saved = 0
for name, layer in zip(layer_names, lora_layers):
    old_rank = R_MAX
    new_rank = layer.R_max
    saved = (old_rank - new_rank) * (layer.MatA.shape[0] + layer.MatB.shape[1])
    total_params_saved += saved
    print(f"  {name}: rank = {new_rank} (was {old_rank}, saved {saved:,} params)")

print(f"\nTotal parameters saved by pruning: {total_params_saved:,}")

# ============================================================
# STAGE 2: SHORT POLISH (No Router)
# ============================================================
print("\n" + "=" * 60)
print("  STAGE 2: Short Polish")
print("=" * 60)

# After pruning, the router is GONE. We train with fixed ranks.
# Use a lower learning rate for fine-tuning the pruned weights.
polish_params = [p for p in model.parameters() if p.requires_grad]
polish_optimizer = torch.optim.Adam(polish_params, lr=LEARNING_RATE_MODEL * 0.1)

model.train()
# Router is no longer used — delete it to free memory
del router
del optimizer_router

dataset_shuffled = dataset.shuffle(seed=123)

for step in range(POLISH_STEPS):
    sample = dataset_shuffled[step % len(dataset_shuffled)]
    text = format_sample(sample)
    input_ids, attention_mask, labels = tokenize_text(text)

    # Set all routing scales to 1.0 — every remaining rank dimension is fully active
    for layer in lora_layers:
        layer.current_scale = torch.ones(input_ids.size(0), layer.R_max).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss

    polish_optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(polish_params, max_norm=1.0)

    if USE_TPU:
        xm.optimizer_step(polish_optimizer)
    else:
        polish_optimizer.step()

    if step % 20 == 0:
        print(f"  Polish Step {step:>3d}/{POLISH_STEPS} | Loss: {loss.item():.4f}")

print("\nStage 2 complete.")

# ============================================================
# SAVE THE FINAL LORA WEIGHTS
# ============================================================
print("\n" + "=" * 60)
print("  SAVING MODEL")
print("=" * 60)

# Save only the LoRA weights (not the full 1B base model)
lora_state = {}
for name, layer in zip(layer_names, lora_layers):
    lora_state[f"{name}.MatA"] = layer.MatA.data.cpu()
    lora_state[f"{name}.MatB"] = layer.MatB.data.cpu()
    lora_state[f"{name}.R_max"] = layer.R_max
    lora_state[f"{name}.alpha"] = layer.alpha

torch.save(lora_state, "raalora_weights.pt")
print("Saved LoRA weights to: raalora_weights.pt")

# Print final summary
print("\n" + "=" * 60)
print("  RaaLoRA PIPELINE COMPLETE")
print("=" * 60)
print(f"  Base Model:     {MODEL_NAME}")
print(f"  Layers adapted: {num_lora_layers}")
print(f"  Original R_max: {R_MAX}")
print(f"  Final ranks:")
for name, layer in zip(layer_names, lora_layers):
    print(f"    {name}: {layer.R_max}")
print(f"  Weights saved:  raalora_weights.pt")
print("=" * 60)
