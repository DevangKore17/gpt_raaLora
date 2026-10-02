# ============================================================
# train_gpt2_e2e.py — RaaLoRA Baseline on E2E NLG Challenge
# ============================================================
# This script trains GPT-2 on the E2E NLG dataset matching the
# exact hyperparameter setup from the benchmark spec:
# - Batch size 8, 5 Epochs
# - AdamW, lr=0.0002, 500 warmup steps, linear decay
# - Weight decay 0.01, label smoothing 0.1, dropout 0.1
# - Adapts W_q and W_v using rank r=4, alpha=32
# - Evaluates best epoch and uses Beam Search (10, len=0.9, ngram=4)
# ============================================================

import torch
import torch.nn.functional as F
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from router import RaaLoRARouter
from RaaLoRA_GPT2 import RaaLoRA_GPT2_c_attn
from loss import compute_loss_penalty
from scheduler import get_penalty_weight, ema
from pruner_gpt2 import prune_gpt2_layers

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main(seed=42, use_tpu=False, inference_only=False):
    set_seed(seed)
    print(f"\n{'='*60}\nStarting GPT-2 E2E Baseline (Seed {seed})\n{'='*60}")
    
    USE_TPU = use_tpu

    if USE_TPU:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla
        device = torch_xla.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # HYPERPARAMETERS
    # ============================================================
    MODEL_NAME = "gpt2"
    BATCH_SIZE = 8
    EPOCHS = 5
    LR = 0.0002
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    DROPOUT = 0.1
    R_MAX = 4
    ALPHA = 32
    TARGET_BUDGET = 0.5
    
    # ============================================================
    # MODEL & TOKENIZER
    # ============================================================
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 has no pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME, 
        resid_pdrop=DROPOUT, 
        embd_pdrop=DROPOUT, 
        attn_pdrop=DROPOUT
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # ============================================================
    # SWAP LAYERS WITH CUSTOM GPT2 RaaLoRA
    # ============================================================
    lora_layers = []
    layer_names = []
    targets_to_swap = []
    
    # In GPT-2, attention is in 'c_attn'. We find them all.
    for name, module in model.named_modules():
        if name.endswith("c_attn"):
            targets_to_swap.append((name, module))

    for name, module in targets_to_swap:
        in_f = module.weight.shape[0] # GPT2 Conv1D uses [in, out]
        
        # Create our specialized GPT-2 layer that adapts only Q and V
        new_layer = RaaLoRA_GPT2_c_attn(module, in_features=in_f, alpha=ALPHA, R_max=R_MAX)
        
        parent_name, attr_name = name.rsplit(".", 1)
        parent_module = dict(model.named_modules())[parent_name]
        setattr(parent_module, attr_name, new_layer)
        
        lora_layers.append(new_layer)
        layer_names.append(name)
        
    d_model = model.config.n_embd
    router = RaaLoRARouter(d_module=d_model, num_layers=len(lora_layers), r_max=R_MAX, bottleneck_dim=256)
    
    model.to(device)
    router.to(device)
    
    # ============================================================
    # DATASET PREP (e2e_nlg)
    # ============================================================
    print("\nLoading GEM/e2e_nlg dataset (Parquet version)...")
    dataset = load_dataset("GEM/e2e_nlg", trust_remote_code=True)
    
    def tokenize_function(examples):
        # E2E format: meaning representation -> human readable text
        prompts = [f"{mr} => {txt}{tokenizer.eos_token}" for mr, txt in zip(examples['meaning_representation'], examples['target'])]
        encodings = tokenizer(prompts, truncation=True, max_length=256, padding="max_length")
        
        labels = []
        for ids, mask in zip(encodings["input_ids"], encodings["attention_mask"]):
            label = list(ids)
            label = [l if m == 1 else -100 for l, m in zip(label, mask)]
            labels.append(label)
        encodings["labels"] = labels
        return encodings
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    tokenized_datasets.set_format("torch")
    
    train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)

    # ============================================================
    # OPTIMIZERS & SCHEDULER
    # ============================================================
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params + list(router.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)
    ema_tracker = ema(beta=0.99)
    
    # Custom CrossEntropy with Label Smoothing
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=-100)

    # ============================================================
    # TRAINING LOOP
    # ============================================================
    best_val_loss = float('inf')
    best_model_state = None
    
    save_path = "gpt2_raa_lora_best.pt"
    if inference_only:
        print(f"\nSkipping training. Loading weights from {save_path}...")
        best_model_state = torch.load(save_path, map_location="cpu")
    else:
        print("\nStarting Training...")
        
    global_step = 0
    for epoch in (range(EPOCHS) if not inference_only else []):
        model.train()
        router.train()
        train_loss = 0.0
        pending_loss = torch.tensor(0.0, device=device)
        
        if USE_TPU:
            epoch_iterator = pl.ParallelLoader(train_dataloader, [device]).per_device_loader(device)
        else:
            epoch_iterator = train_dataloader
            
        for batch in tqdm(epoch_iterator, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.no_grad():
                embeddings = model.transformer.wte(input_ids)
                
            routing_matrix = router(embeddings)
            for layer_idx, layer in enumerate(lora_layers):
                layer.current_scale = routing_matrix[:, layer_idx, :]
                
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute smoothed loss manually
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Router Penalty
            lambda_weight = get_penalty_weight(50, 100, global_step, 1.0)
            budget_loss = compute_loss_penalty(routing_matrix, TARGET_BUDGET, lambda_weight)
            
            total_loss = task_loss + budget_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params + list(router.parameters()), 1.0)
            
            if USE_TPU:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
                
            scheduler.step()
            
            ema_tracker.update(torch.mean(routing_matrix, dim=0))
            pending_loss += task_loss.detach()
            if global_step % 50 == 0:
                train_loss += pending_loss.item()
                pending_loss = torch.tensor(0.0, device=device)
            global_step += 1
            
        train_loss += pending_loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        if USE_TPU:
            val_epoch_iterator = pl.ParallelLoader(val_dataloader, [device]).per_device_loader(device)
        else:
            val_epoch_iterator = val_dataloader
            
        with torch.no_grad():
            for batch in tqdm(val_epoch_iterator, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                embeddings = model.transformer.wte(input_ids)
                routing_matrix = router(embeddings)
                for layer_idx, layer in enumerate(lora_layers):
                    layer.current_scale = routing_matrix[:, layer_idx, :]
                    
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                val_loss += loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
        avg_val = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_dataloader):.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_state = {
                'model': {n: p.cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
                'router': {n: p.cpu().clone() for n, p in router.named_parameters()},
                'ema': ema_tracker.get_value().cpu().clone()
            }
            
    # ============================================================
    # PRUNING (ON BEST EPOCH)
    # ============================================================
    print("\nRestoring best model for pruning...")
    for n, p in model.named_parameters():
        if p.requires_grad: p.data.copy_(best_model_state['model'][n])
    for n, p in router.named_parameters():
        p.data.copy_(best_model_state['router'][n])
        
    final_routing_matrix = best_model_state['ema'].to(device)
    prune_gpt2_layers(final_routing_matrix, lora_layers, threshold=TARGET_BUDGET)
    
    if not inference_only:
        print(f"\nSaving best model weights to {save_path}...")
        torch.save(best_model_state, save_path)
    
    # ============================================================
    # INFERENCE & EVALUATION (E2E Benchmark)
    # ============================================================
    print("\n" + "="*60)
    print("Starting E2E Benchmark Evaluation...")
    print("="*60)
    model.eval()

    # 1. Effective Average Rank
    avg_rank = sum(layer.R_max for layer in lora_layers) / len(lora_layers)
    print(f"\n[Adaptive Routing Benchmark] Effective Average Rank: {avg_rank:.2f} (Baseline LoRA = 4.0)")

    # 2. Generation & Inference Overhead
    print("\nGenerating predictions on test set (Beam=10, Len_Pen=0.9, No_Rep_Ngram=4)...")
    
    import time
    total_generated_tokens = 0
    total_generation_time = 0.0
    
    predictions = []
    # E2E test set can be large; you can subset this for quick testing if needed
    for i, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        
        # We only want the prompt (meaning representation). 
        # For HF E2E dataset, we must extract the prompt up to " => " manually,
        # but since we already tokenized with padding, the input_ids contain the gold response.
        # To strictly do inference, we should tokenize JUST the prompts.
        # We do it dynamically here for the benchmark:
        prompts = [tokenizer.decode(ids, skip_special_tokens=True).split(" => ")[0] + " => " for ids in input_ids]
        tokenizer.padding_side = "left"
        prompt_encodings = tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(device)
        
        for layer in lora_layers:
            layer.current_scale = torch.ones(prompt_encodings.input_ids.size(0), layer.R_max).to(device)
            
        start_time = time.time()
        
        generated_ids = model.generate(
            prompt_encodings.input_ids,
            attention_mask=prompt_encodings.attention_mask,
            max_new_tokens=60,
            num_beams=10,
            length_penalty=0.9,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id
        )
        
        end_time = time.time()
        
        # Calculate tokens generated and latency
        new_tokens = generated_ids.shape[1] - prompt_encodings.input_ids.shape[1]
        total_generated_tokens += (new_tokens * generated_ids.shape[0])
        total_generation_time += (end_time - start_time)
        
        # Decode and store
        prompt_len = prompt_encodings.input_ids.shape[1]
        for gen_ids in generated_ids:
            # Only keep the newly generated text
            pred_text = tokenizer.decode(gen_ids[prompt_len:], skip_special_tokens=True)
            predictions.append(pred_text.strip().replace("\n", " "))
            
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_dataloader)} batches...")

    # Print Latency Overhead
    ms_per_token = (total_generation_time / total_generated_tokens) * 1000
    print(f"\n[Adaptive Routing Benchmark] Inference Overhead:")
    print(f"  Latency: {ms_per_token:.2f} ms per token")

    # 3. Save Predictions for Official e2e-metrics script
    pred_path = "e2e_predictions.txt"
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
        
    print(f"\nPredictions saved to {pred_path}")
    print("\nTo calculate BLEU, NIST, METEOR, ROUGE-L, and CIDEr:")
    print("  1. Clone the official script: git clone https://github.com/tuetschek/e2e-metrics")
    print("  2. Format the gold references from the dataset.")
    print("  3. Run: ./e2e-metrics/measure_scores.py gold_references.txt e2e_predictions.txt")
    print("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed (benchmark requires 3)")
    parser.add_argument("--use_tpu", action="store_true", help="Set this flag to use TPU via PyTorch XLA")
    parser.add_argument("--inference_only", action="store_true", help="Skip training and run inference using gpt2_raa_lora_best.pt")
    args = parser.parse_args()
    main(args.seed, args.use_tpu, args.inference_only)

