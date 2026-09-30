# Graph Report - RaaLoRA  (2026-09-30)

## Corpus Check
- Corpus is ~11,951 words - fits in a single context window. You may not need a graph.

## Summary
- 42 nodes · 49 edges · 11 communities (5 shown, 6 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 4 edges (avg confidence: 0.95)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Training Pipeline
- Router Implementation
- Core Concepts
- Paper Methods
- LoRA Module
- EMA Tracker
- Toy Model
- Base Technology
- RaaLoRA Pipeline
- LLaMA Model
- Structural Pruning

## God Nodes (most connected - your core abstractions)
1. `RaaLoRA` - 7 edges
2. `RaaLoRARouter` - 6 edges
3. `RaaLoRA` - 5 edges
4. `ema` - 5 edges
5. `ToyModel` - 4 edges
6. `RaaLoRA` - 4 edges
7. `LoRA` - 3 edges
8. `Router` - 3 edges
9. `compute_loss_penalty()` - 2 edges
10. `prune_layers()` - 2 edges

## Surprising Connections (you probably didn't know these)
- `RaaLoRA` --semantically_similar_to--> `RaaLoRA`  [INFERRED] [semantically similar]
  docs/paper_text.txt → docs/raalora_explained.md
- `LoRA` --semantically_similar_to--> `LoRA`  [INFERRED] [semantically similar]
  docs/paper_text.txt → docs/raalora_explained.md
- `Average Pooling` --semantically_similar_to--> `Average Pooling`  [INFERRED] [semantically similar]
  docs/paper_text.txt → docs/raalora_explained.md
- `Budget Penalty` --semantically_similar_to--> `Budget Penalty`  [INFERRED] [semantically similar]
  docs/paper_text.txt → docs/raalora_explained.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **PEFT Methods** — docs_paper_text_raalora, docs_paper_text_lora, docs_paper_text_adalora, docs_paper_text_flexilora, docs_paper_text_peft [EXTRACTED 1.00]
- **RaaLoRA Mechanisms** — docs_raalora_explained_router, docs_raalora_explained_average_pooling, docs_raalora_explained_budget_penalty, docs_raalora_explained_ema [EXTRACTED 1.00]

## Communities (11 total, 6 thin omitted)

### Community 0 - "Training Pipeline"
Cohesion: 0.43
Nodes (3): compute_loss_penalty(), prune_layers(), get_penalty_weight()

### Community 2 - "Core Concepts"
Cohesion: 0.40
Nodes (5): Average Pooling, Budget Penalty, Average Pooling, Budget Penalty, Router

### Community 3 - "Paper Methods"
Cohesion: 0.50
Nodes (4): AdaLoRA, Dynamic Routing, FlexiLoRA, RaaLoRA

### Community 7 - "Base Technology"
Cohesion: 0.67
Nodes (3): LoRA, Parameter-Efficient Fine-Tuning, LoRA

### Community 8 - "RaaLoRA Pipeline"
Cohesion: 0.67
Nodes (3): 2-Stage Pipeline, Exponential Moving Average (EMA), RaaLoRA

## Knowledge Gaps
- **4 isolated node(s):** `AdaLoRA`, `FlexiLoRA`, `Parameter-Efficient Fine-Tuning`, `unsloth/Llama-3.2-1B-Instruct`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 19 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RaaLoRA` connect `LoRA Module` to `Training Pipeline`, `Router Implementation`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `RaaLoRARouter` connect `Router Implementation` to `Training Pipeline`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `ema` connect `EMA Tracker` to `Training Pipeline`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **What connects `AdaLoRA`, `FlexiLoRA`, `Parameter-Efficient Fine-Tuning` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._