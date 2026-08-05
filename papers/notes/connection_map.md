# Research Connection Map
# Your Projects ↔ Beidi Chen Group (Updated with All Paper Insights)

## Visual Map

```
                    ┌──────────────────────────────────────────────┐
                    │           Beidi Chen Group                    │
                    │                                              │
  KV Cache ────────►│  H2O ──── ShadowKV ──── Sirius              │
  Management        │  (NeurIPS'23)  (ICML'25)  (NeurIPS'24)      │
       │            │     │                                        │
       │            │     │ both do KV selection                   │
       │            │     ▼                                        │
       │            │  TriForce ◄── Sequoia ◄── SpecInfer          │
       │            │  (COLM'24)   (NeurIPS'24)  (ASPLOS'24)      │
       │            │     │  3-layer     │  tree DP                │
       │            │     │  hierarchy   │                         │
       │            │     ▼              ▼                         │
       │            │  MagicDec     SpecExec                       │
       │            │  (ICLR'25)   (NeurIPS'24)                   │
       │            │  fixed-window                                │
       │            │  batch spec                                  │
       │            │                                              │
       │            │  FlexGen (ICML'23) ─── offloading            │
       │            └──────────────────────────────────────────────┘
       │                   ▲          ▲           ▲
       │                   │          │           │
       │            ┌──────┴──┐ ┌─────┴─────┐ ┌───┴──────┐
       │            │ Bridge 1│ │ Bridge 2  │ │ Bridge 3 │
       │            │         │ │           │ │          │
  ┌────┴──────┐     │combined │ │edge-cloud │ │layer-    │
  │DeltaCache │─────│eviction │ │tree optim │ │aware     │
  │           │     │H2O+Delta│ │Sequoia+GC │ │retrieval │
  │prefix tree│     └─────────┘ └───────────┘ │TriForce  │
  │layer-aware│          │           │        │+DeltaCA  │
  │eviction   │          │           │        └──────────┘
  └───────────┘          │           │             │
       │                 │           │             │
       │            ┌────┴───────────┴─────────────┴───┐
       │            │           GC-Edge                  │
       └───────────►│  edge-cloud speculation            │
                    │  generational GC                   │
                    │  network-aware optimal K            │
                    │  + MagicDec fixed-window (Bridge 4) │
                    └────────────────────────────────────┘
```

## Paper Evolution Timeline

```
2023:  H2O (KV eviction)      FlexGen (offloading)     SpecInfer (tree spec)
         │                         │                         │
         │                         │                         │
2024:  ShadowKV                    │                    Sequoia (DP tree)
       Sirius                      │                         │
         │                         │                    TriForce (hierarchy)
         │                         │                         │
2025:    │                         │                    MagicDec (batch)
         │                         │                         │
         ▼                         ▼                         ▼
      KV compression           Offloading           Speculative decoding
      越来越aggressive         越来越smart            越来越scalable

Your work fits:
  DeltaCache ─────────► prefix-aware KV + layer-aware eviction
  GC-Edge ────────────► edge deployment + network-aware speculation
```

## Concrete Bridges (面试时可展开的合作点)

### Bridge 1: Combined KV Cache Management (H2O + DeltaCache)
```
H2O (token-level, attention-based)
  + DeltaCache (prefix-level, layer-aware)
  = Complete KV cache system

Pipeline:
  Request → DeltaCache prefix lookup → cache hit? reuse KV
     ↓ (for non-cached portion)
  Compute KV → H2O eviction decides which new KV to keep
     ↓
  Layer-aware priority: later layers keep more KV entries

Experiment validation:
  - H2O effectiveness peaks at EARLY layers (layer 4: Gini=0.925, 93.5% capture)
  - DeltaCache weight peaks at LATE layers (layer 20: weight=0.955)
  - Correlation = 0.194 → complementary → combined adds value
  - Natural split: aggressive H2O in early layers, preserve late layers
```

### Bridge 2: Edge-Cloud Tree Speculation (Sequoia + GC-Edge)
```
Sequoia DP (single-device, hardware-aware)
  + GC-Edge (edge-cloud, network-aware)
  = Network-aware speculative decoding

Extended DP formulation:
  maximize: E[accepted_tokens] / total_round_time
  where: total_round_time = draft(tree) + transfer(tree) + verify(tree)
  new term: transfer(tree) = 2*RTT + sizeof(tree) / bandwidth

Experiment validation:
  - Offloading: Sequoia 3.15x vs GC-Edge 2.76x (tree advantage 1.14x)
  - Edge-cloud LAN: both <1x (network RTT dominates)
  - Edge-cloud WAN: both <<1x (50ms RTT kills everything)
  - GAP: Sequoia doesn't optimize for network → opportunity for extension
```

### Bridge 3: Layer-Aware Retrieval Cache (TriForce + DeltaCache/H2O)
```
TriForce (per-query chunk retrieval, uniform budget across layers)
  + H2O insight (layer sparsity varies dramatically)
  + DeltaCache (prefix cache reuse)
  = Per-layer adaptive retrieval + prefix-aware speculation

Current TriForce: ALL layers use same budget (4096 tokens)
With H2O insight:
  Layer 4 (Gini=0.925): only needs ~1024 tokens (already very sparse)
  Layer 12 (Gini=0.705): needs ~4096 tokens (broader attention)
  → Save ~50% retrieval memory with same quality

With DeltaCache:
  Shared prefix (e.g., RAG document) → pre-computed KV
  Skip retrieval for cached portion → only select for new suffix
  → Reduce retrieval overhead from O(16K chunks) to O(suffix chunks)
```

### Bridge 4: Memory-Efficient Edge Speculation (MagicDec + GC-Edge)
```
MagicDec (fixed context window for draft → constant memory)
  + GC-Edge (edge device with limited memory + generational GC)
  = Stable, predictable edge speculation system

On edge device (8GB Jetson):
  - Draft model uses fixed window (MagicDec) → constant 257 tokens KV
  - GC manages fixed-size KV lifecycle → no fragmentation, no leaks
  - SnapKV compression selected once → minimal computation
  - Result: predictable memory, stable long-context draft on edge

With batch serving:
  - MagicDec enables batch 32-64 on datacenter
  - On edge: smaller batch (4-8) but still multiple concurrent requests
  - Fixed memory per request → can pre-allocate → no GC pressure
```

## Summary: Your Unique Value Proposition

1. **Breadth:** KV cache management + speculative decoding (该组两大方向你都有experience)
   - 验证了H2O的core hypothesis (top 20% = 84% attention)
   - 理解了Sequoia的DP, TriForce的3-layer, MagicDec的batch design

2. **Edge perspective:** 该组主要focus datacenter (A100/H100), 你bring edge device experience
   - 实验证明edge-cloud场景下network latency dominates → 需要不同优化策略
   - MagicDec fixed-window天然适合edge, 但没有人做过edge deployment

3. **System thinking:** 两个项目都是完整system, 与该组FlexGen/serving system方向match
   - 理解how pieces fit together: H2O → TriForce → MagicDec evolution

4. **Concrete extensions:** 不是泛泛说"感兴趣", 而是有明确technical bridge + 实验数据支撑
   - Network-aware DP (Sequoia extension, validated gap in simulation)
   - Layer-aware retrieval (TriForce extension, validated by H2O experiment)
   - Prefix-aware speculation (DeltaCache integration)
   - Edge memory management (MagicDec + GC combination)
