# DeltaCache

**Incremental Computation-Aware KV Cache Management for LLM Inference**

DeltaCache is a high-performance KV cache management library that dramatically reduces redundant computation in LLM inference by intelligently reusing cached key-value states across requests with shared prefixes.

## Key Features

- **Prefix-Aware Caching**: Automatically detects and reuses KV cache for shared prefixes (system prompts, few-shot examples, RAG documents)
- **Incremental Computation**: Only computes KV for new tokens, reducing computation by up to 90%+
- **Smart Eviction**: Multiple eviction policies (LRU, LFU, Composite, Tiered, Adaptive) optimized for different workloads
- **Memory Management**: Tiered GPU/CPU memory with automatic offloading
- **vLLM Integration**: Drop-in replacement for vLLM's cache engine
- **CLI Tools**: Command-line interface for benchmarks and analysis

## Performance Highlights

| Scenario | Hit Rate | Token Reuse | Speedup |
|----------|----------|-------------|---------|
| System Prompt (500 tokens) | 99% | 71.8% | 1.96x |
| RAG Documents (1000 tokens) | 80% | 76.2% | ~1.8x |
| 10-Shot Learning | 98% | 93.3% | ~3.2x |

## Installation

### From Source

```bash
git clone https://github.com/your-org/deltacache.git
cd deltacache
pip install -e .

# With HuggingFace integration
pip install -e ".[hf]"

# With CLI tools
pip install -e ".[cli]"

# With all features
pip install -e ".[all]"
```

### Dependencies

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- (Optional) transformers >= 4.30.0 for HuggingFace integration
- (Optional) click >= 8.0.0 for CLI tools
- (Optional) vLLM >= 0.4.0 for vLLM integration

## Quick Start

### Basic Usage

```python
from deltacache import DeltaCacheManager, DeltaCacheConfig

# Create manager with model-specific config
config = DeltaCacheConfig.for_model("llama-3-8b")
manager = DeltaCacheManager(config)

# Define a KV computation function
def compute_kv(input_ids, position_ids, past_kv=None):
    # Your model's KV computation here
    return key_cache, value_cache

# Process requests with automatic caching
system_prompt_tokens = [1, 2, 3, ...]  # Your tokenized system prompt

# First request: computes all tokens
result1 = manager.compute_incremental(system_prompt_tokens + query1_tokens, compute_kv)

# Second request: reuses cached system prompt KV
result2 = manager.compute_incremental(system_prompt_tokens + query2_tokens, compute_kv)
print(f"Reused {result2.matched_length} tokens!")  # Reused system prompt
```

### With HuggingFace Models

```python
from deltacache import DeltaCacheManager
from deltacache.hf_integration import GPT2Adapter

# Load model with adapter
adapter = GPT2Adapter.from_pretrained("gpt2", device="cuda")
manager = DeltaCacheManager(adapter.config)

# Process prompts with automatic prefix caching
system_prompt = "You are a helpful assistant. "
for query in queries:
    tokens = adapter.tokenize(system_prompt + query)[0].tolist()
    result = manager.compute_incremental(tokens, adapter)
    print(f"Computed: {result.computed_length}, Cached: {result.matched_length}")
```

### CLI Usage

```bash
# Run benchmarks
deltacache benchmark --model gpt2 --scenario system_prompt_short

# Analyze prefix patterns in a dataset
deltacache analyze --dataset conversations.json --tokenizer gpt2

# View experiment statistics
deltacache stats --results-file results.json
```

## Architecture

```
deltacache/
├── core/                    # Core data structures
│   ├── prefix_tree.py       # Trie-based prefix index
│   ├── cache_block.py       # KV cache block management
│   └── memory_pool.py       # GPU/CPU memory pool
├── engine/                  # Computation engine
│   ├── incremental.py       # Incremental KV computation
│   └── rope_handler.py      # RoPE position encoding
├── eviction/                # Cache eviction
│   └── policy.py            # LRU, LFU, Tiered, Adaptive policies
├── hf_integration/          # HuggingFace integration
│   ├── model_adapter.py     # Base adapter class
│   └── gpt2_adapter.py      # GPT-2 specific adapter
├── vllm_integration/        # vLLM integration
│   ├── cache_engine.py      # vLLM cache engine replacement
│   └── scheduler_hook.py    # Prefix-aware scheduling
├── cli/                     # Command-line interface
│   └── commands/            # CLI commands
└── api.py                   # Unified API (DeltaCacheManager)
```

## Use Cases

### 1. System Prompt Caching

When many requests share the same system prompt:

```python
# System prompt is cached after first request
system_tokens = tokenizer.encode("You are a helpful coding assistant...")

for user_query in queries:
    tokens = system_tokens + tokenizer.encode(user_query)
    result = manager.compute_incremental(tokens, compute_fn)
    # Only user_query tokens are computed!
```

### 2. RAG Document Caching

When the same documents are queried multiple times:

```python
# Pre-cache documents
for doc in documents:
    doc_tokens = tokenizer.encode(doc.text)
    manager.compute_incremental(doc_tokens, compute_fn)

# Queries reuse cached document KV
for query in queries:
    tokens = doc_tokens + tokenizer.encode(query.text)
    result = manager.compute_incremental(tokens, compute_fn)
```

### 3. Few-Shot Learning

When using shared examples across requests:

```python
# Few-shot examples (expensive, reused across all requests)
examples_tokens = tokenizer.encode(few_shot_prompt)

for input_text in classification_inputs:
    tokens = examples_tokens + tokenizer.encode(input_text)
    result = manager.compute_incremental(tokens, compute_fn)
    # 90%+ tokens reused from cached examples!
```

## Configuration

```python
from deltacache import DeltaCacheConfig

config = DeltaCacheConfig(
    # Model parameters
    num_layers=32,
    num_heads=32,
    head_dim=128,

    # Memory limits
    gpu_memory_limit=8 * 1024**3,  # 8GB
    cpu_memory_limit=16 * 1024**3,  # 16GB

    # Eviction settings
    eviction_policy="tiered",  # Options: lru, lfu, composite, tiered, adaptive
    eviction_threshold=0.9,

    # Device
    device="cuda",
    dtype="float16",
)

# Or use model presets
config = DeltaCacheConfig.for_model("llama-3-8b")
config = DeltaCacheConfig.for_model("gpt2")
```

### Supported Model Presets

- GPT-2: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- Llama: `llama-7b`, `llama-13b`, `llama-70b`
- Llama 2: `llama-2-7b`, `llama-2-13b`, `llama-2-70b`
- Llama 3: `llama-3-8b`, `llama-3-70b`
- Mistral: `mistral-7b`, `mixtral-8x7b`
- Qwen: `qwen-7b`, `qwen-14b`, `qwen-72b`

## Eviction Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| `lru` | Least Recently Used | General purpose |
| `lfu` | Least Frequently Used | Stable workloads |
| `composite` | Multi-factor scoring | Mixed workloads |
| `tiered` | GPU→CPU→Delete | Memory constrained |
| `adaptive` | Online learning | Dynamic workloads |

## Benchmarks

Run benchmarks to measure performance:

```bash
# Run specific scenario
deltacache benchmark --model gpt2 --scenario system_prompt_long

# Run all scenarios
deltacache benchmark --model gpt2 --scenario all --output-json results.json

# List available scenarios
deltacache benchmark --list-scenarios
```

Available scenarios:
- `system_prompt_short`: Short system prompt sharing
- `system_prompt_long`: Long system prompt sharing
- `multi_turn_conversation`: Multi-turn dialogue
- `document_qa`: RAG document caching
- `few_shot_classification`: Few-shot learning
- `no_sharing_baseline`: Baseline without sharing

## API Reference

### DeltaCacheManager

```python
class DeltaCacheManager:
    def lookup(self, tokens: List[int]) -> LookupResult
    def insert(self, tokens: List[int], key_cache, value_cache) -> None
    def compute_incremental(self, tokens: List[int], compute_fn) -> IncrementalResult
    def evict_if_needed(self, required_memory: int) -> EvictionResult
    def get_stats(self) -> Dict[str, Any]
    def reset(self) -> None
```

### LookupResult

```python
@dataclass
class LookupResult:
    matched_length: int          # Tokens matched from cache
    key_cache: Optional[Tensor]  # Cached key tensor
    value_cache: Optional[Tensor] # Cached value tensor
```

### IncrementalResult

```python
@dataclass
class IncrementalResult:
    matched_length: int      # Tokens from cache
    computed_length: int     # Tokens computed
    key_cache: Tensor        # Complete key cache
    value_cache: Tensor      # Complete value cache
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_prefix_tree.py -v

# Run with coverage
pytest --cov=deltacache
```

## License

Apache 2.0

## Citation

```bibtex
@software{deltacache,
  title = {DeltaCache: Incremental Computation-Aware KV Cache Management},
  year = {2024},
  url = {https://github.com/your-org/deltacache}
}
```
