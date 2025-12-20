# DeltaCache Examples

This document provides detailed examples for common DeltaCache use cases.

## Table of Contents

1. [System Prompt Caching](#system-prompt-caching)
2. [RAG Document Caching](#rag-document-caching)
3. [Few-Shot Learning](#few-shot-learning)
4. [Multi-Turn Conversations](#multi-turn-conversations)
5. [vLLM Integration](#vllm-integration)

---

## System Prompt Caching

When serving a chatbot or assistant, most requests share the same system prompt. DeltaCache caches the system prompt's KV states and reuses them across all requests.

```python
from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import GPT2Adapter

# Initialize
adapter = GPT2Adapter.from_pretrained("gpt2", device="cuda")
manager = DeltaCacheManager(adapter.config)

# System prompt (cached once, reused for all requests)
SYSTEM_PROMPT = """You are a helpful AI assistant. You provide accurate,
concise answers to user questions. Always be respectful and professional."""

system_tokens = adapter.tokenize(SYSTEM_PROMPT)[0].tolist()

# Process multiple user queries
user_queries = [
    "What is machine learning?",
    "How does gradient descent work?",
    "Explain transformers in simple terms.",
    "What is the difference between AI and ML?",
]

for i, query in enumerate(user_queries):
    query_tokens = adapter.tokenize(query)[0].tolist()
    full_tokens = system_tokens + query_tokens

    result = manager.compute_incremental(full_tokens, adapter)

    print(f"Query {i+1}: '{query[:30]}...'")
    print(f"  Total tokens: {len(full_tokens)}")
    print(f"  Cached: {result.matched_length}")
    print(f"  Computed: {result.computed_length}")
    print(f"  Savings: {result.matched_length / len(full_tokens) * 100:.1f}%")
    print()
```

Output:
```
Query 1: 'What is machine learning?...'
  Total tokens: 45
  Cached: 0
  Computed: 45
  Savings: 0.0%

Query 2: 'How does gradient descent work...'
  Total tokens: 48
  Cached: 35
  Computed: 13
  Savings: 72.9%

Query 3: 'Explain transformers in simple...'
  Total tokens: 46
  Cached: 35
  Computed: 11
  Savings: 76.1%
...
```

---

## RAG Document Caching

For RAG (Retrieval-Augmented Generation) applications, the same documents are often queried multiple times. DeltaCache caches document KV states.

```python
from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import GPT2Adapter

adapter = GPT2Adapter.from_pretrained("gpt2", device="cuda")
manager = DeltaCacheManager(adapter.config)

# Sample documents
documents = {
    "python": """Python is a high-level programming language known for its
    simplicity and readability. It was created by Guido van Rossum and first
    released in 1991. Python supports multiple programming paradigms including
    procedural, object-oriented, and functional programming.""",

    "javascript": """JavaScript is a versatile programming language primarily
    used for web development. It was created by Brendan Eich in 1995. JavaScript
    enables interactive web pages and is supported by all modern web browsers.""",
}

# Pre-cache documents
doc_tokens = {}
for doc_id, doc_text in documents.items():
    tokens = adapter.tokenize(doc_text)[0].tolist()
    doc_tokens[doc_id] = tokens
    # Pre-compute and cache
    manager.compute_incremental(tokens, adapter)
    print(f"Cached document '{doc_id}': {len(tokens)} tokens")

# Query documents
queries = [
    ("python", "Who created Python?"),
    ("python", "When was Python released?"),
    ("javascript", "What is JavaScript used for?"),
    ("python", "What paradigms does Python support?"),
]

print("\nProcessing queries:")
for doc_id, question in queries:
    doc_toks = doc_tokens[doc_id]
    q_toks = adapter.tokenize(f"\nQuestion: {question}\nAnswer:")[0].tolist()
    full_tokens = doc_toks + q_toks

    result = manager.compute_incremental(full_tokens, adapter)

    print(f"  Q: {question}")
    print(f"    Document: {doc_id}, Cached: {result.matched_length}, Computed: {result.computed_length}")
```

---

## Few-Shot Learning

For few-shot classification tasks, the examples are shared across all inputs.

```python
from deltacache import DeltaCacheManager
from deltacache.hf_integration import GPT2Adapter

adapter = GPT2Adapter.from_pretrained("gpt2", device="cuda")
manager = DeltaCacheManager(adapter.config)

# Few-shot examples (expensive prefix, reused for all inputs)
FEW_SHOT_PROMPT = """Classify the sentiment of the text as positive, negative, or neutral.

Example 1:
Text: "I absolutely love this product! Best purchase ever!"
Sentiment: positive

Example 2:
Text: "Terrible quality. Complete waste of money."
Sentiment: negative

Example 3:
Text: "The product works as described."
Sentiment: neutral

Now classify:
"""

example_tokens = adapter.tokenize(FEW_SHOT_PROMPT)[0].tolist()

# Texts to classify
texts = [
    "This is the best phone I've ever owned!",
    "Never buying from this company again.",
    "Average product, nothing special.",
    "Exceeded my expectations in every way!",
    "Disappointed with the quality.",
]

print("Few-Shot Sentiment Classification")
print(f"Example prefix: {len(example_tokens)} tokens\n")

for text in texts:
    input_tokens = adapter.tokenize(f'Text: "{text}"\nSentiment:')[0].tolist()
    full_tokens = example_tokens + input_tokens

    result = manager.compute_incremental(full_tokens, adapter)

    reuse_rate = result.matched_length / len(full_tokens) * 100
    print(f"Text: \"{text[:40]}...\"")
    print(f"  Reuse rate: {reuse_rate:.1f}% ({result.matched_length}/{len(full_tokens)} tokens)")
```

---

## Multi-Turn Conversations

For multi-turn conversations, each turn builds on previous context.

```python
from deltacache import DeltaCacheManager
from deltacache.hf_integration import GPT2Adapter

adapter = GPT2Adapter.from_pretrained("gpt2", device="cuda")
manager = DeltaCacheManager(adapter.config)

# Simulate conversation
conversation = []
turns = [
    ("user", "Hello! I'd like to learn about Python."),
    ("assistant", "Hello! I'd be happy to help you learn Python. What would you like to know?"),
    ("user", "What are the basic data types?"),
    ("assistant", "Python has several basic data types: int, float, str, bool, list, dict, tuple, and set."),
    ("user", "Can you explain lists?"),
]

print("Multi-Turn Conversation Caching\n")

for role, content in turns:
    conversation.append(f"{role}: {content}")

    # Build full context
    context = "\n".join(conversation)
    tokens = adapter.tokenize(context)[0].tolist()

    result = manager.compute_incremental(tokens, adapter)

    print(f"Turn {len(conversation)}: {role}")
    print(f"  Context length: {len(tokens)} tokens")
    print(f"  Cached: {result.matched_length}, Computed: {result.computed_length}")
    print(f"  Reuse: {result.matched_length / len(tokens) * 100:.1f}%")
    print()
```

---

## vLLM Integration

DeltaCache can integrate with vLLM for production deployments.

```python
# Note: Requires vLLM installation
from vllm import LLM, SamplingParams
from deltacache.vllm_integration import patch_vllm_cache_engine, DeltaCacheEngine

# Patch vLLM to use DeltaCache
patch_vllm_cache_engine()

# Create vLLM model (now uses DeltaCache internally)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.8,
)

# System prompt shared across requests
system_prompt = "You are a helpful coding assistant. "

prompts = [
    system_prompt + "Write a Python function to reverse a string.",
    system_prompt + "Explain what a decorator is in Python.",
    system_prompt + "How do I read a JSON file in Python?",
]

# Generate - DeltaCache automatically caches shared prefixes
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(prompts, sampling_params)

# Check cache statistics
stats = DeltaCacheEngine.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1%}")
print(f"Token reuse rate: {stats['token_reuse_rate']:.1%}")
```

---

## Best Practices

1. **Order requests by shared prefix**: Process requests with the same prefix consecutively to maximize cache hits.

2. **Pre-cache common prefixes**: For frequently used system prompts or documents, pre-compute their KV states.

3. **Choose appropriate eviction policy**:
   - `tiered`: Best for memory-constrained environments
   - `adaptive`: Best for dynamic workloads
   - `composite`: Good default for mixed workloads

4. **Monitor cache statistics**: Use `manager.get_stats()` to track hit rates and adjust configuration.

5. **Set appropriate memory limits**: Configure `gpu_memory_limit` based on your available GPU memory.
