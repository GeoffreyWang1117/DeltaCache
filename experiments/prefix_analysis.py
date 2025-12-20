"""Analyze prefix sharing patterns in conversation datasets."""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Use simple tokenizer for analysis
from transformers import AutoTokenizer


DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_ultrachat() -> List[List[Dict]]:
    """Load UltraChat conversations."""
    with open(DATA_DIR / "ultrachat.json") as f:
        data = json.load(f)

    conversations = []
    for item in data:
        if "messages" in item:
            conversations.append(item["messages"])

    return conversations


def tokenize_conversations(
    conversations: List[List[Dict]],
    tokenizer,
    max_conversations: int = 1000,
) -> List[List[int]]:
    """Tokenize conversations into token sequences."""
    tokenized = []

    for conv in tqdm(conversations[:max_conversations], desc="Tokenizing"):
        # Build full conversation text
        text = ""
        for msg in conv:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|{role}|>\n{content}\n"

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokenized.append(tokens)

    return tokenized


def analyze_prefix_patterns(token_sequences: List[List[int]]) -> Dict:
    """Analyze prefix sharing patterns."""
    results = {
        "total_sequences": len(token_sequences),
        "avg_length": 0,
        "min_length": 0,
        "max_length": 0,
        "prefix_sharing": {},
        "common_prefixes": [],
    }

    if not token_sequences:
        return results

    lengths = [len(seq) for seq in token_sequences]
    results["avg_length"] = np.mean(lengths)
    results["min_length"] = min(lengths)
    results["max_length"] = max(lengths)

    # Analyze prefix sharing between pairs
    print("\nAnalyzing prefix sharing patterns...")

    # Sample pairs for analysis (full pairwise is O(n^2))
    n_samples = min(len(token_sequences), 500)
    sampled = token_sequences[:n_samples]

    shared_lengths = []
    for i in tqdm(range(len(sampled)), desc="Analyzing pairs"):
        for j in range(i + 1, len(sampled)):
            shared = compute_shared_prefix_length(sampled[i], sampled[j])
            if shared > 0:
                shared_lengths.append(shared)

    if shared_lengths:
        results["prefix_sharing"] = {
            "pairs_with_shared_prefix": len(shared_lengths),
            "total_pairs": n_samples * (n_samples - 1) // 2,
            "sharing_rate": len(shared_lengths) / (n_samples * (n_samples - 1) // 2),
            "avg_shared_length": np.mean(shared_lengths),
            "max_shared_length": max(shared_lengths),
            "median_shared_length": np.median(shared_lengths),
        }

    # Find common prefixes (system prompts, etc.)
    prefix_counts = defaultdict(int)
    for seq in token_sequences:
        # Check various prefix lengths
        for prefix_len in [10, 20, 50, 100]:
            if len(seq) >= prefix_len:
                prefix = tuple(seq[:prefix_len])
                prefix_counts[prefix] += 1

    # Get most common prefixes
    sorted_prefixes = sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)
    results["common_prefixes"] = [
        {"length": len(p), "count": c, "share": c / len(token_sequences)}
        for p, c in sorted_prefixes[:20]
        if c > 1
    ]

    return results


def compute_shared_prefix_length(seq1: List[int], seq2: List[int]) -> int:
    """Compute length of shared prefix between two sequences."""
    shared = 0
    for t1, t2 in zip(seq1, seq2):
        if t1 == t2:
            shared += 1
        else:
            break
    return shared


def simulate_system_prompt_sharing(
    token_sequences: List[List[int]],
    system_prompt_tokens: List[int],
) -> Dict:
    """Simulate prefix sharing with common system prompt."""
    results = {
        "system_prompt_length": len(system_prompt_tokens),
        "sequences_with_prompt": 0,
        "total_tokens": 0,
        "reusable_tokens": 0,
    }

    # Prepend system prompt to all sequences
    augmented = []
    for seq in token_sequences:
        augmented.append(system_prompt_tokens + seq)

    results["sequences_with_prompt"] = len(augmented)

    # Calculate potential token reuse
    for seq in augmented:
        results["total_tokens"] += len(seq)

    # First request: compute all tokens
    # Subsequent requests: only compute non-prefix tokens
    first_seq_len = len(augmented[0]) if augmented else 0
    results["reusable_tokens"] = len(system_prompt_tokens) * (len(augmented) - 1)

    if results["total_tokens"] > 0:
        results["reuse_rate"] = results["reusable_tokens"] / results["total_tokens"]
    else:
        results["reuse_rate"] = 0

    return results


def analyze_rag_scenario(n_documents: int = 5, n_queries: int = 10, doc_length: int = 500):
    """Analyze prefix sharing in RAG scenario."""
    print("\n" + "="*60)
    print("RAG Scenario Analysis")
    print("="*60)

    # In RAG: same document chunk is used with multiple queries
    # Potential reuse: document tokens can be shared across queries

    total_requests = n_documents * n_queries
    tokens_per_request = doc_length + 50  # doc + query
    total_tokens_baseline = total_requests * tokens_per_request

    # With caching: each document computed once, queries added
    tokens_with_cache = n_documents * doc_length + total_requests * 50

    savings = (total_tokens_baseline - tokens_with_cache) / total_tokens_baseline

    print(f"\nConfiguration:")
    print(f"  Documents: {n_documents}")
    print(f"  Queries per document: {n_queries}")
    print(f"  Document length: {doc_length} tokens")
    print(f"  Query length: ~50 tokens")

    print(f"\nResults:")
    print(f"  Total requests: {total_requests}")
    print(f"  Baseline tokens: {total_tokens_baseline:,}")
    print(f"  With DeltaCache: {tokens_with_cache:,}")
    print(f"  Token savings: {savings*100:.1f}%")

    return {
        "n_documents": n_documents,
        "n_queries": n_queries,
        "total_requests": total_requests,
        "baseline_tokens": total_tokens_baseline,
        "cached_tokens": tokens_with_cache,
        "savings_rate": savings,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeltaCache Prefix Sharing Analysis")
    print("="*60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load and analyze UltraChat
    print("\nLoading UltraChat dataset...")
    conversations = load_ultrachat()
    print(f"Loaded {len(conversations)} conversations")

    # Tokenize
    token_sequences = tokenize_conversations(conversations, tokenizer, max_conversations=1000)

    # Analyze natural prefix patterns
    print("\n" + "="*60)
    print("Natural Prefix Sharing Analysis")
    print("="*60)

    results = analyze_prefix_patterns(token_sequences)

    print(f"\nDataset Statistics:")
    print(f"  Total sequences: {results['total_sequences']}")
    print(f"  Avg length: {results['avg_length']:.0f} tokens")
    print(f"  Min length: {results['min_length']} tokens")
    print(f"  Max length: {results['max_length']} tokens")

    if results["prefix_sharing"]:
        ps = results["prefix_sharing"]
        print(f"\nPrefix Sharing (sampled pairs):")
        print(f"  Pairs with shared prefix: {ps['pairs_with_shared_prefix']}")
        print(f"  Sharing rate: {ps['sharing_rate']*100:.2f}%")
        print(f"  Avg shared length: {ps['avg_shared_length']:.1f} tokens")
        print(f"  Max shared length: {ps['max_shared_length']} tokens")

    if results["common_prefixes"]:
        print(f"\nMost common prefixes:")
        for i, p in enumerate(results["common_prefixes"][:5]):
            print(f"  {i+1}. Length {p['length']}: {p['count']} occurrences ({p['share']*100:.1f}%)")

    # Simulate system prompt scenario
    print("\n" + "="*60)
    print("System Prompt Simulation")
    print("="*60)

    # Create a mock system prompt
    system_prompt = tokenizer.encode(
        "<|system|>\nYou are a helpful assistant. Answer questions accurately and concisely.\n",
        add_special_tokens=False
    )

    sim_results = simulate_system_prompt_sharing(token_sequences[:500], system_prompt)

    print(f"\nSystem prompt: {sim_results['system_prompt_length']} tokens")
    print(f"Sequences: {sim_results['sequences_with_prompt']}")
    print(f"Total tokens (baseline): {sim_results['total_tokens']:,}")
    print(f"Reusable tokens: {sim_results['reusable_tokens']:,}")
    print(f"Potential reuse rate: {sim_results['reuse_rate']*100:.1f}%")

    # RAG scenario
    rag_results = analyze_rag_scenario()

    # Save results
    all_results = {
        "dataset_stats": {
            "total_sequences": results["total_sequences"],
            "avg_length": results["avg_length"],
            "min_length": results["min_length"],
            "max_length": results["max_length"],
        },
        "prefix_sharing": results["prefix_sharing"],
        "system_prompt_simulation": sim_results,
        "rag_scenario": rag_results,
    }

    with open(RESULTS_DIR / "prefix_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to {RESULTS_DIR / 'prefix_analysis.json'}")


if __name__ == "__main__":
    main()
