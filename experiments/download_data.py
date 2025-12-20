"""Download and prepare datasets for DeltaCache experiments."""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.parent / "data"


def download_sharegpt():
    """Download ShareGPT dataset from Hugging Face."""
    print("Downloading ShareGPT dataset...")

    try:
        # Try to load ShareGPT dataset
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")

        # Save to local file
        output_path = DATA_DIR / "sharegpt.json"

        conversations = []
        for item in tqdm(dataset, desc="Processing"):
            if "conversations" in item:
                conversations.append({
                    "id": item.get("id", ""),
                    "conversations": item["conversations"]
                })

        with open(output_path, "w") as f:
            json.dump(conversations, f)

        print(f"Saved {len(conversations)} conversations to {output_path}")
        return output_path

    except Exception as e:
        print(f"Error downloading ShareGPT: {e}")
        print("Trying alternative source...")

        # Try alternative
        try:
            dataset = load_dataset("lmsys/lmsys-chat-1m", split="train[:10000]")

            output_path = DATA_DIR / "lmsys_chat.json"

            conversations = []
            for item in tqdm(dataset, desc="Processing"):
                conversations.append({
                    "id": item.get("conversation_id", ""),
                    "conversation": item.get("conversation", [])
                })

            with open(output_path, "w") as f:
                json.dump(conversations, f)

            print(f"Saved {len(conversations)} conversations to {output_path}")
            return output_path

        except Exception as e2:
            print(f"Error with alternative: {e2}")
            return None


def download_longbench():
    """Download LongBench dataset."""
    print("\nDownloading LongBench dataset...")

    try:
        # LongBench has multiple subsets
        subsets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa"]

        all_data = {}
        for subset in subsets:
            try:
                print(f"  Loading {subset}...")
                dataset = load_dataset("THUDM/LongBench", subset, split="test")
                all_data[subset] = [dict(item) for item in dataset]
                print(f"    Loaded {len(all_data[subset])} samples")
            except Exception as e:
                print(f"    Failed to load {subset}: {e}")

        if all_data:
            output_path = DATA_DIR / "longbench.json"
            with open(output_path, "w") as f:
                json.dump(all_data, f)
            print(f"Saved LongBench data to {output_path}")
            return output_path

        return None

    except Exception as e:
        print(f"Error downloading LongBench: {e}")
        return None


def create_synthetic_rag_dataset():
    """Create synthetic RAG-style dataset for testing."""
    print("\nCreating synthetic RAG dataset...")

    # Simulate RAG scenario: shared document chunks + different queries
    documents = [
        "The transformer architecture was introduced in the paper 'Attention is All You Need' by Vaswani et al. in 2017. " * 10,
        "Large language models have revolutionized natural language processing. They can perform various tasks including translation, summarization, and question answering. " * 10,
        "Retrieval-augmented generation combines the power of large language models with external knowledge retrieval. " * 10,
        "KV cache optimization is crucial for efficient LLM inference. The cache stores key-value pairs computed during the forward pass. " * 10,
        "PagedAttention introduced by vLLM enables efficient memory management for LLM serving through paging. " * 10,
    ]

    queries = [
        "What is the transformer architecture?",
        "When was the transformer introduced?",
        "Who wrote the attention paper?",
        "What can LLMs do?",
        "How does RAG work?",
        "What is KV cache?",
        "How does PagedAttention work?",
        "What is vLLM?",
    ]

    # Create RAG-style requests: document + query
    rag_requests = []
    for doc_idx, doc in enumerate(documents):
        for query in queries:
            rag_requests.append({
                "document_id": doc_idx,
                "document": doc,
                "query": query,
                "full_prompt": f"Document: {doc}\n\nQuestion: {query}\n\nAnswer:"
            })

    output_path = DATA_DIR / "synthetic_rag.json"
    with open(output_path, "w") as f:
        json.dump(rag_requests, f)

    print(f"Created {len(rag_requests)} RAG requests to {output_path}")
    return output_path


def main():
    """Download all datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("DeltaCache Dataset Download")
    print("="*60)

    # Download datasets
    sharegpt_path = download_sharegpt()
    longbench_path = download_longbench()
    rag_path = create_synthetic_rag_dataset()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"ShareGPT:      {'✓' if sharegpt_path else '✗'}")
    print(f"LongBench:     {'✓' if longbench_path else '✗'}")
    print(f"Synthetic RAG: {'✓' if rag_path else '✗'}")


if __name__ == "__main__":
    main()
