"""ICML 2026 Targeted Benchmarks - Scenarios Where DeltaCache Excels.

This script tests scenarios specifically designed to showcase DeltaCache's strengths:
1. RAG with Fixed Knowledge Base - Same long document, many different queries
2. Few-shot Prompting - Fixed examples, varying queries
3. Chatbot with Long System Prompt - Fixed instructions, varying user inputs
4. Batch Inference - Same prompt template, different inputs

Key insight: DeltaCache benefits scale with:
- PREFIX LENGTH: Longer prefixes = more speedup
- PREFIX REUSE: Same prefix across queries = cache hits
- QUERY VARIABILITY: Only the suffix changes
"""

import os
import gc
import json
import time
import statistics
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltacache import DeltaCacheManager, DeltaCacheConfig
from deltacache.hf_integration import LlamaStyleAdapter

RESULTS_DIR = Path(__file__).parent / "results" / "paper"


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0


# =============================================================================
# Scenario 1: RAG with Fixed Knowledge Base
# =============================================================================

def create_rag_fixed_kb_dataset(n_queries_per_doc: int = 30) -> List[Dict]:
    """Create RAG dataset with FIXED knowledge base documents.

    Key: The SAME document is queried multiple times with DIFFERENT questions.
    This maximizes prefix reuse - the document prefix is cached once and reused.
    """
    # Long knowledge base documents (aim for 500-1000 tokens each)
    knowledge_base = {
        "python_guide": """
# Python Programming Complete Guide

## Chapter 1: Introduction to Python
Python is a high-level, interpreted programming language created by Guido van Rossum
and first released in 1991. Python's design philosophy emphasizes code readability
with the use of significant indentation. Its language constructs and object-oriented
approach aim to help programmers write clear, logical code for small and large-scale projects.

Python is dynamically typed and garbage-collected. It supports multiple programming
paradigms, including structured, object-oriented, and functional programming. Python
is often described as a "batteries included" language due to its comprehensive
standard library.

## Chapter 2: Data Types and Variables
Python has several built-in data types including integers, floats, strings, lists,
tuples, sets, and dictionaries. Variables in Python don't need explicit declaration
to reserve memory space. The declaration happens automatically when you assign a value.

Numbers in Python can be integers (int), floating-point (float), or complex numbers.
Strings are sequences of characters enclosed in quotes. Lists are ordered, mutable
collections. Tuples are ordered, immutable collections. Sets are unordered collections
of unique elements. Dictionaries are key-value pairs.

## Chapter 3: Control Flow
Python uses if-elif-else statements for conditional execution. For loops iterate
over sequences, while loops continue until a condition is false. Python also supports
break, continue, and pass statements for loop control.

## Chapter 4: Functions
Functions are defined using the 'def' keyword. Python supports default arguments,
keyword arguments, and variable-length arguments (*args, **kwargs). Lambda functions
provide a way to create small anonymous functions.

## Chapter 5: Object-Oriented Programming
Python supports object-oriented programming with classes and objects. Classes define
the structure and behavior of objects. Inheritance allows classes to inherit attributes
and methods from parent classes. Polymorphism enables objects to be treated as instances
of their parent class.

## Chapter 6: Modules and Packages
Modules are Python files containing definitions and statements. Packages are collections
of modules organized in directories. The import statement is used to access code from
other modules. Python's package manager pip is used to install third-party packages.

## Chapter 7: File Handling
Python provides built-in functions for file operations. The open() function creates
a file object. Files can be opened in read ('r'), write ('w'), append ('a'), or
binary ('b') modes. The with statement ensures proper file handling and automatic closure.

## Chapter 8: Error Handling
Python uses try-except blocks for error handling. Multiple except clauses can handle
different exception types. The finally clause executes regardless of exceptions.
Custom exceptions can be created by inheriting from the Exception class.
""",

        "ml_tutorial": """
# Machine Learning Comprehensive Tutorial

## Section 1: Fundamentals of Machine Learning
Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. The field evolved from
pattern recognition and computational learning theory in artificial intelligence.

Machine learning algorithms build mathematical models based on sample data, known as
training data, to make predictions or decisions without being explicitly programmed.
Machine learning is closely related to computational statistics and mathematical optimization.

## Section 2: Types of Machine Learning

### Supervised Learning
Supervised learning algorithms learn from labeled training data. The algorithm analyzes
the training data and produces an inferred function that can map new examples. Common
algorithms include linear regression, logistic regression, decision trees, random forests,
support vector machines, and neural networks.

### Unsupervised Learning
Unsupervised learning algorithms find patterns in unlabeled data. Clustering algorithms
group similar data points together. Dimensionality reduction techniques reduce the number
of features while preserving important information. Common algorithms include k-means
clustering, hierarchical clustering, and principal component analysis (PCA).

### Reinforcement Learning
Reinforcement learning involves an agent learning to make decisions by interacting with
an environment. The agent receives rewards or penalties based on its actions and learns
to maximize cumulative reward. Applications include game playing, robotics, and autonomous systems.

## Section 3: Deep Learning
Deep learning uses neural networks with multiple layers (hence "deep") to learn
hierarchical representations of data. Convolutional Neural Networks (CNNs) excel at
image analysis. Recurrent Neural Networks (RNNs) and Transformers handle sequential data.

The transformer architecture, introduced in 2017, has become the foundation for large
language models. These models demonstrate remarkable capabilities in understanding and
generating human language. Key concepts include attention mechanisms, self-attention,
and multi-head attention.

## Section 4: Model Training and Evaluation
Training involves optimizing model parameters to minimize a loss function. Common
optimizers include SGD, Adam, and RMSprop. Regularization techniques like dropout and
L2 regularization prevent overfitting.

Evaluation metrics depend on the task: accuracy, precision, recall, F1-score for
classification; MSE, MAE, R-squared for regression. Cross-validation provides robust
performance estimates.

## Section 5: Practical Considerations
Data preprocessing is crucial: handling missing values, feature scaling, encoding
categorical variables. Feature engineering can significantly improve model performance.
Hyperparameter tuning optimizes model configuration. Model deployment requires
consideration of latency, throughput, and resource constraints.
""",

        "api_documentation": """
# RESTful API Design and Implementation Guide

## Part 1: REST Architecture Principles
REST (Representational State Transfer) is an architectural style for designing networked
applications. RESTful APIs use HTTP requests to perform CRUD operations. Key principles
include statelessness, uniform interface, client-server separation, and layered system.

### HTTP Methods
- GET: Retrieve resources (should be idempotent and safe)
- POST: Create new resources
- PUT: Update/replace existing resources (idempotent)
- PATCH: Partial update of resources
- DELETE: Remove resources (idempotent)

### Status Codes
- 2xx Success: 200 OK, 201 Created, 204 No Content
- 3xx Redirection: 301 Moved Permanently, 304 Not Modified
- 4xx Client Error: 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found
- 5xx Server Error: 500 Internal Server Error, 503 Service Unavailable

## Part 2: API Design Best Practices

### URL Structure
Use nouns for resources: /users, /products, /orders
Use hierarchy for relationships: /users/{id}/orders
Use query parameters for filtering: /products?category=electronics&price_max=100

### Request/Response Format
JSON is the standard format. Include Content-Type headers. Use consistent naming
conventions (camelCase or snake_case). Implement pagination for large collections.

### Authentication and Authorization
Common methods: API keys, OAuth 2.0, JWT tokens. Use HTTPS for all communications.
Implement rate limiting to prevent abuse. Handle CORS for browser clients.

## Part 3: Error Handling
Return meaningful error messages with appropriate status codes. Include error codes
for programmatic handling. Provide documentation links for common errors. Log errors
for debugging while avoiding sensitive data exposure.

## Part 4: Versioning
Include version in URL (/v1/users) or headers (Accept: application/vnd.api+json;version=1).
Maintain backward compatibility when possible. Document deprecation timelines clearly.

## Part 5: Documentation
Use OpenAPI/Swagger for API specification. Provide interactive documentation with
examples. Include authentication instructions, rate limits, and error codes.
Keep documentation synchronized with implementation.

## Part 6: Performance Optimization
Implement caching with ETags and Cache-Control headers. Use compression (gzip) for
responses. Consider GraphQL for complex data requirements. Monitor API performance
and set up alerting.
"""
    }

    # Diverse queries for each document
    query_templates = {
        "python_guide": [
            "What is Python and who created it?",
            "Explain Python's data types.",
            "How does Python handle control flow?",
            "What are Python functions?",
            "Describe object-oriented programming in Python.",
            "How do modules and packages work?",
            "Explain file handling in Python.",
            "How does error handling work?",
            "What is Python's design philosophy?",
            "List Python's built-in data structures.",
            "What is the difference between lists and tuples?",
            "How are dictionaries used in Python?",
            "Explain the import statement.",
            "What does 'batteries included' mean?",
            "How do you define a class in Python?",
        ],
        "ml_tutorial": [
            "What is machine learning?",
            "Explain supervised learning.",
            "What is unsupervised learning?",
            "Describe reinforcement learning.",
            "What is deep learning?",
            "Explain the transformer architecture.",
            "What are CNNs used for?",
            "How is a model trained?",
            "What evaluation metrics exist?",
            "What is feature engineering?",
            "Explain regularization techniques.",
            "What is cross-validation?",
            "How do you handle missing values?",
            "What is hyperparameter tuning?",
            "Describe model deployment considerations.",
        ],
        "api_documentation": [
            "What is REST?",
            "Explain HTTP methods.",
            "What are HTTP status codes?",
            "How should URLs be structured?",
            "What format should responses use?",
            "Explain API authentication methods.",
            "How should errors be handled?",
            "What are API versioning strategies?",
            "How should APIs be documented?",
            "What performance optimizations exist?",
            "Explain rate limiting.",
            "What is OAuth 2.0?",
            "How does pagination work?",
            "What is CORS?",
            "Describe caching strategies.",
        ],
    }

    dataset = []
    for doc_name, doc_content in knowledge_base.items():
        queries = query_templates[doc_name]
        # Sample queries (with replacement to get more)
        selected_queries = random.choices(queries, k=n_queries_per_doc)
        dataset.append({
            "doc_name": doc_name,
            "document": doc_content,
            "queries": selected_queries,
        })

    return dataset


# =============================================================================
# Scenario 2: Few-shot Prompting
# =============================================================================

def create_fewshot_dataset(n_queries: int = 30) -> Dict:
    """Create few-shot prompting dataset.

    Key: FIXED examples (long prefix), varying test queries (short suffix).
    """
    # Fixed few-shot examples (this becomes the cached prefix)
    fewshot_examples = """You are a sentiment classifier. Classify the sentiment of the given text as 'positive', 'negative', or 'neutral'.

Example 1:
Text: "I absolutely loved this movie! The acting was superb and the plot kept me engaged throughout."
Sentiment: positive

Example 2:
Text: "The service at this restaurant was terrible. We waited an hour for cold food."
Sentiment: negative

Example 3:
Text: "The weather today is partly cloudy with temperatures around 70 degrees."
Sentiment: neutral

Example 4:
Text: "This is the best purchase I've ever made. Highly recommend to everyone!"
Sentiment: positive

Example 5:
Text: "I'm disappointed with the quality. It broke after just two days of use."
Sentiment: negative

Example 6:
Text: "The meeting has been rescheduled to 3 PM tomorrow."
Sentiment: neutral

Example 7:
Text: "What an amazing concert! The band exceeded all my expectations."
Sentiment: positive

Example 8:
Text: "The customer support was unhelpful and rude. Very frustrating experience."
Sentiment: negative

Now classify the following:
"""

    # Test queries (these are the varying suffixes)
    test_queries = [
        "The product arrived on time and works as expected.",
        "I can't believe how bad this experience was.",
        "The store opens at 9 AM on weekdays.",
        "This exceeded my expectations in every way!",
        "Worst purchase ever. Complete waste of money.",
        "The package weighs approximately 2.5 pounds.",
        "So happy with my decision to buy this!",
        "Never shopping here again. Terrible quality.",
        "The event will be held in the main auditorium.",
        "Absolutely fantastic! Five stars!",
        "Broken on arrival. Very disappointed.",
        "The report contains quarterly financial data.",
        "Best customer service I've experienced!",
        "Took forever to arrive and was damaged.",
        "The office is located on the third floor.",
        "I'm thrilled with the results!",
        "Poor quality, poor service, poor everything.",
        "The meeting minutes have been distributed.",
        "Exceeded all expectations! Amazing!",
        "Regret buying this. Total disappointment.",
        "The schedule has been updated.",
        "Love it! Will definitely buy again!",
        "Awful experience from start to finish.",
        "The document requires your signature.",
        "Incredible value for the price!",
        "Frustrated with the lack of support.",
        "The training session starts at 10 AM.",
        "Simply the best! Highly recommended!",
        "Very unhappy with this purchase.",
        "The deadline is next Friday.",
    ]

    return {
        "prefix": fewshot_examples,
        "queries": random.sample(test_queries, min(n_queries, len(test_queries))),
    }


# =============================================================================
# Scenario 3: Long System Prompt Chatbot
# =============================================================================

def create_system_prompt_dataset(n_queries: int = 30) -> Dict:
    """Create chatbot dataset with long system prompt.

    Key: FIXED long system prompt, varying user queries.
    """
    # Long detailed system prompt (cached prefix)
    system_prompt = """You are an advanced AI assistant created by Anthropic. Your role is to help users with a wide range of tasks while following these comprehensive guidelines:

## Core Principles
1. Always be helpful, harmless, and honest
2. Provide accurate information based on your training data
3. Acknowledge uncertainty when you're not sure about something
4. Respect user privacy and don't ask for unnecessary personal information
5. Decline requests that could cause harm

## Communication Style
- Use clear, concise language appropriate to the user's level of expertise
- Break down complex topics into understandable parts
- Provide examples when helpful
- Ask clarifying questions when the request is ambiguous
- Use proper formatting (lists, code blocks, etc.) to improve readability

## Task Handling
- For coding questions: Provide working code with explanations
- For factual questions: Give accurate information with context
- For creative tasks: Offer multiple options when appropriate
- For analysis tasks: Consider multiple perspectives
- For technical support: Provide step-by-step instructions

## Limitations
- Training data has a knowledge cutoff date
- Cannot access the internet or external databases
- Cannot execute code or access files
- Cannot make purchases or bookings
- Cannot access real-time information

## Response Format
- Start with a direct answer to the question
- Provide supporting details and context
- Include relevant examples or code snippets
- End with next steps or additional resources if applicable

## Safety Guidelines
- Do not generate harmful, illegal, or unethical content
- Do not provide medical, legal, or financial advice as a substitute for professional consultation
- Do not assist with academic dishonesty
- Do not generate content that violates intellectual property rights
- Report concerning requests appropriately

## Interaction Quality
- Maintain a professional yet friendly tone
- Show empathy when users express frustration
- Celebrate user successes and progress
- Offer encouragement for learning efforts
- Be patient with repeated or basic questions

Remember: Your goal is to be maximally helpful while ensuring safety and accuracy in all interactions.

User: """

    # Varying user queries
    user_queries = [
        "How do I sort a list in Python?",
        "What's the capital of Australia?",
        "Explain quantum computing briefly.",
        "Write a haiku about programming.",
        "What causes rain?",
        "How do I make pasta?",
        "Explain machine learning to a child.",
        "What's the difference between HTTP and HTTPS?",
        "How do I calculate compound interest?",
        "What are the planets in our solar system?",
        "How do I center a div in CSS?",
        "What is photosynthesis?",
        "Recommend a good book to read.",
        "How do I make my code faster?",
        "What's the best way to learn a new language?",
        "Explain blockchain simply.",
        "How do I handle errors in JavaScript?",
        "What causes earthquakes?",
        "How do I write a good resume?",
        "What is the meaning of life?",
        "How do I install Python packages?",
        "Explain relativity theory.",
        "What's the best programming language?",
        "How do I improve my memory?",
        "What is dark matter?",
        "How do databases work?",
        "What causes climate change?",
        "How do I start exercising?",
        "Explain neural networks.",
        "What makes a good leader?",
    ]

    return {
        "system_prompt": system_prompt,
        "queries": random.sample(user_queries, min(n_queries, len(user_queries))),
    }


# =============================================================================
# Scenario 4: Batch Code Completion
# =============================================================================

def create_code_completion_dataset(n_queries: int = 30) -> Dict:
    """Create code completion dataset with fixed context.

    Key: FIXED code context (imports, class definitions), varying completion requests.
    """
    # Fixed code context (long prefix)
    code_context = '''"""
Data Processing Library - Complete Implementation
"""

import os
import json
import csv
import logging
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import pickle


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataRecord:
    """Represents a single data record with metadata."""
    id: str
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'tags': self.tags,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataRecord':
        return cls(
            id=d['id'],
            data=d['data'],
            created_at=datetime.fromisoformat(d['created_at']),
            updated_at=datetime.fromisoformat(d['updated_at']) if d.get('updated_at') else None,
            tags=d.get('tags', []),
            metadata=d.get('metadata', {}),
        )


class DataProcessor(ABC):
    """Abstract base class for data processors."""

    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        pass


class DataStore:
    """Main data storage and retrieval class."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data_store")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, DataRecord] = {}
        self._index: Dict[str, List[str]] = {}
        logger.info(f"DataStore initialized at {self.storage_path}")

    def add_record(self, record: DataRecord) -> bool:
        """Add a new record to the store."""
        if record.id in self._cache:
            logger.warning(f"Record {record.id} already exists")
            return False
        self._cache[record.id] = record
        self._update_index(record)
        logger.info(f"Added record {record.id}")
        return True

    def get_record(self, record_id: str) -> Optional[DataRecord]:
        """Retrieve a record by ID."""
        return self._cache.get(record_id)

    def update_record(self, record_id: str, data: Dict[str, Any]) -> bool:
        """Update an existing record."""
        if record_id not in self._cache:
            logger.error(f"Record {record_id} not found")
            return False
        record = self._cache[record_id]
        record.data.update(data)
        record.updated_at = datetime.now()
        return True

    def delete_record(self, record_id: str) -> bool:
        """Delete a record from the store."""
        if record_id not in self._cache:
            return False
        del self._cache[record_id]
        self._rebuild_index()
        return True

    def search_by_tag(self, tag: str) -> List[DataRecord]:
        """Search records by tag."""
        record_ids = self._index.get(tag, [])
        return [self._cache[rid] for rid in record_ids if rid in self._cache]

    def _update_index(self, record: DataRecord) -> None:
        """Update the tag index for a record."""
        for tag in record.tags:
            if tag not in self._index:
                self._index[tag] = []
            if record.id not in self._index[tag]:
                self._index[tag].append(record.id)

    def _rebuild_index(self) -> None:
        """Rebuild the entire tag index."""
        self._index.clear()
        for record in self._cache.values():
            self._update_index(record)

    def save_to_disk(self) -> None:
        """Persist all records to disk."""
        data_file = self.storage_path / "records.json"
        records_dict = {rid: rec.to_dict() for rid, rec in self._cache.items()}
        with open(data_file, 'w') as f:
            json.dump(records_dict, f, indent=2)
        logger.info(f"Saved {len(self._cache)} records to disk")

    def load_from_disk(self) -> None:
        """Load records from disk."""
        data_file = self.storage_path / "records.json"
        if not data_file.exists():
            return
        with open(data_file, 'r') as f:
            records_dict = json.load(f)
        self._cache = {rid: DataRecord.from_dict(rec) for rid, rec in records_dict.items()}
        self._rebuild_index()
        logger.info(f"Loaded {len(self._cache)} records from disk")


# Now implement the following function:
'''

    # Completion requests (varying suffixes)
    completion_requests = [
        "def filter_records_by_date(self, start_date: datetime, end_date: datetime) -> List[DataRecord]:",
        "def export_to_csv(self, filepath: Path) -> None:",
        "def import_from_csv(self, filepath: Path) -> int:",
        "def get_statistics(self) -> Dict[str, Any]:",
        "def bulk_add_records(self, records: List[DataRecord]) -> int:",
        "def merge_records(self, record_ids: List[str], new_id: str) -> Optional[DataRecord]:",
        "def clone_record(self, record_id: str, new_id: str) -> Optional[DataRecord]:",
        "def add_tag_to_records(self, record_ids: List[str], tag: str) -> int:",
        "def remove_tag_from_records(self, record_ids: List[str], tag: str) -> int:",
        "def search_by_data_field(self, field: str, value: Any) -> List[DataRecord]:",
        "def get_records_count(self) -> int:",
        "def clear_all_records(self) -> None:",
        "def backup_store(self, backup_path: Path) -> bool:",
        "def restore_from_backup(self, backup_path: Path) -> bool:",
        "def validate_all_records(self, validator: Callable) -> List[str]:",
        "def get_recent_records(self, limit: int = 10) -> List[DataRecord]:",
        "def get_records_by_metadata(self, key: str, value: Any) -> List[DataRecord]:",
        "def update_metadata(self, record_id: str, metadata: Dict) -> bool:",
        "def get_all_tags(self) -> List[str]:",
        "def get_tag_counts(self) -> Dict[str, int]:",
        "def compress_store(self) -> None:",
        "def decompress_store(self) -> None:",
        "def compute_checksum(self) -> str:",
        "def verify_integrity(self) -> bool:",
        "def get_storage_size(self) -> int:",
        "def optimize_storage(self) -> None:",
        "def create_snapshot(self) -> str:",
        "def restore_snapshot(self, snapshot_id: str) -> bool:",
        "def get_record_history(self, record_id: str) -> List[Dict]:",
        "def rollback_record(self, record_id: str, version: int) -> bool:",
    ]

    return {
        "context": code_context,
        "completions": random.sample(completion_requests, min(n_queries, len(completion_requests))),
    }


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    adapter,
    tokenizer,
    config,
    prefix: str,
    queries: List[str],
    scenario_name: str,
    n_runs: int = 3,
) -> Dict:
    """Run benchmark for a given scenario."""
    print(f"\n{'='*70}")
    print(f"Scenario: {scenario_name}")
    print(f"Prefix tokens: {len(tokenizer.encode(prefix))}")
    print(f"Number of queries: {len(queries)}")
    print(f"{'='*70}")

    all_dc_latencies = []
    all_hf_latencies = []
    all_dc_first_latencies = []
    all_hf_first_latencies = []
    total_matched = 0
    total_tokens = 0

    for run in range(n_runs):
        print(f"\n  Run {run + 1}/{n_runs}")

        # DeltaCache benchmark
        manager = DeltaCacheManager(config)
        dc_latencies = []
        matched_tokens = 0
        run_total_tokens = 0

        for i, query in enumerate(tqdm(queries, desc="  DeltaCache")):
            prompt = f"{prefix}\n{query}"
            tokens = tokenizer.encode(prompt)
            run_total_tokens += len(tokens)

            torch.cuda.synchronize()
            start = time.perf_counter()
            result = manager.compute_incremental(tokens, adapter.compute_kv)
            torch.cuda.synchronize()

            latency = (time.perf_counter() - start) * 1000
            dc_latencies.append(latency)
            matched_tokens += result.matched_length

            if i == 0:
                all_dc_first_latencies.append(latency)

        total_matched += matched_tokens
        total_tokens += run_total_tokens
        all_dc_latencies.extend(dc_latencies)

        del manager
        clear_gpu()

        # HuggingFace baseline (no cache)
        hf_latencies = []

        for i, query in enumerate(tqdm(queries, desc="  Baseline")):
            prompt = f"{prefix}\n{query}"
            tokens = tokenizer.encode(prompt)

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = adapter.compute_kv_for_tokens(tokens)
            torch.cuda.synchronize()

            latency = (time.perf_counter() - start) * 1000
            hf_latencies.append(latency)

            if i == 0:
                all_hf_first_latencies.append(latency)

        all_hf_latencies.extend(hf_latencies)
        clear_gpu()

    # Compute metrics
    dc_mean = statistics.mean(all_dc_latencies)
    hf_mean = statistics.mean(all_hf_latencies)

    # Skip first query for cached speedup (first query has no cache)
    dc_cached = all_dc_latencies[1::len(queries)]  # Every query except first in each run
    hf_cached = all_hf_latencies[1::len(queries)]

    results = {
        "scenario": scenario_name,
        "prefix_tokens": len(tokenizer.encode(prefix)),
        "n_queries": len(queries),
        "n_runs": n_runs,
        "latency": {
            "deltacache_mean_ms": dc_mean,
            "deltacache_std_ms": statistics.stdev(all_dc_latencies) if len(all_dc_latencies) > 1 else 0,
            "baseline_mean_ms": hf_mean,
            "baseline_std_ms": statistics.stdev(all_hf_latencies) if len(all_hf_latencies) > 1 else 0,
        },
        "speedup": {
            "overall": hf_mean / dc_mean if dc_mean > 0 else 0,
            "cached_queries": statistics.mean(hf_cached) / statistics.mean(dc_cached) if dc_cached else 0,
        },
        "token_reuse_rate": total_matched / total_tokens if total_tokens > 0 else 0,
        "ttft": {
            "deltacache_first_ms": statistics.mean(all_dc_first_latencies),
            "baseline_first_ms": statistics.mean(all_hf_first_latencies),
        },
    }

    print(f"\n  Results:")
    print(f"    Overall Speedup: {results['speedup']['overall']:.2f}x")
    print(f"    Cached Speedup: {results['speedup']['cached_queries']:.2f}x")
    print(f"    Token Reuse: {results['token_reuse_rate']*100:.1f}%")

    return results


def run_all_targeted_benchmarks(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cuda",
    n_queries: int = 30,
    n_runs: int = 3,
) -> Dict:
    """Run all targeted benchmark scenarios."""
    print("\n" + "#" * 70)
    print("# ICML 2026 TARGETED BENCHMARKS")
    print(f"# Model: {model_name}")
    print(f"# Timestamp: {datetime.now().isoformat()}")
    print("#" * 70)

    # Load model
    print("\nLoading model...")
    adapter = LlamaStyleAdapter.from_pretrained(model_name, device=device)
    tokenizer = adapter.tokenizer
    config = DeltaCacheConfig.for_model(model_name)
    config.device = device
    print(f"Model loaded. GPU memory: {get_gpu_mem():.2f} GB")

    results = {
        "metadata": {
            "model": model_name,
            "device": device,
            "n_queries_per_scenario": n_queries,
            "n_runs": n_runs,
            "timestamp": datetime.now().isoformat(),
        },
        "scenarios": {}
    }

    try:
        # Scenario 1: RAG with Fixed Knowledge Base
        rag_data = create_rag_fixed_kb_dataset(n_queries_per_doc=n_queries)
        for doc_data in rag_data:
            scenario_name = f"rag_{doc_data['doc_name']}"
            results["scenarios"][scenario_name] = run_benchmark(
                adapter, tokenizer, config,
                prefix=f"Knowledge Base:\n{doc_data['document']}\n\nQuestion: ",
                queries=[f"{q}\nAnswer:" for q in doc_data['queries']],
                scenario_name=scenario_name,
                n_runs=n_runs,
            )

        # Scenario 2: Few-shot Prompting
        fewshot_data = create_fewshot_dataset(n_queries=n_queries)
        results["scenarios"]["fewshot_sentiment"] = run_benchmark(
            adapter, tokenizer, config,
            prefix=fewshot_data["prefix"],
            queries=[f"Text: \"{q}\"\nSentiment:" for q in fewshot_data["queries"]],
            scenario_name="fewshot_sentiment",
            n_runs=n_runs,
        )

        # Scenario 3: Long System Prompt
        system_data = create_system_prompt_dataset(n_queries=n_queries)
        results["scenarios"]["long_system_prompt"] = run_benchmark(
            adapter, tokenizer, config,
            prefix=system_data["system_prompt"],
            queries=system_data["queries"],
            scenario_name="long_system_prompt",
            n_runs=n_runs,
        )

        # Scenario 4: Code Completion
        code_data = create_code_completion_dataset(n_queries=n_queries)
        results["scenarios"]["code_completion"] = run_benchmark(
            adapter, tokenizer, config,
            prefix=code_data["context"],
            queries=code_data["completions"],
            scenario_name="code_completion",
            n_runs=n_runs,
        )

    finally:
        del adapter
        clear_gpu()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, data in results["scenarios"].items():
        print(f"\n{name}:")
        print(f"  Prefix: {data['prefix_tokens']} tokens")
        print(f"  Speedup: {data['speedup']['overall']:.2f}x (cached: {data['speedup']['cached_queries']:.2f}x)")
        print(f"  Token Reuse: {data['token_reuse_rate']*100:.1f}%")

    # Overall stats
    all_speedups = [s["speedup"]["overall"] for s in results["scenarios"].values()]
    all_cached = [s["speedup"]["cached_queries"] for s in results["scenarios"].values()]

    print(f"\n{'='*70}")
    print(f"OVERALL: {statistics.mean(all_speedups):.2f}x mean speedup")
    print(f"         {statistics.mean(all_cached):.2f}x mean cached speedup")
    print(f"         {max(all_speedups):.2f}x max speedup")

    results["summary"] = {
        "mean_speedup": statistics.mean(all_speedups),
        "mean_cached_speedup": statistics.mean(all_cached),
        "max_speedup": max(all_speedups),
        "min_speedup": min(all_speedups),
    }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ICML 2026 Targeted Benchmarks")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    results = run_all_targeted_benchmarks(
        model_name=args.model,
        device=args.device,
        n_queries=args.n_queries,
        n_runs=args.n_runs,
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_path = Path(args.output)
    else:
        model_short = args.model.split("/")[-1].lower().replace("-", "_")
        output_path = RESULTS_DIR / f"targeted_benchmark_{model_short}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
