"""Benchmark scenarios for DeltaCache evaluation."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BenchmarkScenario:
    """Definition of a benchmark scenario.

    Attributes:
        name: Human-readable scenario name
        description: Detailed description of the scenario
        prompts: List of prompts to process
        expected_prefix_sharing: Expected fraction of tokens that can be reused
        system_prompt: Optional shared system prompt
        num_iterations: Number of times to repeat for timing
    """

    name: str
    description: str
    prompts: List[str]
    expected_prefix_sharing: float
    system_prompt: Optional[str] = None
    num_iterations: int = 1


# Pre-defined benchmark scenarios
SCENARIOS = {
    "system_prompt_short": BenchmarkScenario(
        name="System Prompt (Short)",
        description="Multiple requests with a short shared system prompt",
        system_prompt="You are a helpful assistant.",
        prompts=[
            "What is Python?",
            "Explain machine learning.",
            "How does the internet work?",
            "What is quantum computing?",
            "Describe blockchain technology.",
            "What is artificial intelligence?",
            "Explain neural networks.",
            "What is cloud computing?",
            "How do databases work?",
            "What is cybersecurity?",
        ],
        expected_prefix_sharing=0.3,
    ),
    "system_prompt_long": BenchmarkScenario(
        name="System Prompt (Long)",
        description="Multiple requests with a long shared system prompt",
        system_prompt="""You are an expert AI assistant with deep knowledge in multiple domains including science, technology, mathematics, history, and the arts. Your responses should be accurate, well-structured, and educational. Always provide context and examples when explaining concepts. If you're unsure about something, acknowledge the uncertainty. Aim to be helpful while maintaining intellectual honesty.

When answering questions:
1. Start with a clear, concise summary
2. Provide detailed explanations with examples
3. Mention related concepts when relevant
4. Cite sources or suggest further reading when appropriate

Remember to be respectful and patient with all users, regardless of their level of expertise.""",
        prompts=[
            "What is Python?",
            "Explain machine learning.",
            "How does the internet work?",
            "What is quantum computing?",
            "Describe blockchain technology.",
            "What is artificial intelligence?",
            "Explain neural networks.",
            "What is cloud computing?",
            "How do databases work?",
            "What is cybersecurity?",
        ],
        expected_prefix_sharing=0.6,
    ),
    "multi_turn_conversation": BenchmarkScenario(
        name="Multi-turn Conversation",
        description="Simulated multi-turn conversation with growing context",
        prompts=[
            "User: Hi there!\nAssistant: Hello! How can I help you today?",
            "User: Hi there!\nAssistant: Hello! How can I help you today?\nUser: I have a question about Python.",
            "User: Hi there!\nAssistant: Hello! How can I help you today?\nUser: I have a question about Python.\nAssistant: Of course! I'd be happy to help with Python. What would you like to know?",
            "User: Hi there!\nAssistant: Hello! How can I help you today?\nUser: I have a question about Python.\nAssistant: Of course! I'd be happy to help with Python. What would you like to know?\nUser: How do I read a file?",
            "User: Hi there!\nAssistant: Hello! How can I help you today?\nUser: I have a question about Python.\nAssistant: Of course! I'd be happy to help with Python. What would you like to know?\nUser: How do I read a file?\nAssistant: You can use the open() function with the 'r' mode.\nUser: Can you show me an example?",
        ],
        expected_prefix_sharing=0.7,
    ),
    "document_qa": BenchmarkScenario(
        name="Document QA",
        description="Same document with different questions (RAG scenario)",
        system_prompt=None,
        prompts=[
            # All start with the same document context
            """Document: Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

Question: Who created Python?""",
            """Document: Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

Question: When was Python first released?""",
            """Document: Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

Question: What programming paradigms does Python support?""",
            """Document: Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

Question: Is Python statically or dynamically typed?""",
            """Document: Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.

Question: What is Python known for?""",
        ],
        expected_prefix_sharing=0.85,
    ),
    "few_shot_classification": BenchmarkScenario(
        name="Few-Shot Classification",
        description="Sentiment classification with shared few-shot examples",
        system_prompt=None,
        prompts=[
            # All share the same few-shot prefix
            """Classify the sentiment of the following text as positive, negative, or neutral.

Examples:
Text: "I love this product! It works perfectly."
Sentiment: positive

Text: "This is the worst purchase I've ever made."
Sentiment: negative

Text: "The item arrived on time."
Sentiment: neutral

Text: "This phone is amazing, best I've ever owned!"
Sentiment:""",
            """Classify the sentiment of the following text as positive, negative, or neutral.

Examples:
Text: "I love this product! It works perfectly."
Sentiment: positive

Text: "This is the worst purchase I've ever made."
Sentiment: negative

Text: "The item arrived on time."
Sentiment: neutral

Text: "Terrible customer service, never buying again."
Sentiment:""",
            """Classify the sentiment of the following text as positive, negative, or neutral.

Examples:
Text: "I love this product! It works perfectly."
Sentiment: positive

Text: "This is the worst purchase I've ever made."
Sentiment: negative

Text: "The item arrived on time."
Sentiment: neutral

Text: "The product is okay, nothing special."
Sentiment:""",
            """Classify the sentiment of the following text as positive, negative, or neutral.

Examples:
Text: "I love this product! It works perfectly."
Sentiment: positive

Text: "This is the worst purchase I've ever made."
Sentiment: negative

Text: "The item arrived on time."
Sentiment: neutral

Text: "Absolutely fantastic experience, highly recommend!"
Sentiment:""",
            """Classify the sentiment of the following text as positive, negative, or neutral.

Examples:
Text: "I love this product! It works perfectly."
Sentiment: positive

Text: "This is the worst purchase I've ever made."
Sentiment: negative

Text: "The item arrived on time."
Sentiment: neutral

Text: "Waste of money, broke after one day."
Sentiment:""",
        ],
        expected_prefix_sharing=0.9,
    ),
    "no_sharing_baseline": BenchmarkScenario(
        name="No Sharing Baseline",
        description="Completely different prompts with no prefix sharing",
        prompts=[
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming industries worldwide.",
            "The history of Rome spans over a thousand years.",
            "Quantum mechanics describes nature at the atomic scale.",
            "Climate change poses significant challenges to humanity.",
            "The Renaissance was a period of cultural rebirth.",
            "Artificial neural networks are inspired by the brain.",
            "The ocean covers over 70% of Earth's surface.",
            "Democracy originated in ancient Greece.",
            "Modern medicine has greatly extended human lifespan.",
        ],
        expected_prefix_sharing=0.0,
    ),
}


def get_scenario(name: str) -> BenchmarkScenario:
    """Get a benchmark scenario by name.

    Args:
        name: Scenario name (key in SCENARIOS dict)

    Returns:
        BenchmarkScenario instance

    Raises:
        KeyError: If scenario not found
    """
    if name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())
        raise KeyError(f"Unknown scenario: {name}. Available: {available}")
    return SCENARIOS[name]


def list_scenarios() -> List[str]:
    """List all available scenario names."""
    return list(SCENARIOS.keys())
