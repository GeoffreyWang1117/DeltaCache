"""Evaluation task modules."""

from .ppl import PPLTask
from .mmlu import MMLUTask
from .longbench import LongBenchTask
from .gsm8k import GSM8KTask
from .ruler import RULERTask
from .niah import NIAHTask
from .throughput import ThroughputTask
from .math import MATHTask

TASK_REGISTRY = {
    "ppl": PPLTask,
    "mmlu": MMLUTask,
    "longbench": LongBenchTask,
    "gsm8k": GSM8KTask,
    "ruler": RULERTask,
    "niah": NIAHTask,
    "throughput": ThroughputTask,
    "math": MATHTask,
}
