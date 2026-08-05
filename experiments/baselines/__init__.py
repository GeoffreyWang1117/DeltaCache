"""KV cache compression baselines for controlled comparison.

Import this module to populate the REGISTRY with all available baselines.
"""

from .base import REGISTRY, BaselineMethod, CompressResult, h2o_token_selection

# Import all baselines to trigger @register_baseline decorators
from .h2o import H2OUniform
from .kivi import KIVIUniform
from .snapkv import SnapKV
from .pyramidkv import PyramidKV
from .d2o import D2O
from .squeeze_attention import SqueezeAttention
from .adakv import AdaKV
from .dynamickv import DynamicKV
from .xquant import XQuant
from .evolkv import EvolKV
from .lava import LAVa
from .cake_wrapper import CAKE
from .kvtuner_wrapper import KVTuner
from .minikv import MiniKV
from .streaming_llm import StreamingLLM
from .duo_attention import DuoAttention

# Legacy baselines (original implementations)
from .cake import CAKEBaseline
from .kvtuner import KVTunerBaseline

__all__ = [
    "REGISTRY",
    "BaselineMethod",
    "CompressResult",
    "h2o_token_selection",
    # Registered baselines
    "H2OUniform",
    "KIVIUniform",
    "SnapKV",
    "PyramidKV",
    "D2O",
    "SqueezeAttention",
    "AdaKV",
    "DynamicKV",
    "XQuant",
    "EvolKV",
    "LAVa",
    "MiniKV",
    "StreamingLLM",
    "DuoAttention",
    # Legacy
    "CAKEBaseline",
    "KVTunerBaseline",
]
