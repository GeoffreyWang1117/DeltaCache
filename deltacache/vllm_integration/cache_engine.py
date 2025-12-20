"""vLLM cache engine integration for DeltaCache."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from deltacache.api import DeltaCacheManager
from deltacache.core.cache_block import CacheBlock
from deltacache.utils.config import DeltaCacheConfig

if TYPE_CHECKING:
    try:
        from vllm.config import CacheConfig, ModelConfig, ParallelConfig
    except ImportError:
        CacheConfig = None
        ModelConfig = None
        ParallelConfig = None


class DeltaCacheEngine:
    """
    DeltaCache integration as vLLM cache engine replacement.

    This class wraps DeltaCacheManager to provide vLLM-compatible interface.
    It can be used as a drop-in replacement for vLLM's CacheEngine.

    Usage with vLLM:
        ```python
        from vllm import LLM
        from deltacache.vllm_integration import DeltaCacheEngine

        # Method 1: Monkey-patch
        import vllm.worker.cache_engine
        vllm.worker.cache_engine.CacheEngine = DeltaCacheEngine

        llm = LLM(model="meta-llama/Llama-2-7b-hf")

        # Method 2: Custom worker (recommended for production)
        # See vLLM documentation for custom worker setup
        ```
    """

    def __init__(
        self,
        cache_config: "CacheConfig",
        model_config: "ModelConfig",
        parallel_config: "ParallelConfig",
        device_config: Optional[object] = None,
    ) -> None:
        """
        Initialize DeltaCacheEngine with vLLM configs.

        Args:
            cache_config: vLLM cache configuration.
            model_config: vLLM model configuration.
            parallel_config: vLLM parallel configuration.
            device_config: vLLM device configuration (optional).
        """
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        # Extract model parameters
        hf_config = model_config.hf_config
        num_layers = hf_config.num_hidden_layers
        num_heads = hf_config.num_attention_heads

        # Handle GQA (grouped query attention)
        num_kv_heads = getattr(hf_config, "num_key_value_heads", num_heads)
        head_dim = hf_config.hidden_size // num_heads

        # Create DeltaCache config
        delta_config = DeltaCacheConfig(
            gpu_memory_limit=int(cache_config.gpu_memory_utilization * self._get_gpu_memory()),
            cpu_memory_limit=cache_config.swap_space_bytes if hasattr(cache_config, 'swap_space_bytes') else 0,
            num_layers=num_layers,
            num_heads=num_kv_heads,  # Use KV heads for cache
            head_dim=head_dim,
            block_size=cache_config.block_size,
            dtype=str(model_config.dtype).split(".")[-1],
            device="cuda",
        )

        # Initialize DeltaCacheManager
        self.manager = DeltaCacheManager(delta_config)

        # Track block allocations for vLLM compatibility
        self._gpu_cache: Dict[int, List[Tensor]] = {}
        self._cpu_cache: Dict[int, List[Tensor]] = {}

        # Sequence to block mapping
        self._seq_block_map: Dict[int, List[int]] = {}

        # Block size for allocation
        self.block_size = cache_config.block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = model_config.dtype

    def _get_gpu_memory(self) -> int:
        """Get total GPU memory."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory
        return 0

    def get_num_free_gpu_blocks(self) -> int:
        """Get number of free GPU blocks."""
        stats = self.manager.get_memory_stats()
        free_bytes = stats.free_bytes
        block_size_bytes = self._block_size_bytes()
        return free_bytes // block_size_bytes

    def get_num_free_cpu_blocks(self) -> int:
        """Get number of free CPU blocks."""
        # For simplicity, report large number if CPU caching enabled
        if self.manager.config.cpu_memory_limit > 0:
            return self.manager.config.cpu_memory_limit // self._block_size_bytes()
        return 0

    def _block_size_bytes(self) -> int:
        """Calculate memory per block in bytes."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        # key + value per layer per block
        return 2 * self.num_layers * self.block_size * self.num_kv_heads * self.head_dim * element_size

    def allocate_gpu_cache(self) -> List[Tensor]:
        """
        Allocate GPU cache tensors.

        Returns list of KV cache tensors per layer.
        This is called by vLLM during initialization.
        """
        # vLLM expects list of [2, num_blocks, block_size, num_heads, head_dim]
        # We allocate dynamically, so return empty placeholder
        gpu_cache = []
        num_blocks = self.get_num_free_gpu_blocks()

        for _ in range(self.num_layers):
            # Each layer has key and value cache
            cache_shape = (2, num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
            cache = torch.empty(cache_shape, dtype=self.dtype, device="cuda")
            gpu_cache.append(cache)

        return gpu_cache

    def allocate_cpu_cache(self) -> List[Tensor]:
        """Allocate CPU cache tensors."""
        cpu_cache = []
        num_blocks = self.get_num_free_cpu_blocks()

        if num_blocks == 0:
            return cpu_cache

        for _ in range(self.num_layers):
            cache_shape = (2, num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
            cache = torch.empty(cache_shape, dtype=self.dtype, device="cpu", pin_memory=True)
            cpu_cache.append(cache)

        return cpu_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        """
        Swap blocks from CPU to GPU.

        Args:
            src_to_dst: Mapping from CPU block IDs to GPU block IDs.
        """
        for cpu_block_id, gpu_block_id in src_to_dst.items():
            self.manager.memory_pool.move_to_gpu(cpu_block_id)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        """
        Swap blocks from GPU to CPU.

        Args:
            src_to_dst: Mapping from GPU block IDs to CPU block IDs.
        """
        for gpu_block_id, cpu_block_id in src_to_dst.items():
            self.manager.memory_pool.move_to_cpu(gpu_block_id)

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        """
        Copy blocks (for copy-on-write).

        Args:
            src_to_dsts: Mapping from source block to destination blocks.
        """
        for src_id, dst_ids in src_to_dsts.items():
            src_block = self.manager.memory_pool.get_block(src_id)
            if src_block is None:
                continue

            for dst_id in dst_ids:
                # Clone the block
                cloned = src_block.clone()
                self.manager.memory_pool.register(cloned)

    def lookup_prefix(self, token_ids: List[int]) -> Tuple[int, Optional[List[int]]]:
        """
        Look up cached prefix for a sequence.

        Args:
            token_ids: Token IDs for the sequence.

        Returns:
            Tuple of (matched_length, block_ids).
        """
        result = self.manager.lookup(token_ids)

        if result.has_match and result.matched_node:
            # Get block IDs for the matched prefix
            block_ids = []
            node = result.matched_node
            while node and node.cache_block:
                block_ids.append(node.cache_block.block_id)
                node = node.parent
            block_ids.reverse()
            return result.matched_length, block_ids

        return 0, None

    def store_kv_cache(
        self,
        seq_id: int,
        token_ids: List[int],
        key_cache: Tensor,
        value_cache: Tensor,
    ) -> List[int]:
        """
        Store KV cache for a sequence.

        Args:
            seq_id: Sequence ID.
            token_ids: Token IDs.
            key_cache: Key cache tensor.
            value_cache: Value cache tensor.

        Returns:
            List of allocated block IDs.
        """
        cache_block = self.manager.insert(token_ids, key_cache, value_cache)
        block_ids = [cache_block.block_id]

        # Track for this sequence
        self._seq_block_map[seq_id] = block_ids

        return block_ids

    def free_sequence(self, seq_id: int) -> None:
        """
        Free cache blocks for a sequence.

        Args:
            seq_id: Sequence ID to free.
        """
        if seq_id in self._seq_block_map:
            block_ids = self._seq_block_map.pop(seq_id)
            for block_id in block_ids:
                # Release reference, don't delete (may be shared)
                block = self.manager.memory_pool.get_block(block_id)
                if block:
                    block.release_ref()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.manager.get_stats()

    @staticmethod
    def get_cache_block_size(
        block_size: int,
        cache_dtype: torch.dtype,
        model_config: "ModelConfig",
        parallel_config: "ParallelConfig",
    ) -> int:
        """
        Calculate cache block size in bytes.

        This is a static method called by vLLM for planning.
        """
        hf_config = model_config.hf_config
        num_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        num_layers = hf_config.num_hidden_layers

        element_size = torch.tensor([], dtype=cache_dtype).element_size()

        # Key + Value for all layers
        return 2 * num_layers * block_size * num_heads * head_dim * element_size


def patch_vllm_cache_engine() -> None:
    """
    Monkey-patch vLLM to use DeltaCacheEngine.

    Call this before creating vLLM LLM instance.
    """
    try:
        import vllm.worker.cache_engine as cache_module
        cache_module.CacheEngine = DeltaCacheEngine
        print("DeltaCache: Patched vLLM CacheEngine successfully")
    except ImportError:
        print("DeltaCache: vLLM not installed, skipping patch")
    except Exception as e:
        print(f"DeltaCache: Failed to patch vLLM: {e}")
