"""Tests for PrefixTree."""

import pytest
import torch

from deltacache.core.prefix_tree import PrefixTree, PrefixTreeNode, LookupResult
from deltacache.core.cache_block import CacheBlock


class TestPrefixTreeNode:
    """Tests for PrefixTreeNode."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = PrefixTreeNode(token=42)
        assert node.token == 42
        assert node.parent is None
        assert len(node.children) == 0
        assert node.cache_block is None
        assert node.ref_count == 0

    def test_node_touch(self):
        """Test access tracking."""
        node = PrefixTreeNode(token=1)
        initial_access = node.access_count
        initial_time = node.last_access

        node.touch()

        assert node.access_count == initial_access + 1
        assert node.last_access >= initial_time

    def test_node_ref_counting(self):
        """Test reference counting."""
        node = PrefixTreeNode(token=1)
        assert node.ref_count == 0

        node.add_ref()
        assert node.ref_count == 1

        node.add_ref()
        assert node.ref_count == 2

        node.release_ref()
        assert node.ref_count == 1

        node.release_ref()
        assert node.ref_count == 0

        # Should not go negative
        node.release_ref()
        assert node.ref_count == 0

    def test_is_leaf(self):
        """Test leaf detection."""
        parent = PrefixTreeNode(token=1)
        child = PrefixTreeNode(token=2, parent=parent)
        parent.children[2] = child

        assert child.is_leaf
        assert not parent.is_leaf

    def test_subtree_size(self):
        """Test subtree size calculation."""
        root = PrefixTreeNode(token=0)
        child1 = PrefixTreeNode(token=1, parent=root)
        child2 = PrefixTreeNode(token=2, parent=root)
        grandchild = PrefixTreeNode(token=3, parent=child1)

        root.children[1] = child1
        root.children[2] = child2
        child1.children[3] = grandchild

        assert grandchild.subtree_size == 1
        assert child2.subtree_size == 1
        assert child1.subtree_size == 2
        assert root.subtree_size == 4

    def test_path_to_root(self):
        """Test path extraction."""
        root = PrefixTreeNode(token=-1)
        node1 = PrefixTreeNode(token=10, parent=root, depth=1)
        node2 = PrefixTreeNode(token=20, parent=node1, depth=2)
        node3 = PrefixTreeNode(token=30, parent=node2, depth=3)

        root.children[10] = node1
        node1.children[20] = node2
        node2.children[30] = node3

        path = node3.get_path_to_root()
        assert path == [10, 20, 30]


class TestPrefixTree:
    """Tests for PrefixTree."""

    def test_empty_tree(self):
        """Test empty tree."""
        tree = PrefixTree()
        assert tree.size == 0
        assert tree.num_cached == 0

    def test_insert_simple(self):
        """Test simple insertion."""
        tree = PrefixTree()
        tokens = [1, 2, 3, 4, 5]

        node = tree.insert(tokens)

        assert tree.size == 5
        assert node.token == 5
        assert tokens in tree

    def test_insert_with_cache(self):
        """Test insertion with cache block."""
        tree = PrefixTree()
        tokens = [1, 2, 3]

        # Create cache block
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        cache = CacheBlock(key, value)

        node = tree.insert(tokens, cache)

        assert node.has_cache
        assert node.cache_block is cache
        # With prefix caching enabled, all intermediate nodes also get cache
        # For [1, 2, 3] this means nodes 1, 2, and 3 all have cache
        assert tree.num_cached == 3

    def test_insert_overlapping(self):
        """Test insertion of overlapping sequences."""
        tree = PrefixTree()

        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])
        tree.insert([1, 2, 3, 5])

        # [1, 2] is shared, then branches
        assert tree.size == 5  # 1, 2, 3, 4, 5

    def test_lookup_exact_match(self):
        """Test lookup with exact match."""
        tree = PrefixTree()
        tokens = [1, 2, 3]

        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        cache = CacheBlock(key, value)
        tree.insert(tokens, cache)

        result = tree.lookup(tokens)

        assert result.has_match
        assert result.matched_length == 3
        assert result.kv_cache is not None

    def test_lookup_prefix_match(self):
        """Test lookup with prefix match."""
        tree = PrefixTree()

        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        tree.insert([1, 2, 3], CacheBlock(key, value))

        result = tree.lookup([1, 2, 3, 4, 5])

        assert result.has_match
        assert result.matched_length == 3
        assert result.kv_cache is not None

    def test_lookup_partial_match(self):
        """Test lookup with partial match (no cache at matched node)."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])  # No cache

        result = tree.lookup([1, 2, 3])

        assert result.matched_length == 3
        assert result.kv_cache is None

    def test_lookup_no_match(self):
        """Test lookup with no match."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])

        result = tree.lookup([4, 5, 6])

        assert not result.has_match
        assert result.matched_length == 0

    def test_lookup_longest_cached_prefix(self):
        """Test that lookup returns longest cached prefix."""
        tree = PrefixTree()

        key1 = torch.randn(2, 2, 4, 8)
        value1 = torch.randn(2, 2, 4, 8)
        tree.insert([1, 2], CacheBlock(key1, value1))

        # Insert longer sequence without cache
        tree.insert([1, 2, 3, 4])

        result = tree.lookup([1, 2, 3, 4, 5])

        # Should return the cached prefix at [1, 2]
        assert result.matched_length == 2
        assert result.kv_cache is not None

    def test_remove_leaf(self):
        """Test removal of leaf node."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])

        assert tree.remove([1, 2, 3])
        assert tree.size == 0
        assert [1, 2, 3] not in tree

    def test_remove_non_leaf_fails(self):
        """Test that removing non-leaf fails."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])

        # [1, 2] is not a leaf
        assert not tree.remove([1, 2])
        assert tree.size == 4

    def test_remove_cache(self):
        """Test cache removal by block ID."""
        tree = PrefixTree()

        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        cache = CacheBlock(key, value)
        tree.insert([1, 2, 3], cache)

        # With prefix caching, [1, 2, 3] creates caches at nodes 1, 2, and 3
        assert tree.num_cached == 3

        tree.remove_cache(cache.block_id)

        # Only removes the leaf cache, prefix caches remain
        assert tree.num_cached == 2
        result = tree.lookup([1, 2, 3])
        # The prefix cache at node 2 should still be available
        assert result.kv_cache is not None
        assert result.matched_length == 2

    def test_get_all_cached_nodes(self):
        """Test getting all cached nodes."""
        tree = PrefixTree()

        key1 = torch.randn(2, 2, 4, 8)
        value1 = torch.randn(2, 2, 4, 8)
        tree.insert([1, 2], CacheBlock(key1, value1))

        key2 = torch.randn(2, 3, 4, 8)
        value2 = torch.randn(2, 3, 4, 8)
        tree.insert([3, 4, 5], CacheBlock(key2, value2))

        cached = tree.get_all_cached_nodes()
        # With prefix caching:
        # [1, 2] creates caches at nodes 1 and 2 (2 caches)
        # [3, 4, 5] creates caches at nodes 3, 4, and 5 (3 caches)
        assert len(cached) == 5

    def test_iter_nodes(self):
        """Test iterating over all nodes."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])

        nodes = list(tree.iter_nodes())
        assert len(nodes) == 3

    def test_iter_leaves(self):
        """Test iterating over leaf nodes."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])

        leaves = list(tree.iter_leaves())
        assert len(leaves) == 2
        tokens = {leaf.token for leaf in leaves}
        assert tokens == {3, 4}

    def test_get_eviction_candidates(self):
        """Test getting eviction candidates."""
        tree = PrefixTree()

        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        tree.insert([1, 2, 3], CacheBlock(key, value))

        candidates = tree.get_eviction_candidates()
        # With prefix caching, [1, 2, 3] creates 3 cached nodes
        assert len(candidates) == 3

    def test_clear(self):
        """Test clearing tree."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])
        tree.insert([4, 5, 6])

        tree.clear()

        assert tree.size == 0
        assert tree.num_cached == 0

    def test_find_shared_prefix(self):
        """Test finding shared prefix."""
        tree = PrefixTree()
        tree.insert([1, 2, 3])
        tree.insert([1, 2, 4])

        tokens_list = [[1, 2, 3], [1, 2, 4], [1, 2, 5]]
        prefix, nodes = tree.find_shared_prefix(tokens_list)

        assert prefix == [1, 2]


class TestLookupResult:
    """Tests for LookupResult."""

    def test_has_match(self):
        """Test has_match property."""
        result1 = LookupResult(matched_length=0, matched_node=None)
        assert not result1.has_match

        node = PrefixTreeNode(token=1)
        result2 = LookupResult(matched_length=1, matched_node=node)
        assert result2.has_match

    def test_kv_cache(self):
        """Test kv_cache access."""
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)

        result = LookupResult(
            matched_length=3,
            matched_node=None,
            kv_cache=(key, value),
        )

        assert result.kv_cache is not None
        assert result.kv_cache[0].shape == key.shape
