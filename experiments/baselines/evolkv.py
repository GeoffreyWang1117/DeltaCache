"""EvolKV baseline (EMNLP 2025 Findings): Evolutionary per-layer budget.

Uses evolutionary search to find optimal per-layer token budget vector.
Calibrates on a few prompts, then applies the found allocation to test.

Note: This is a two-phase method — calibration (slow) then application
(fast). We provide both a calibrate() method and a compress() method.
For fair timing comparison, compress() uses pre-calibrated allocations.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from .base import BaselineMethod, h2o_token_selection, register_baseline


@register_baseline
class EvolKV(BaselineMethod):
    name = "evolkv"
    category = "eviction"
    requires_attention = True
    is_per_layer = True
    reference = "Yu & Chai, EMNLP 2025 Findings"

    def __init__(self, num_layers, num_heads, head_dim, **kwargs):
        super().__init__(num_layers, num_heads, head_dim)
        self.sink_tokens = kwargs.get("sink_tokens", 4)
        self.recent_tokens = kwargs.get("recent_tokens", 16)
        self.pop_size = kwargs.get("pop_size", 12)
        self.generations = kwargs.get("generations", 15)
        self.mutation_rate = kwargs.get("mutation_rate", 0.15)
        self.token_step = kwargs.get("token_step", 8)
        # Pre-calibrated allocation (set by calibrate())
        self._calibrated_budgets: Optional[List[int]] = None

    def calibrate(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
    ) -> List[int]:
        """Run evolutionary search to find optimal per-layer budgets.

        Call this once on calibration data, then reuse for test data.
        """
        seq_len = full_keys.shape[1]
        total_budget = int(seq_len * self.num_layers / compression_ratio)
        min_tokens = min(self.sink_tokens + self.recent_tokens, seq_len)

        # Initialize population
        population = []
        for _ in range(self.pop_size):
            individual = self._random_allocation(total_budget, seq_len, min_tokens)
            population.append(individual)

        # Add uniform allocation as one individual
        uniform = [total_budget // self.num_layers] * self.num_layers
        uniform = [max(min_tokens, min(b, seq_len)) for b in uniform]
        population[0] = uniform

        best_fitness = -1.0
        best_individual = population[0]

        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = []
            for individual in population:
                fitness = self._evaluate_fitness(
                    individual, full_keys, full_values, attention_weights, seq_len,
                )
                fitnesses.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual[:]

            # Selection + crossover + mutation
            new_pop = [best_individual[:]]  # Elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                child = self._crossover(p1, p2, total_budget, seq_len, min_tokens)
                child = self._mutate(child, total_budget, seq_len, min_tokens)
                new_pop.append(child)
            population = new_pop

        self._calibrated_budgets = best_individual
        return best_individual

    def compress(
        self,
        full_keys: Tensor,
        full_values: Tensor,
        compression_ratio: float,
        attention_weights: Optional[List[Tensor]] = None,
        hidden_states: Optional[List[Tensor]] = None,
    ) -> List[Tuple[Tensor, Tensor, Tensor]]:
        seq_len = full_keys.shape[1]

        if self._calibrated_budgets is None:
            # Auto-calibrate on the given data
            self.calibrate(full_keys, full_values, compression_ratio, attention_weights)

        budgets = self._calibrated_budgets
        assert budgets is not None

        # Adjust budgets if seq_len differs from calibration
        if len(budgets) != self.num_layers:
            budgets = [int(seq_len / compression_ratio)] * self.num_layers

        results = []
        for l in range(self.num_layers):
            n = min(budgets[l], seq_len)
            attn = attention_weights[l] if attention_weights and l < len(attention_weights) else None
            indices = h2o_token_selection(
                full_keys[l:l+1], full_values[l:l+1],
                n, seq_len,
                attention_weights=attn,
                sink_tokens=self.sink_tokens,
                recent_tokens=self.recent_tokens,
            )
            results.append((
                full_keys[l:l+1, indices.long()],
                full_values[l:l+1, indices.long()],
                indices,
            ))
        return results

    def _random_allocation(
        self, total_budget: int, seq_len: int, min_tokens: int,
    ) -> List[int]:
        """Generate random allocation summing to ~total_budget."""
        raw = [random.randint(min_tokens, seq_len) for _ in range(self.num_layers)]
        total = sum(raw)
        if total > 0:
            scale = total_budget / total
            raw = [max(min_tokens, min(int(b * scale), seq_len)) for b in raw]
        return raw

    def _evaluate_fitness(
        self,
        budgets: List[int],
        full_keys: Tensor,
        full_values: Tensor,
        attention_weights: Optional[List[Tensor]],
        seq_len: int,
    ) -> float:
        """Fast fitness: fraction of attention mass retained by selected tokens."""
        total_mass = 0.0
        for l in range(self.num_layers):
            n = min(budgets[l], seq_len)
            attn = attention_weights[l] if attention_weights and l < len(attention_weights) else None
            indices = h2o_token_selection(
                full_keys[l:l+1], full_values[l:l+1],
                n, seq_len, attention_weights=attn,
            )
            idx = indices.long()

            # Fitness = attention mass captured by retained tokens
            if attn is not None:
                attn_row = attn[0, :, -1, :].mean(dim=0).cpu()  # (S,)
                total_attn = attn_row.sum().item()
                retained_attn = attn_row[idx].sum().item()
                mass = retained_attn / max(total_attn, 1e-10)
            else:
                mass = n / max(seq_len, 1)
            total_mass += mass

        return total_mass / self.num_layers

    def _tournament(
        self, population: List[List[int]], fitnesses: List[float], k: int = 3,
    ) -> List[int]:
        """Tournament selection."""
        candidates = random.sample(range(len(population)), min(k, len(population)))
        best = max(candidates, key=lambda i: fitnesses[i])
        return population[best][:]

    def _crossover(
        self, p1: List[int], p2: List[int],
        total_budget: int, seq_len: int, min_tokens: int,
    ) -> List[int]:
        """Uniform crossover with budget normalization."""
        child = []
        for l in range(self.num_layers):
            if random.random() < 0.5:
                child.append(p1[l])
            else:
                child.append(p2[l])
        # Normalize to total budget
        total = sum(child)
        if total > 0:
            scale = total_budget / total
            child = [max(min_tokens, min(int(b * scale), seq_len)) for b in child]
        return child

    def _mutate(
        self, individual: List[int],
        total_budget: int, seq_len: int, min_tokens: int,
    ) -> List[int]:
        """Mutate: randomly adjust one layer's budget."""
        if random.random() > self.mutation_rate:
            return individual

        l = random.randint(0, self.num_layers - 1)
        delta = random.choice([-self.token_step, self.token_step])
        individual[l] = max(min_tokens, min(individual[l] + delta, seq_len))

        # Compensate on a random neighbor
        neighbor = random.randint(0, self.num_layers - 1)
        individual[neighbor] = max(min_tokens, min(individual[neighbor] - delta, seq_len))
        return individual
