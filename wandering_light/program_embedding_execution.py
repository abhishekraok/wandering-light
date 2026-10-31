"""Execution-derived I/O supervision, with incomplete results kept unknown."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from wandering_light.program_embedding_eval import _matches


class ExecutionMatcher:
    """Cache bounded, per-example grades; never cache program output states."""

    def __init__(self, seconds: float = 1.0, cache_size: int = 65536):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Execution label timeout must be positive and finite")
        if cache_size < 0:
            raise ValueError("Execution label cache size must be nonnegative")
        self.seconds = seconds
        self.counts = {"positive": 0, "negative": 0, "unknown": 0}

        @lru_cache(maxsize=cache_size)
        def grade(program, given, wanted):
            return _matches(program, [(given, wanted)], seconds=seconds)

        self._grade = grade

    def matches(self, program, examples):
        if not examples:
            raise ValueError("Execution supervision requires shown examples")
        unknown = False
        for given, wanted in examples:
            grade = self._grade(program, given, wanted)
            if grade is False:
                return False
            unknown |= grade is None
        return None if unknown else True

    def masks(self, examples, programs, *, device=None):
        """Rows are I/O queries; columns are candidates, independent of record IDs."""
        positive = torch.zeros((len(examples), len(programs)), dtype=torch.bool)
        known = torch.zeros_like(positive)
        for i, query in enumerate(examples):
            for j, program in enumerate(programs):
                grade = self.matches(program, query)
                positive[i, j] = grade is True
                known[i, j] = grade is not None
                self.counts[
                    "unknown" if grade is None else "positive" if grade else "negative"
                ] += 1
        return positive.to(device), known.to(device)

    def statistics(self):
        cache = self._grade.cache_info()
        return {
            **self.counts,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "cache_entries": cache.currsize,
        }
