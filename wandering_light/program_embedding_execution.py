"""Execution-derived I/O supervision, with incomplete results kept unknown."""

from __future__ import annotations

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
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
        self.program_counts = dict.fromkeys(self.counts, 0)

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

    def program_masks(
        self,
        examples_a,
        programs_a,
        examples_b,
        programs_b,
        *,
        own_a=None,
        own_b=None,
        device=None,
    ):
        """Mutual agreement on both shown sets, restricted to reliable witnesses.

        Optional own masks are the existing query-to-own-bank execution masks.
        Only their diagonals are needed; cross-grades retain original row order.
        """
        if len(examples_a) != len(programs_a) or len(examples_b) != len(programs_b):
            raise ValueError("Each program must have its own shown example set")

        def reliable(examples, programs, own):
            if own is not None:
                positive, known = own
                return (positive.diagonal() & known.diagonal()).cpu().tolist()
            return [
                self.matches(program, query) is True
                for query, program in zip(examples, programs, strict=True)
            ]

        reliable_a = reliable(examples_a, programs_a, own_a)
        reliable_b = reliable(examples_b, programs_b, own_b)
        positive = torch.zeros((len(programs_a), len(programs_b)), dtype=torch.bool)
        known = torch.zeros_like(positive)
        for i, (program_a, query_a) in enumerate(
            zip(programs_a, examples_a, strict=True)
        ):
            for j, (program_b, query_b) in enumerate(
                zip(programs_b, examples_b, strict=True)
            ):
                grade = None
                if reliable_a[i] and reliable_b[j]:
                    cross_b = self.matches(program_b, query_a)
                    cross_a = (
                        False if cross_b is False else self.matches(program_a, query_b)
                    )
                    if cross_b is False or cross_a is False:
                        grade = False
                    elif cross_b is True and cross_a is True:
                        grade = True
                positive[i, j] = grade is True
                known[i, j] = grade is not None
                self.program_counts[
                    "unknown" if grade is None else "positive" if grade else "negative"
                ] += 1
        return positive.to(device), known.to(device)

    def statistics(self):
        cache = self._grade.cache_info()
        return {
            **self.counts,
            **{f"program_{key}": value for key, value in self.program_counts.items()},
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "cache_entries": cache.currsize,
        }


def _label_rows(matcher, batch, rows, io, program):
    examples_a, programs_a, examples_b, programs_b = batch
    queries_a = [examples_a[i] for i in rows]
    queries_b = [examples_b[i] for i in rows]
    own_a = matcher.masks(queries_a, programs_a) if io else None
    own_b = matcher.masks(queries_b, programs_b) if io else None
    pairs = None
    if program:
        reliable_a = reliable_b = None
        if io:
            reliable_a = tuple(
                torch.diag(mask[torch.arange(len(rows)), rows]) for mask in own_a
            )
            if len(rows) == len(programs_b):
                reliable_b = own_b
            else:
                grades = [
                    matcher.matches(p, q)
                    for p, q in zip(programs_b, examples_b, strict=True)
                ]
                reliable_b = (
                    torch.diag(torch.tensor([grade is True for grade in grades])),
                    torch.diag(torch.tensor([grade is not None for grade in grades])),
                )
        pairs = matcher.program_masks(
            queries_a,
            [programs_a[i] for i in rows],
            examples_b,
            programs_b,
            own_a=reliable_a,
            own_b=reliable_b,
        )
    return own_a, own_b, pairs


def _init_label_worker(seconds, cache_size):
    global _worker_matcher  # noqa: PLW0603 -- one matcher per spawned process
    torch.set_num_threads(1)
    _worker_matcher = ExecutionMatcher(seconds, cache_size)


def _label_worker(task):
    batch, rows, io, program = task
    masks = _label_rows(_worker_matcher, batch, rows, io, program)
    # No tensors or execution output states cross the process boundary.
    payload = [
        [mask.tolist() for mask in pair] if pair is not None else None for pair in masks
    ]
    return payload, os.getpid(), _worker_matcher.statistics()


class ExecutionLabeler:
    """Whole-batch labels with optional persistent, CPU-only spawned workers."""

    def __init__(self, seconds=1.0, cache_size=65536, workers=0):
        if workers < 0:
            raise ValueError("Execution label workers must be nonnegative")
        self.matcher = ExecutionMatcher(seconds, cache_size)
        self.workers = workers
        self._worker_stats = {}
        self._closed = False
        self._pool = (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_label_worker,
                initargs=(seconds, cache_size),
            )
            if workers
            else None
        )

    def batch_masks(
        self,
        examples_a,
        programs_a,
        examples_b,
        programs_b,
        *,
        io,
        program,
        device=None,
    ):
        if self._closed:
            raise RuntimeError("Execution labeler is closed")
        size = len(programs_a)
        if not size or any(
            len(items) != size for items in (examples_a, examples_b, programs_b)
        ):
            raise ValueError("Execution labels require nonempty aligned batches")
        if not io and not program:
            return None, None, None
        batch = examples_a, programs_a, examples_b, programs_b
        if self._pool is None:
            masks = _label_rows(self.matcher, batch, list(range(size)), io, program)
        else:
            width = (size + self.workers - 1) // self.workers
            tasks = [
                (batch, list(range(start, min(start + width, size))), io, program)
                for start in range(0, size, width)
            ]
            chunks = []
            for payload, pid, statistics in self._pool.map(_label_worker, tasks):
                chunks.append(payload)
                self._worker_stats[pid] = statistics
            masks = [
                tuple(
                    torch.tensor(
                        [row for chunk in chunks for row in chunk[kind][field]],
                        dtype=torch.bool,
                    )
                    for field in range(2)
                )
                if chunks[0][kind] is not None
                else None
                for kind in range(3)
            ]
        return tuple(
            tuple(mask.to(device) for mask in pair) if pair is not None else None
            for pair in masks
        )

    def statistics(self):
        if not self.workers:
            return self.matcher.statistics()
        return {
            key: sum(stats[key] for stats in self._worker_stats.values())
            for key in self.matcher.statistics()
        }

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
