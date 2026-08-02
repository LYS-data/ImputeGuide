"""Lightweight timing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator


@dataclass
class RuntimeMeasurement:
    """Mutable container populated by the timing context manager."""

    start: float
    end: float | None = None

    @property
    def elapsed_sec(self) -> float | None:
        if self.end is None:
            return None
        return self.end - self.start


@contextmanager
def measure_runtime() -> Iterator[RuntimeMeasurement]:
    """Measure wall-clock runtime with a small, reusable context manager."""
    measurement = RuntimeMeasurement(start=perf_counter())
    try:
        yield measurement
    finally:
        measurement.end = perf_counter()
