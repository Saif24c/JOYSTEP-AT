"""Low-overhead opt-in measurements for baseline performance runs."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import os
import time
import tracemalloc


@dataclass
class BaselineMetrics:
    enabled: bool = False
    frames: int = 0
    cursor_updates: int = 0
    samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    process_samples: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    started_at: float = field(default_factory=time.monotonic)

    def sample(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.samples[name].append(seconds)

    def frame(self) -> None:
        if self.enabled:
            self.frames += 1

    def cursor_update(self) -> None:
        if self.enabled:
            self.cursor_updates += 1

    def process_sample(self, values: dict[str, float]) -> None:
        if self.enabled:
            for name, value in values.items():
                self.process_samples[name].append(value)

    def summary(self) -> dict[str, float]:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        values: dict[str, float] = {
            "elapsed_s": elapsed,
            "fps": self.frames / elapsed,
            "cursor_updates_per_s": self.cursor_updates / elapsed,
        }
        for name, samples in self.samples.items():
            if samples:
                values[f"{name}_mean_ms"] = 1000 * sum(samples) / len(samples)
                values[f"{name}_max_ms"] = 1000 * max(samples)
        for name, samples in self.process_samples.items():
            if samples:
                values[f"{name}_mean"] = sum(samples) / len(samples)
                values[f"{name}_max"] = max(samples)
        return values


class ProcessSampler:
    """Best-effort CPU/RSS sampler with no mandatory production dependency."""
    def __init__(self) -> None:
        self._process = None
        try:
            import psutil  # Optional: installed for baseline measurements.
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(None)
        except ImportError:
            if not tracemalloc.is_tracing():
                tracemalloc.start()

    def sample(self) -> dict[str, float]:
        if self._process is not None:
            return {
                "cpu_percent": self._process.cpu_percent(None),
                "rss_bytes": float(self._process.memory_info().rss),
            }
        current, peak = tracemalloc.get_traced_memory()
        return {"python_heap_bytes": float(current), "python_heap_peak_bytes": float(peak)}
