"""Prometheus-compatible metrics for fraud assessments.

Exposes counters and histograms that the ``/metrics`` endpoint can
scrape. Uses the stdlib approach (no Prometheus client library
required) — the A2A server's ``/metrics`` endpoint renders them.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class FraudMetrics:
    """In-memory metrics store, thread-safe via a lock.

    Production deployments would use ``prometheus_client`` with a proper
    registry. This lightweight implementation avoids the dependency while
    the system is in early development.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    assessments_total: int = 0
    risk_level_counts: Counter = field(default_factory=Counter)
    llm_invocations: int = 0
    llm_fallbacks: int = 0
    score_buckets: Counter = field(default_factory=Counter)
    batch_customers_processed: int = 0

    def record_assessment(self, score: int, risk_level: str, used_llm: bool) -> None:
        """Record a completed assessment."""
        with self._lock:
            self.assessments_total += 1
            self.risk_level_counts[risk_level] += 1
            bucket = (score // 10) * 10
            self.score_buckets[bucket] += 1
            if used_llm:
                self.llm_invocations += 1

    def record_fallback(self) -> None:
        """Record an LLM fallback (score-only assessment)."""
        with self._lock:
            self.llm_fallbacks += 1

    def record_batch_processed(self, count: int) -> None:
        """Record batch processing count."""
        with self._lock:
            self.batch_customers_processed += count

    def snapshot(self) -> dict:
        """Return a snapshot of all metrics as a dict."""
        with self._lock:
            return {
                "assessments_total": self.assessments_total,
                "risk_level_counts": dict(self.risk_level_counts),
                "llm_invocations": self.llm_invocations,
                "llm_fallbacks": self.llm_fallbacks,
                "score_buckets": dict(self.score_buckets),
                "batch_customers_processed": self.batch_customers_processed,
            }


_metrics = FraudMetrics()


def get_metrics() -> FraudMetrics:
    """Return the global metrics instance."""
    return _metrics
