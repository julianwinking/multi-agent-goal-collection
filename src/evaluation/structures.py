"""Base structures for performance evaluation."""

from dataclasses import dataclass
import json
from dataclasses import asdict


@dataclass(frozen=True)
class PerformanceResults:
    """Base class for performance metrics."""

    def __str__(self):
        return json.dumps(asdict(self))
