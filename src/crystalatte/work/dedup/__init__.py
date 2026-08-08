"""Multimer deduplication.

The pipeline is: invariants -> matchers -> routing.  :mod:`.context` holds the
per-run state they share (see the warning there about rebinding).
"""

from .equivalence import are_equivalent, deduplicate_bucket
from .context import bind, init_worker

__all__ = ["are_equivalent", "deduplicate_bucket", "bind", "init_worker"]
