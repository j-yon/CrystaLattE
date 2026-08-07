"""Multimer deduplication.

The pipeline is: invariants -> matchers -> routing.  :mod:`.context` holds the
per-run state they share (see the warning there about rebinding).
"""

from .bijection import _is_bijection, _process_bucket
from .context import _bind_globals, _init_bucket_worker

__all__ = ["_is_bijection", "_process_bucket", "_bind_globals", "_init_bucket_worker"]
