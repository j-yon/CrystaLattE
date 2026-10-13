"""Benchmark batched orbit labels against the per-cluster orbit key they replaced, on test fixtures

:func:`~crystalatte.work.dedup.invariants.orbit_labels` keys every candidate in one batch: relative
transforms for all clusters at once, the site-symmetry dressing once per distinct relative transform,
and rows compared as packed int64 codes. :func:`reference_orbit_key` is the per-cluster key it replaced,
kept here verbatim; both must induce the same partition of the candidates.

Run from the repository root: ``PYTHONPATH=src python scripts/benchmark_orbit_keys.py``
"""
import sys
import time
import warnings
from itertools import chain
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import load  # noqa: E402

from crystalatte.work import generate, neighbors  # noqa: E402
from crystalatte.work.dedup import context  # noqa: E402
from crystalatte.work.dedup.invariants import orbit_labels, relative_transforms  # noqa: E402

CASES = [
    ("acetic_acid", 2, 30.0), ("acetic_acid", 3, 9.0), ("acetic_acid", 3, 12.0),
    ("benzene", 2, 30.0), ("benzene", 3, 10.0),
    ("urea", 2, 30.0), ("urea", 3, 10.0),
    ("hexamine", 2, 20.0), ("triazine", 3, 9.0),
    ("ammonia", 2, 15.0), ("ammonia", 3, 7.0),
    ("fomepizole", 2, 15.0), ("hydroxyurea", 2, 15.0), ("hydroxyurea", 3, 8.0),
]


def _lex_order(x):
    """Per-group lexicographic sort orders for the rows of a ``(G, K, m)`` array"""
    G, K, m = x.shape
    flat = x.reshape(G * K, m)
    order = np.lexsort((*flat.T[::-1], np.repeat(np.arange(G), K)))
    return order.reshape(G, K) - (np.arange(G) * K)[:, None]


def reference_orbit_key(g_N):
    """The per-cluster orbit key replaced by ``orbit_labels`` (verbatim apart from caching)"""
    L, v = relative_transforms(g_N, integer=True)
    refs = g_N.ref_indices.tolist()
    n = len(refs)

    if all(len(context.W_stacked_by_ref[r][0]) == 1 for r in refs):
        rows = np.concatenate((L.reshape(n, n, 9), v), axis=2).tolist()
        return min(
            (refs[a], *chain.from_iterable(sorted((refs[b], *rows[a][b]) for b in range(n) if b != a)))
            for a in range(n)
        )

    best = None
    for a in range(n):
        _, Wa_inv, da = context.W_stacked_by_ref[refs[a]]
        S = len(Wa_inv)
        cosets = []
        for b in range(n):
            if b == a:
                continue
            Wb, _, db = context.W_stacked_by_ref[refs[b]]
            K = len(Wb)
            Lb = L[a, b] @ Wb
            vb = db @ L[a, b].T + v[a, b]
            Lx = np.einsum("sij,kjl->skil", Wa_inv, Lb).reshape(S, K, 9)
            vx = np.einsum("sij,skj->ski", Wa_inv, vb[None] - da[:, None])
            rows = np.concatenate((np.full((S, K, 1), refs[b]), Lx, vx), axis=2)
            cosets.append(rows[np.arange(S), _lex_order(rows)[:, 0]])
        rows = np.stack(cosets, axis=1)
        rows = rows[np.arange(S)[:, None], _lex_order(rows)].reshape(S, -1)
        cand = (refs[a], *rows[np.lexsort(rows.T[::-1])[0]].tolist())
        if best is None or cand < best:
            best = cand
    return best


def run(name, N, R):
    crystal, _ = load(name)
    cands = []
    real = neighbors.deduplicate_bucket
    neighbors.deduplicate_bucket = lambda bucket: (cands.extend(bucket), real(bucket))[1]
    try:
        generate.generate(crystal, N, R, use_origin=True, n_jobs=1, verbose=0)
    finally:
        neighbors.deduplicate_bucket = real

    for g in cands:
        g._rel_int_cache = None
    t0 = time.perf_counter()
    ref = [reference_orbit_key(g) for g in cands]
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels = orbit_labels(cands).tolist()
    t_bat = time.perf_counter() - t0

    same = len(set(zip(ref, labels))) == len(set(ref)) == len(set(labels))
    W_sizes = [len(W) for W, _, _ in context.W_stacked_by_ref]
    print(
        f"{name:12s} N={N} R={R:4.1f} |W|={W_sizes} candidates={len(cands):6d} classes={len(set(labels)):5d} | "
        f"per-cluster {1e6 * t_ref / len(cands):6.1f} us  batched {1e6 * t_bat / len(cands):5.2f} us  "
        f"x{t_ref / t_bat:5.1f} | same partition: {same}"
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    for case in CASES:
        run(*case)
