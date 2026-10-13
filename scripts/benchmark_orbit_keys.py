"""Benchmark the per-cluster orbit key against a batched prototype on X23 fixtures

The prototype (Z' = 1 only) computes every candidate's relative transforms at once, runs the
site-symmetry dressing once per distinct relative transform, and packs each 13-integer row into an
order-preserving int64. It must induce the same partition of the candidates as
:func:`~crystalatte.work.dedup.invariants.orbit_key`.

Run from the repository root: ``PYTHONPATH=src python scripts/benchmark_orbit_keys.py``
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import load  # noqa: E402

from crystalatte.work import generate, neighbors  # noqa: E402
from crystalatte.work.dedup import context  # noqa: E402
from crystalatte.work.dedup.invariants import orbit_key  # noqa: E402

CASES = [
    ("acetic_acid", 2, 30.0), ("acetic_acid", 3, 9.0), ("acetic_acid", 3, 12.0),
    ("benzene", 2, 30.0), ("benzene", 3, 10.0),
    ("urea", 2, 30.0), ("urea", 3, 10.0),
    ("hexamine", 2, 20.0), ("triazine", 3, 9.0),
]


def pack(Lx, vx):
    """Order-preserving int64 code for rows with L entries in [-4, 3] and v entries in [-2048, 2047]"""
    assert Lx.min() >= -4 and Lx.max() <= 3 and np.abs(vx).max() < 2048
    code = np.zeros(Lx.shape[:-1], dtype=np.int64)
    for i in range(9):
        code = code * 8 + (Lx[..., i] + 4)
    for i in range(3):
        code = code * 4096 + (vx[..., i] + 2048)
    return code


def batched_keys(cands, W_stacked):
    """Z' = 1, one cluster size: every candidate's orbit key as an ``(N, n - 1)`` int64 array"""
    R = np.stack([g.rot_cache for g in cands]).astype(np.int64)
    t = np.stack([g.tr_cache for g in cands]).astype(np.int64)
    N, n = R.shape[:2]
    R_inv = np.rint(np.linalg.inv(R)).astype(np.int64)
    L = R_inv[:, :, None] @ R[:, None, :]
    v = (R_inv[:, :, None] @ (t[:, None] - t[:, :, None])[..., None])[..., 0]
    W, W_inv, d = W_stacked[0]
    S = len(W)

    # dress each distinct relative transform once
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    D = np.concatenate([np.concatenate((L[:, a, b].reshape(N, 9), v[:, a, b]), 1) for a, b in pairs])
    U, inv = np.unique(D, axis=0, return_inverse=True)
    Lu, vu = U[:, :9].reshape(-1, 1, 1, 3, 3), U[:, 9:].reshape(-1, 1, 1, 3, 1)
    Lx = W_inv[None, :, None] @ Lu @ W[None, None]
    vx = (W_inv[None, :, None] @ (Lu @ d[None, None, :, :, None] + vu - d[None, :, None, :, None]))[..., 0]
    table = pack(Lx.reshape(*Lx.shape[:3], 9), vx).min(axis=2)  # (U, S): coset minimum per anchor dressing
    T = table[inv.ravel()].reshape(len(pairs), N, S)

    M = np.empty((N, n, S, n - 1), dtype=np.int64)
    for p, (a, b) in enumerate(pairs):
        M[:, a, :, b - (b > a)] = T[p]
    M.sort(axis=3)
    flat = M.reshape(N * n * S, n - 1)
    order = np.lexsort((*flat.T[::-1], np.repeat(np.arange(N), n * S)))
    return flat[order[:: n * S]]


def run(name, N, R):
    crystal, _ = load(name)
    cands = []
    real = neighbors.deduplicate_bucket
    neighbors.deduplicate_bucket = lambda bucket: (cands.extend(bucket), real(bucket))[1]
    try:
        generate.generate(crystal, N, R, use_origin=True, n_jobs=1, verbose=0)
    finally:
        neighbors.deduplicate_bucket = real
    if len(context.W_stacked_by_ref) != 1:
        print(f"{name}: Z' > 1, prototype skipped")
        return

    for g in cands:
        g._rel_int_cache = g._orbit_key_cache = None
    t0 = time.perf_counter()
    ref = [orbit_key(g) for g in cands]
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, labels = np.unique(batched_keys(cands, context.W_stacked_by_ref), axis=0, return_inverse=True)
    t_bat = time.perf_counter() - t0

    labels = labels.ravel().tolist()
    same = len(set(zip(ref, labels))) == len(set(ref)) == len(set(labels))
    print(
        f"{name:12s} N={N} R={R:4.1f} |W|={len(context.W_stacked_by_ref[0][0]):2d} "
        f"candidates={len(cands):6d} | orbit_key {1e6 * t_ref / len(cands):6.1f} us "
        f"batched {1e6 * t_bat / len(cands):5.2f} us  x{t_ref / t_bat:5.1f} | same partition: {same}"
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    for case in CASES:
        run(*case)
