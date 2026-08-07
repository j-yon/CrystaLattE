"""Generate unique N-mers from a crystal.

This module is the public entry point and the orchestration layer; the work is
split across three siblings, each with one job:

- :mod:`crystalatte.work.neighbors` — enumerate candidate N-mers by anchoring
  each reference monomer and combining the partners within the cutoff.
- :mod:`crystalatte.work.dedup` — decide which candidates are the same
  multimer. Internally: ``invariants`` (isometry invariants of a cluster),
  ``matchers`` (the equivalence tests), ``bijection`` (routing + bucket dedup),
  and ``context`` (the per-run state they share).
- this module — supercell bounds, tolerance plumbing, and turning each
  surviving cluster into a :class:`~crystalatte.core.multimer.Multimer`.

The correctness argument for the deduplication lives in
``docs/dedup_correctness.tex``.
"""

import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOpList
from .dedup import context
from .neighbors import _generate_neighbors


def _process_multimer(g_N: SymOpList) -> Multimer | None:
    """Construct a :class:`~crystalatte.core.multimer.Multimer` from a deduped
    candidate cluster.

    :param g_N: The deduplicated cluster to materialize
    :returns: The constructed :class:`~crystalatte.core.multimer.Multimer`, or
        ``None`` if ``g_N`` contains two members that are the same
        ``(base reference, SymOp)`` pair (which can happen when multiple
        translations land on a degenerate Wyckoff orbit and would produce
        overlapping monomers)
    """
    refs = g_N.ref_indices
    ops = list(g_N)
    # degeneracy: reject only if two members are the SAME (ref, op) pair
    seen = set(zip(refs.tolist(), ops))
    if len(seen) != len(ops):
        return None
    mons: list[Monomer] = []
    for k, g in enumerate(ops):
        base = context._monomers[refs[k]]
        masses = context._mass_by_ref[refs[k]]
        mon_frac = g.apply(base.frac_coords)
        mon_cart = context._crystal.to_cartesian(mon_frac)
        com_frac = np.average(mon_frac, axis=0, weights=masses)
        com_cart = context._crystal.to_cartesian(com_frac)
        mons.append(Monomer(base.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
    return Multimer(mons, g_N, g_N.multiplicity)


def generate(crystal: Crystal, N: int, R: float, *, detect_symmetry=True, symprec=1e-3, **kwargs) -> list[Multimer]:
    """Generate unique multimers of every reference molecule in the crystal.

    :param crystal: The crystal structure to generate multimers from; all
        symmetry-independent reference molecules (:meth:`Crystal.get_references`)
        are enumerated
    :param N: The number of monomers in the multimer (e.g. ``N=2`` for dimers)
    :param R: The maximum center-to-center distance in Angstroms for monomers to
        be considered part of the same multimer
    :param detect_symmetry: If True, automatically detect and apply symmetry via
        :meth:`Crystal.detect_symmetry` before enumeration (default: True)
    :param symprec: Symmetry detection tolerance in Angstroms (default: 1e-3)
    :param kwargs: Additional options forwarded to :func:`_generate_neighbors`
    :returns: A list of unique :class:`~crystalatte.core.multimer.Multimer`
        objects representing the multimers found in the crystal
    """
    if detect_symmetry:
        crystal = crystal.detect_symmetry(symprec)
    monomers = crystal.get_references()
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    R_eff = R + max(pdist(m.cart_coords).max() for m in monomers)

    lows, highs = [], []
    for m in monomers:
        t = m.centroid_frac
        lows.append(-np.ceil((R_eff - t * d) / d))
        highs.append(np.ceil((R_eff - (1 - t) * d) / d))
    bounds = np.vstack((np.min(lows, axis=0), np.max(highs, axis=0))).T.astype(int)

    # Site-symmetry tests compare FRACTIONAL vectors, but symprec is in
    # Angstroms, so convert. Using the shortest cell edge keeps the fractional
    # tolerance conservative along every axis. Floored at the module default so
    # a tight symprec cannot reintroduce the CIF-rounding failure (a special
    # position written as 0.3333 instead of 1/3 sits ~1e-4 off the element).
    kwargs.setdefault(
        "site_atol", max(SITE_ATOL, float(symprec) / float(d.min()))
    )

    unique_neighbors, _ = _generate_neighbors(crystal, monomers, bounds, R, N, **kwargs)

    multimers: list[Multimer] = []
    for g_N in tqdm(
        unique_neighbors,
        total=len(unique_neighbors),
        desc="Processing unique multimers",
        leave=False,
        disable=not kwargs.get("verbose", 1),
    ):
        m = _process_multimer(g_N)
        if m is not None:
            multimers.append(m)
    return multimers
