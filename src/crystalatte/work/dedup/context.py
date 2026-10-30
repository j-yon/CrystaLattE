"""Per-run state shared by the deduplication modules.

These are module-level globals rather than parameters so every deduplication
module reads the same per-run binding without threading it through each call.

IMPORTANT: :func:`bind` REBINDS these names, so consumers must reach
them through the module (``context.mass_by_ref``) and never via
``from .context import mass_by_ref`` — the latter captures the object present at
import time and silently keeps reading stale state.
"""

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from ...core.crystal import Crystal
from ...core.multimer import Monomer
from ...core.space_group import SITE_ATOL
from . import dressing, linear

# Module-level bindings set by `bind`.
# Per-reference lists indexed by reference index (one entry per independent monomer).
active_crystal: Crystal | None = None
references: list[Monomer] = []
mass_by_ref: list[NDArray] = []
# Each reference's dressing group D (site symmetry plus exact cell-symmetry self-isometries, see
# :mod:`.dressing`), stacked as ``(D, D^-1, D_tr)`` arrays for the vectorized orbit key.
dress_stacked_by_ref: list[tuple[NDArray, NDArray, NDArray]] = []
# Whether each reference takes the linear dot-product key, and its line unknowns (``None`` if not).
linear_by_ref: list[bool] = []
line_by_ref: list[linear.LineParams | None] = []
# Basis of the symmetric metrics the cell symmetry preserves, shared by every reference.
metric_pieces: NDArray = np.empty((0, 3, 3), dtype=np.int64)
# Rank of each reference sorted by ``(linear_by_ref[i], i)``: non-linear references anchor first.
anchor_rank: NDArray = np.empty(0, dtype=np.int64)


def bind(
    crystal: Crystal,
    monomers: list[Monomer],
    site_atol: float = SITE_ATOL,
) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the module globals for per-reference state: the reference monomers, their mass
    arrays, the dressing group the orbit key canonicalises over, the linear setup of each
    reference, the cell's metric basis, and the anchor rank that puts non-linear references first.

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomers: The reference monomers (one per crystal asymmetric unit)
    :param site_atol: Fractional tolerance for the site-symmetry test; must
        match the one :func:`generate_neighbors` passes to
        ``get_unique_sym_ops`` or stabilizer and orbit disagree
    :returns: ``None``
    """
    global active_crystal, references, mass_by_ref, dress_stacked_by_ref
    global linear_by_ref, line_by_ref, metric_pieces, anchor_rank
    active_crystal = crystal
    references = list(monomers)
    lattice = np.asarray(crystal.lattice_vectors, dtype=float)

    # the cell's integer symmetry depends only on the crystal, so it is shared by every reference
    H = dressing.cell_symmetry(
        np.stack([op.rot for op in crystal.space_group.sym_ops]).astype(np.int64), lattice
    )

    metric_pieces = linear.metric_basis(H)

    mass_by_ref, dress_stacked_by_ref = [], []
    line_by_ref = []
    for mon in references:
        mass_by_ref.append(
            np.array([qcel.periodictable.to_mass(s) for s in mon.symbols])
        )
        W, W_tr = dressing.site_symmetry(crystal.space_group, mon.centroid_frac, site_atol)
        Z = np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)
        D, D_tr = dressing.dressing_group(
            mon.frac_coords, mon.centroid_frac, Z, lattice, W, W_tr, H
        )
        dress_stacked_by_ref.append(
            (D, np.rint(np.linalg.inv(D)).astype(np.int64), D_tr)
        )
        line_by_ref.append(
            linear.line_params(mon.frac_coords, mon.centroid_frac, Z, lattice, D, D_tr)
            if linear.is_linear(mon.cart_coords)
            else None
        )

    linear_by_ref = [p is not None for p in line_by_ref]
    anchor_rank = np.empty(len(references), dtype=np.int64)
    anchor_rank[sorted(range(len(references)), key=lambda i: (linear_by_ref[i], i))] = np.arange(
        len(references)
    )
