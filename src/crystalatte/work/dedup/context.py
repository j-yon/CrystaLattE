"""Per-run state shared by the deduplication modules.

These are module-level globals rather than parameters so every deduplication
module reads the same per-run binding without threading it through each call.

IMPORTANT: :func:`bind` REBINDS these names, so consumers must reach
them through the module (``context.Z_by_ref``) and never via
``from .context import Z_by_ref`` — the latter captures the object present at
import time and silently keeps reading stale state.
"""

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from ...core.crystal import Crystal
from ...core.multimer import Monomer
from ...core.space_group import SITE_ATOL
from . import dressing

# Module-level bindings set by `bind`.
# Per-reference lists indexed by reference index (one entry per independent monomer).
active_crystal: Crystal | None = None
references: list[Monomer] = []
Z_by_ref: list[NDArray] = []
mass_by_ref: list[NDArray] = []
# Site symmetry: the space-group operations fixing each reference monomer. A dressing is
# an isometry about the monomer's CENTROID, not the coordinate origin, so as a map on the
# reference's raw coordinates it is the affine ``x -> w @ x + t / 12`` with
# ``t / 12 = c - w @ c``; dropping ``t`` is only valid when ``w @ c == c``.
W_lin_by_ref: list[list[NDArray]] = [[np.eye(3, dtype=np.int64)]]
W_tr_by_ref: list[list[NDArray]] = [
    [np.zeros(3, dtype=np.int64)]
]  # lattice basis, twelfths
# Each reference's dressing group D (site symmetry plus exact cell-symmetry self-isometries, see
# :mod:`.dressing`), stacked as ``(D, D^-1, D_tr)`` arrays for the vectorized orbit key.
dress_stacked_by_ref: list[tuple[NDArray, NDArray, NDArray]] = []
frac_to_cart: NDArray | None = (
    None  # fractional -> Cartesian matrix (cart = frac_to_cart @ frac)
)


def bind(
    crystal: Crystal,
    monomers: list[Monomer],
    site_atol: float = SITE_ATOL,
) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the module globals for per-reference state: per-reference Z and mass
    arrays, the site symmetry, and the dressing group the orbit key canonicalises
    over. Also sets the lattice matrix ``frac_to_cart``.

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomers: The reference monomers (one per crystal asymmetric unit)
    :param site_atol: Fractional tolerance for the site-symmetry test; must
        match the one :func:`generate_neighbors` passes to
        ``get_unique_sym_ops`` or stabilizer and orbit disagree
    :returns: ``None``
    """
    global active_crystal, references, Z_by_ref, mass_by_ref, frac_to_cart
    global W_lin_by_ref, W_tr_by_ref, dress_stacked_by_ref
    active_crystal = crystal
    references = list(monomers)
    frac_to_cart = np.asarray(crystal.lattice_vectors, dtype=float)

    # the cell's integer symmetry depends only on the crystal, so it is shared by every reference
    H = dressing.cell_symmetry(
        np.stack([op.rot for op in crystal.space_group.sym_ops]).astype(np.int64), frac_to_cart
    )

    Z_by_ref, mass_by_ref = [], []
    W_lin_by_ref, W_tr_by_ref, dress_stacked_by_ref = [], [], []
    for mon in references:
        Z_by_ref.append(
            np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)
        )
        mass_by_ref.append(
            np.array([qcel.periodictable.to_mass(s) for s in mon.symbols])
        )
        site_ops = crystal.space_group.get_site_symmetry(
            mon.centroid_frac, atol=site_atol
        )
        c_frac = np.asarray(mon.centroid_frac, dtype=float)
        W, W_tr = [], []
        for op in site_ops:
            r = op.rot.astype(np.int64)
            if any(np.array_equal(r, w) for w in W):
                continue
            # The dressing must fix the centroid EXACTLY, but a site op fixes it
            # only modulo a lattice vector, so strip that lattice part off the
            # op's own translation. All integer, in twelfths.
            tr = op.tr.astype(np.int64)
            lattice = np.rint(r @ c_frac + tr / 12.0 - c_frac).astype(np.int64)
            W.append(r)
            W_tr.append(tr - 12 * lattice)
        if not W:
            W = [np.eye(3, dtype=np.int64)]
            W_tr = [np.zeros(3, dtype=np.int64)]
        W_lin_by_ref.append(W)
        W_tr_by_ref.append(W_tr)
        D, D_tr = dressing.dressing_group(
            mon.frac_coords, c_frac, Z_by_ref[-1], frac_to_cart, W, W_tr, H
        )
        dress_stacked_by_ref.append(
            (D, np.rint(np.linalg.inv(D)).astype(np.int64), D_tr)
        )
