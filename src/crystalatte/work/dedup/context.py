"""Per-run state shared by the deduplication modules.

These are module-level globals rather than parameters because the bucket
deduplication runs under a :class:`~concurrent.futures.ProcessPoolExecutor`,
whose ``initializer`` has to install the state in each worker process before
any task runs.

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

# Module-level bindings set by `bind` (in main process) and by the
# worker initializer (`init_worker`, in ProcessPoolExecutor children).
# Per-reference lists indexed by reference index (one entry per independent monomer).
active_crystal: Crystal | None = None
references: list[Monomer] = []
Z_by_ref: list[NDArray] = []
mass_by_ref: list[NDArray] = []
self_iso_by_ref: list[list[NDArray]] = [[np.eye(3)]]
W_lin_by_ref: list[list[NDArray]] = [[np.eye(3, dtype=np.int64)]]
check_congruence_by_ref: list[bool] = [False]
# Translation parts of the dressings, index-aligned with the groups above.
# A monomer self-isometry is an isometry about the monomer's CENTROID, not
# about the coordinate origin, so as a map on the reference monomer's raw
# coordinates it is the affine map ``x -> s @ x + d`` with ``d = c - s @ c``.
# Dropping ``d`` (using ``s`` alone) is only valid when ``s @ c == c``.
self_iso_tr_by_ref: list[list[NDArray]] = [[np.zeros(3)]]        # Cartesian, Angstroms
W_tr_by_ref: list[list[NDArray]] = [[np.zeros(3, dtype=np.int64)]]  # lattice basis, twelfths
frac_to_cart: NDArray | None = None      # fractional -> Cartesian matrix (cart = frac_to_cart @ frac)
cart_to_frac: NDArray | None = None  # Cartesian -> fractional matrix


def bind(
    crystal: Crystal, monomers: list[Monomer], site_atol: float = SITE_ATOL
) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the module globals for per-reference state: per-reference Z and mass
    arrays, self-isometries, site-symmetry dressings and the congruence
    gates. Also sets the lattice vectors ``frac_to_cart`` / ``cart_to_frac`` for
    fractional<->Cartesian conversion.

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomers: The reference monomers (one per crystal asymmetric unit)
    :param site_atol: Fractional tolerance for the site-symmetry test; must
        match the one :func:`generate_neighbors` passes to
        ``get_unique_sym_ops`` or stabilizer and orbit disagree
    :returns: ``None``
    """
    global active_crystal, references, Z_by_ref, mass_by_ref, self_iso_by_ref
    global W_lin_by_ref, check_congruence_by_ref, frac_to_cart, cart_to_frac
    global self_iso_tr_by_ref, W_tr_by_ref
    active_crystal = crystal
    references = list(monomers)
    frac_to_cart = np.asarray(crystal.lattice_vectors, dtype=float)
    cart_to_frac = np.linalg.inv(frac_to_cart)

    Z_by_ref, mass_by_ref, self_iso_by_ref = [], [], []
    W_lin_by_ref, check_congruence_by_ref = [], []
    self_iso_tr_by_ref, W_tr_by_ref = [], []
    for mon in references:
        Z_by_ref.append(
            np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)
        )
        mass_by_ref.append(
            np.array([qcel.periodictable.to_mass(s) for s in mon.symbols])
        )
        S = mon.self_isometries
        self_iso_by_ref.append(S)
        # Cartesian dressing translations: d = c - s @ c (zero iff s fixes c).
        c_cart = np.asarray(mon.centroid_cart, dtype=float)
        self_iso_tr_by_ref.append([c_cart - s @ c_cart for s in S])

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
        check_congruence_by_ref.append(len(S) > len(W))


def init_worker(
    crystal: Crystal, monomers: list[Monomer], site_atol: float = SITE_ATOL
) -> None:
    """Process-pool initializer: bind the deduper globals in a worker process.

    :param crystal: The crystal passed to :func:`bind`
    :param monomers: The reference monomers passed to :func:`bind`
    :param site_atol: The tolerance passed to :func:`bind`
    :returns: ``None``
    """
    bind(crystal, monomers, site_atol)
