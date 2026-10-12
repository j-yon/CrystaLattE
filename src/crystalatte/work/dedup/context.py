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
# The same site-symmetry dressings stacked as ``(W, W^-1, W_tr)`` arrays for the vectorized orbit key.
W_stacked_by_ref: list[tuple[NDArray, NDArray, NDArray]] = []
frac_to_cart: NDArray | None = None      # fractional -> Cartesian matrix (cart = frac_to_cart @ frac)
cart_to_frac: NDArray | None = None  # Cartesian -> fractional matrix


def bind(
    crystal: Crystal,
    monomers: list[Monomer],
    site_atol: float = SITE_ATOL,
    full_congruence_check: bool = False,
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
    :param full_congruence_check: Run the Cartesian congruence branch
        unconditionally, instead of only when the detected molecular point group
        is richer than the site symmetry (see ``check_congruence_by_ref``). The
        escape hatch for under-detected point groups; it cannot recover a
        dressing that was never detected in the first place.
    :returns: ``None``
    """
    global active_crystal, references, Z_by_ref, mass_by_ref, self_iso_by_ref
    global W_lin_by_ref, check_congruence_by_ref, frac_to_cart, cart_to_frac
    global self_iso_tr_by_ref, W_tr_by_ref, W_stacked_by_ref
    active_crystal = crystal
    references = list(monomers)
    frac_to_cart = np.asarray(crystal.lattice_vectors, dtype=float)
    cart_to_frac = np.linalg.inv(frac_to_cart)

    Z_by_ref, mass_by_ref, self_iso_by_ref = [], [], []
    W_lin_by_ref, check_congruence_by_ref = [], []
    self_iso_tr_by_ref, W_tr_by_ref, W_stacked_by_ref = [], [], []
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
        W_stack = np.stack(W)
        W_stacked_by_ref.append(
            (W_stack, np.rint(np.linalg.inv(W_stack)).astype(np.int64), np.stack(W_tr))
        )

        # W must be built BEFORE S, because S is seeded with it. `W_lin <= S` is
        # a theorem, but the detector cannot realise it on its own: the missing
        # operations are rotoreflections, which its candidate library does not
        # contain at any tolerance and which closure cannot reach. Supplying
        # them here is the only route. Measured effect: adamantane 2 -> 4, ice
        # 2 -> 6, the other ten X23 references unchanged.
        #
        # Note the site symmetry was determined at the mass-weighted centroid
        # while `PointGroup.detect` centres on the unweighted one. That is
        # harmless: a genuine symmetry permutes atoms preserving mass, so it
        # fixes the offset between the two centroids, and its linear part about
        # either is the same matrix.
        W_cart = [frac_to_cart @ w.astype(float) @ cart_to_frac for w in W]
        S = mon.point_group_seeded(W_cart)
        self_iso_by_ref.append(S)
        # Cartesian dressing translations: d = c - s @ c (zero iff s fixes c).
        c_cart = np.asarray(mon.centroid_cart, dtype=float)
        self_iso_tr_by_ref.append([c_cart - s @ c_cart for s in S])

        # Gate for the tolerant Cartesian branch: it can only help when the molecular
        # point group is richer than the site symmetry. Molecular operations outside W
        # can coincide with lattice symmetry the space group lacks (triazine in R-3c:
        # sigma_h and sigma_v pair up dimers beyond ~17 A), in any space group, so the
        # gate is no longer restricted to Sohncke groups. The branch runs only on the
        # orbit-key representatives, which keeps it cheap.
        #
        # S is detected numerically and can under-detect (adamantane finds 2 of 24
        # operations), so `full_congruence_check` must stay unconditional.
        check_congruence_by_ref.append(full_congruence_check or len(S) > len(W))


def init_worker(
    crystal: Crystal,
    monomers: list[Monomer],
    site_atol: float = SITE_ATOL,
    full_congruence_check: bool = False,
) -> None:
    """Process-pool initializer: bind the deduper globals in a worker process.

    :param crystal: The crystal passed to :func:`bind`
    :param monomers: The reference monomers passed to :func:`bind`
    :param site_atol: The tolerance passed to :func:`bind`
    :param full_congruence_check: The congruence gate passed to :func:`bind`;
        must match the main process or workers disagree with it
    :returns: ``None``
    """
    bind(crystal, monomers, site_atol, full_congruence_check)
