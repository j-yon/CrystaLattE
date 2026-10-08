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


def _is_sohncke(crystal: Crystal) -> bool:
    """Return True iff the space group contains no improper operation.

    In such a group every crystallographic linear part has ``det = +1``, so by
    ``docs/dedup_correctness.tex`` Thm ``thm:sohncke`` an enantiomeric merge
    forces an IMPROPER dressing, which the site symmetry cannot supply. Only the
    Cartesian branch, dressed by the full molecular point group, can certify it.

    :param crystal: The crystal whose space group is tested
    :returns: True iff every symmetry operation is proper
    """
    return all(
        np.linalg.det(np.asarray(op.rot, dtype=float)) > 0
        for op in crystal.space_group.sym_ops
    )


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
    sohncke = _is_sohncke(crystal)
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

        # Gate for the tolerant Cartesian branch. The branch can only ever help
        # when the molecular point group is richer than the site symmetry, so
        # `len(S) > len(W)` is a necessary condition -- but it is a very weak
        # one: it holds for benzene, ammonia and most other crystals, where the
        # branch then costs 32-44% of trimer runtime and yields ZERO merges.
        #
        # It is retained for exactly ONE situation, and that one is a theorem:
        # a SOHNCKE group with an ACHIRAL monomer, where an enantiomeric merge
        # forces det(s_0) = -1 while W_lin is entirely proper (Thm thm:sohncke's
        # determinant identity), so the site symmetry cannot supply the dressing.
        # No X23 crystal actually exercises it -- ammonia opens the gate and
        # yields 0 merges -- but the requirement is real.
        #
        # LINEAR MONOMERS USED TO OPEN THIS GATE AND DELIBERATELY NO LONGER DO.
        # A linear molecule's continuous point group is only ever finitely
        # sampled, and the dressings it needed were cancelling a phantom degree
        # of freedom (the twist about the molecular axis) rather than doing any
        # physical work -- one of them displaced no atom by more than 0.000 A.
        # Linear monomers are now out of scope; see the note in `dedup/__init__`.
        #
        # The gate is NOT proven exhaustive: a proper, finite dressing in
        # S \ W_lin in a non-Sohncke group would also need the branch, and no
        # X23 crystal exhibits one. `full_congruence_check` is the escape hatch
        # for anything exotic, and it must run the branch UNCONDITIONALLY.
        #
        # `len(S) > len(W)` therefore sits INSIDE the narrow arm, not outside it.
        # Factoring it out looks equivalent and is not: S is detected numerically
        # and under-detects (adamantane finds 2 of 24 operations, ice 2 of 6), so
        # `len(S) > len(W)` can be false even though the true molecular group is
        # strictly richer than the site symmetry. Guarding the hatch with it made
        # the hatch silently inoperative for exactly the crystals whose detected
        # S is deficient -- the ones most likely to need it.
        #
        # Note the hatch still cannot recover a dressing that was never detected;
        # that is the separate under-detection defect (Caveat C2 in the .tex).
        # What it now guarantees is that the branch actually runs when asked.
        improper_S = any(float(np.linalg.det(s)) < 0 for s in S)
        check_congruence_by_ref.append(
            full_congruence_check or (len(S) > len(W) and sohncke and improper_S)
        )


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
