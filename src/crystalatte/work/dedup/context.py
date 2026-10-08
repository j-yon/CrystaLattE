"""Per-run state shared by the deduplication modules.

Module globals rather than parameters, so a ProcessPoolExecutor initializer can
install them in each worker. Access them as ``context.X``, never by
from-import: :func:`bind` rebinds the names and a from-import goes stale.
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
# A self-isometry acts about the centroid, not the origin, so on raw
# coordinates it is ``x -> s @ x + d`` with ``d = c - s @ c``.
self_iso_tr_by_ref: list[list[NDArray]] = [[np.zeros(3)]]        # Cartesian, Angstroms
W_tr_by_ref: list[list[NDArray]] = [[np.zeros(3, dtype=np.int64)]]  # lattice basis, twelfths
frac_to_cart: NDArray | None = None      # fractional -> Cartesian matrix (cart = frac_to_cart @ frac)
cart_to_frac: NDArray | None = None  # Cartesian -> fractional matrix


def _is_sohncke(crystal: Crystal) -> bool:
    """Return True iff the space group contains no improper operation.

    :param crystal: The crystal whose space group is tested
    :returns: True iff every symmetry operation is proper
    """
    return all(
        np.linalg.det(np.asarray(op.rot, dtype=float)) > 0
        for op in crystal.space_group.sym_ops
    )


def bind(
    crystal: Crystal,
    monomers: list[Monomer],
    site_atol: float = SITE_ATOL,
    full_congruence_check: bool = False,
) -> None:
    """Bind the deduper's per-run inputs at module scope.

    Sets the per-reference Z and mass arrays, self-isometries, site-symmetry
    dressings and congruence gates, plus ``frac_to_cart`` / ``cart_to_frac``.

    :param crystal: The crystal supplying the lattice and Cartesian conversion
    :param monomers: The reference monomers (one per crystal asymmetric unit)
    :param site_atol: Fractional tolerance for the site-symmetry test; must
        match the one :func:`generate_neighbors` passes to
        ``get_unique_sym_ops`` or stabilizer and orbit disagree
    :param full_congruence_check: Always run the Cartesian congruence branch,
        not just in the Sohncke-enantiomer case. Slower; an escape hatch for
        exotic crystals, but it can't recover an undetected dressing
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

        # Seed S with W: `W_lin <= S` holds in theory, but the missing ops are
        # rotoreflections the detector's candidate library can't produce.
        # W was found at the mass-weighted centroid and S at the unweighted
        # one; a true symmetry fixes their offset, so the matrices agree.
        W_cart = [frac_to_cart @ w.astype(float) @ cart_to_frac for w in W]
        S = mon.point_group_seeded(W_cart)
        self_iso_by_ref.append(S)
        # Cartesian dressing translations: d = c - s @ c (zero iff s fixes c).
        c_cart = np.asarray(mon.centroid_cart, dtype=float)
        self_iso_tr_by_ref.append([c_cart - s @ c_cart for s in S])

        # Gate for the tolerant Cartesian branch. Needed only for a Sohncke
        # group with an achiral monomer: an enantiomeric merge then forces an
        # improper dressing the all-proper W can't supply (thm:sohncke).
        # Elsewhere it costs 32-44% of trimer runtime for no merges. Linear
        # monomers no longer open it (out of scope; see generate).
        # `len(S) > len(W)` stays inside the narrow arm: S under-detects, so
        # guarding the escape hatch with it would silently disable the hatch.
        improper_S = any(float(np.linalg.det(s)) < 0 for s in S)
        check_congruence_by_ref.append(
            full_congruence_check or (len(S) > len(W) and sohncke and improper_S)
        )


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
    :param full_congruence_check: The gate passed to :func:`bind`; must match
        the main process
    :returns: ``None``
    """
    bind(crystal, monomers, site_atol, full_congruence_check)
