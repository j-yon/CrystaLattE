from dataclasses import dataclass
from itertools import islice, product
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from tqdm import tqdm

from ..core.crystal import Crystal
from ..core.multimer import Monomer
from ..core.space_group import SITE_ATOL
from ..core.sym_ops import SymOp, SymOpList
from .dedup import context
from .dedup.equivalence import representatives
from .dedup.invariants import chunk_keys, labels_from_keys, pair_codes, relative_rows


def _distance_mode(kwargs: dict) -> str:
    """Resolve the desired distance convention

    :param kwargs: user keyword arguments
    :returns: ``"origin"``, ``"com"`` or ``"contact"``
    """
    # closest partner atom to anchor centroid
    if kwargs.get("use_origin"):
        return "origin"
    # closest atom-pair contact
    if kwargs.get("use_com", False):
        return "com"
    # default: closest atom-pair contact
    return "contact"


def _neighbor_distance(
    cart_mon: NDArray, partner_masses: NDArray, monomer: Monomer, mode: str
) -> float:
    """Return the cutoff distance for a partner monomer

    :param cart_mon: partner monomer's Cartesian coordinates
    :param partner_masses: partner's atomic masses, aligned with ``cart_mon``.
        (these must come from the partner's own base reference for ``Z' > 1``)
    :param monomer: anchor monomer
    :param mode: desired distance convention
    :returns: partner-to-anchor distance in Angstroms
    """
    if mode == "origin":
        return float(np.linalg.norm(cart_mon - monomer.centroid_cart, axis=1).min())
    if mode == "com":
        com_mon = np.average(cart_mon, axis=0, weights=partner_masses)
        return float(np.linalg.norm(com_mon - monomer.centroid_cart))
    return float(cdist(monomer.cart_coords, cart_mon).min())


def _partner_adjacency(
    carts: list[NDArray], masses: list[NDArray], R: float, mode: str
) -> NDArray:
    """Return which pairs of partners lie within ``R`` of each other under the distance convention

    The origin convention is anchor-centred, so for a partner pair it takes the smaller of the two
    closest-atom-to-centroid distances.

    :param carts: partner Cartesian coordinates
    :param masses: partner atomic masses, aligned with ``carts``
    :param R: distance cutoff in Angstroms
    :param mode: desired distance convention
    :returns: a symmetric ``(k, k)`` boolean matrix
    """
    coms = np.array([np.average(c, axis=0, weights=m) for c, m in zip(carts, masses)])
    D = cdist(coms, coms)
    if mode == "com":
        return D <= R

    # any pair beyond R + both radii is out of range; only the rest need an exact distance
    radii = np.array(
        [np.linalg.norm(c - com, axis=1).max() for c, com in zip(carts, coms)]
    )
    adj = np.zeros_like(D, dtype=bool)
    for p, q in zip(
        *np.nonzero(np.triu(D <= R + radii[:, None] + radii[None, :], k=1))
    ):
        if mode == "origin":
            d = min(
                np.linalg.norm(carts[q] - coms[p], axis=1).min(),
                np.linalg.norm(carts[p] - coms[q], axis=1).min(),
            )
        else:
            d = cdist(carts[p], carts[q]).min()
        adj[p, q] = adj[q, p] = d <= R
    return adj


def _cliques(adj: NDArray | None, n: int, size: int):
    """Yield sorted index tuples of ``size`` partners that are pairwise adjacent

    :param adj: partner adjacency matrix, or ``None`` when ``size == 1``
    :param n: number of partners
    :param size: number of partners per combination
    :returns: generator of index tuples
    """

    def extend(chosen: tuple, cands: NDArray):
        if len(chosen) == size:
            yield chosen
            return
        for x, k in enumerate(cands):
            rest = cands[x + 1 :]
            yield from extend(
                (*chosen, int(k)), rest[adj[k, rest]] if adj is not None else rest
            )

    yield from extend((), np.arange(n))


#: (cluster, anchor, dressing) key rows built per batch of clusters; bounds the keying intermediates to ~100 MB
KEY_CHUNK_ROWS: Final[int] = 1 << 20


@dataclass
class Candidates:
    """Every candidate cluster, stored as member indices into per-anchor partner tables

    Table ``i`` holds reference ``i``'s anchor (row 0, the identity) and each partner within ``R`` of it, and
    ``members[i]`` holds one row of table indices per cluster, the anchor first. Clusters are numbered across
    the tables in order, and only the ones asked for become :class:`SymOpList` objects.

    :ivar rot: per-table ``(P, 3, 3)`` int16 member rotations
    :ivar tr: per-table ``(P, 3)`` int16 member translations, in twelfths
    :ivar ref: per-table ``(P,)`` int64 member base references
    :ivar members: per-table ``(C, N)`` int32 member rows
    :ivar group: ``(M,)`` group code of each cluster (:func:`~.dedup.invariants.chunk_keys`)
    :ivar keys: ``(M, N - 1)`` orbit key of each cluster
    :ivar tnorm: ``(M,)`` summed translation norm of each cluster's members
    """

    rot: list[NDArray]
    tr: list[NDArray]
    ref: list[NDArray]
    members: list[NDArray]
    group: NDArray
    keys: NDArray
    tnorm: NDArray

    def __post_init__(self):
        self._offsets = np.cumsum([0] + [len(m) for m in self.members])

    def __len__(self) -> int:
        return len(self.group)

    def cluster(self, k: int) -> SymOpList:
        """Return cluster ``k`` as a :class:`SymOpList` of multiplicity 1

        :param k: cluster index across all tables
        :returns: the cluster's members, its anchor first
        """
        return self.clusters(np.array([k]), np.ones(1, dtype=np.int64))[0]

    def clusters(self, ks: NDArray, multiplicities: NDArray) -> list[SymOpList]:
        """Return clusters ``ks`` as :class:`SymOpList` objects, gathering each table's members at once

        :param ks: ``(K,)`` cluster indices across all tables
        :param multiplicities: ``(K,)`` multiplicity to give each
        :returns: the clusters in the order of ``ks``, each with its anchor first
        """
        rot, tr, ref = self.arrays(ks)
        return [
            SymOpList.from_components(rot[n], tr[n], int(m), ref[n])
            for n, m in enumerate(multiplicities.tolist())
        ]

    def arrays(self, ks: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        """Return the stacked member arrays of clusters ``ks``, gathering each table's members at once

        :param ks: ``(K,)`` cluster indices across all tables
        :returns: ``(rot, tr, ref)`` of shapes ``(K, N, 3, 3)``, ``(K, N, 3)`` and ``(K, N)``, each cluster's
            anchor first
        """
        N = self.members[0].shape[1]
        rot = np.empty((len(ks), N, 3, 3), dtype=self.rot[0].dtype)
        tr = np.empty((len(ks), N, 3), dtype=self.tr[0].dtype)
        ref = np.empty((len(ks), N), dtype=np.int64)
        table = np.searchsorted(self._offsets, ks, side="right") - 1
        for i in np.unique(table).tolist():
            at = np.flatnonzero(table == i)
            rows = self.members[i][ks[at] - self._offsets[i]]
            rot[at], tr[at], ref[at] = self.rot[i][rows], self.tr[i][rows], self.ref[i][rows]
        return rot, tr, ref


def enumerate_candidates(
    crystal: Crystal,
    monomers: list[Monomer],
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> Candidates:
    """Enumerate every candidate cluster in the supercell defined by ``cutoff`` and key it for deduplication

    Each anchor's partners form a table, and every ordered pair of mutually close members gets the dressed codes
    of its relative transform once (:func:`~.dedup.invariants.pair_codes`), over all tables together so their
    codes compare. Clusters are then enumerated in batches and keyed by looking up their member pairs
    (:func:`~.dedup.invariants.chunk_keys`), so per cluster only its member indices, key and translation norm
    are kept.

    :param crystal: :class:`Crystal` containing the monomers
    :param monomers: Reference :class:`Monomer` instance(s) used to anchor the N-mer search
    :param cutoff: Per-axis ``(low, high)`` integer lattice translation bounds
    :param R: Distance cutoff in Angstroms, applied to every pair of members. The distance convention is selected by ``kwargs``
    :param N: N-mer order
    :param kwargs: Additional options (``verbose``, ``site_atol``, etc.)
    :returns: the keyed :class:`Candidates`
    """
    verbose = kwargs.get("verbose", 1)
    mode = _distance_mode(kwargs)
    site_atol = kwargs.get("site_atol", SITE_ATOL)

    # generate all integer translations within the cutoff box
    translations = list(product(*[range(cm, cp + 1) for cm, cp in cutoff]))

    # bind per-reference state
    context.bind(crystal, monomers, site_atol)

    # full partner pool across ALL references
    pool = []
    for j, mon in enumerate(monomers):
        sym_ops = crystal.space_group.get_unique_sym_ops(
            mon.centroid_frac, atol=site_atol
        )
        for op, tr in product(sym_ops, translations):
            pool.append((j, op.translate(np.array(tr, dtype=np.int16) * 12)))

    # anchor each reference i: its partner table and the ordered member pairs a cluster can contain
    identity = SymOp.identity()  # parse once; SymOps are never mutated, so members can share it
    rot, tr, ref, adjs, eids, pairs = [], [], [], [], [], []
    n_edges = 0
    with tqdm(
        total=len(monomers) * len(pool),
        desc="Generating candidate operations",
        leave=False,
        disable=not verbose,
    ) as pbar:
        for i, anchor in enumerate(monomers):
            near, carts = [], []
            for j, op in pool:
                pbar.update()
                if j == i and op == identity:
                    continue  # skip self-identity

                # filter partners by distance to anchor i
                partner_cart = crystal.to_cartesian(op.apply(monomers[j].frac_coords))
                if (_neighbor_distance(partner_cart, context.mass_by_ref[j], anchor, mode) <= R):
                    near.append((j, op))
                    carts.append(partner_cart)

            if verbose:
                tqdm.write(f"Reference {i}: {len(near)} monomers within {R} Angstroms.")

            rot.append(np.stack([identity.rot] + [op.rot for _, op in near]))
            tr.append(np.stack([identity.tr] + [op.tr for _, op in near]))
            ref.append(np.array([i] + [j for j, _ in near], dtype=np.int64))
            P = len(near) + 1

            # every pair of members must lie within R, so partners must also be within R of each other
            adj = None
            close = np.zeros((P, P), dtype=bool)
            close[0, 1:] = close[1:, 0] = True
            if N > 2 and near:
                adj = _partner_adjacency(
                    carts, [context.mass_by_ref[j] for j, _ in near], R, mode
                )
                close[1:, 1:] = adj
            p, q = np.nonzero(close)
            eid = np.full((P, P), -1, dtype=np.int32)
            eid[p, q] = n_edges + np.arange(len(p))
            n_edges += len(p)
            adjs.append(adj)
            eids.append(eid)
            pairs.append((relative_rows(rot[i], tr[i], p, q), ref[i][p], ref[i][q]))

    codes = pair_codes(
        np.concatenate([r for r, _, _ in pairs]),
        np.concatenate([a for _, a, _ in pairs]),
        np.concatenate([b for _, _, b in pairs]),
    )

    # enumerate and key the clusters of each anchor in batches
    S_max = codes.shape[1]
    batch = max(1, KEY_CHUNK_ROWS // (N * S_max))
    members, groups, keys, tnorms = [], [], [], []
    for i in range(len(monomers)):
        tnorm = np.linalg.norm(tr[i], axis=1)
        cliques = _cliques(adjs[i], len(ref[i]) - 1, N - 1)
        rows = []
        with tqdm(
            desc=f"Enumerating clusters for reference {i}",
            unit=" clusters",
            position=1,
            leave=False,
            disable=not verbose,
        ) as bar:
            while chunk := list(islice(cliques, batch)):
                mem = np.empty((len(chunk), N), dtype=np.int32)
                mem[:, 0] = 0
                mem[:, 1:] = np.array(chunk, dtype=np.int32) + 1
                g, k = chunk_keys(codes, eids[i][mem[:, :, None], mem[:, None, :]], ref[i][mem])
                rows.append(mem)
                groups.append(g)
                keys.append(k)
                tnorms.append(tnorm[mem].sum(axis=1))
                bar.update(len(chunk))
        members.append(np.concatenate(rows) if rows else np.empty((0, N), dtype=np.int32))

    return Candidates(
        rot,
        tr,
        ref,
        members,
        np.concatenate(groups) if groups else np.empty(0, dtype=np.int64),
        np.concatenate(keys) if keys else np.empty((0, N - 1), dtype=np.int32),
        np.concatenate(tnorms) if tnorms else np.empty(0),
    )


def generate_neighbors(
    crystal: Crystal,
    monomers: list[Monomer],
    cutoff: NDArray,
    R: float,
    N: int,
    **kwargs,
) -> list[SymOpList]:
    """Generate and deduplicate all unique multimers in the supercell defined by ``cutoff``.
    Multimers are not actually constructed, only their symmetry operations are considered

    :param crystal: :class:`Crystal` containing the monomers
    :param monomers: Reference :class:`Monomer` instance(s) used to anchor the N-mer search
    :param cutoff: Per-axis ``(low, high)`` integer lattice translation bounds
    :param R: Distance cutoff in Angstroms, applied to every pair of members. The distance convention is selected by ``kwargs``
    :param N: N-mer order
    :param kwargs: Additional options (``verbose``, ``site_atol``, etc.)
    :returns: Unique clusters as a list of :class:`SymOpList` instances
    """
    candidates = enumerate_candidates(crystal, monomers, cutoff, R, N, **kwargs)
    if not len(candidates):
        return []

    # exact pass: clusters sharing an orbit label are the same multimer
    labels = labels_from_keys(candidates.group, candidates.keys, candidates.arrays)
    rep, multiplicity = representatives(labels, candidates.tnorm)

    if kwargs.get("verbose", 1):
        print(
            f"Found {len(candidates) - len(rep)} equivalent multimers from neighboring unit cell search."
        )

    return candidates.clusters(rep, multiplicity)
