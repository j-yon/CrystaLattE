"""Enumerate the ordered states a disordered CIF block admits

Two kinds of disorder are handled:

* explicit groups (``_atom_site_disorder_assembly`` / ``_atom_site_disorder_group``): one group
  is chosen per assembly, and the declared space group still holds
* symmetry-generated disorder (untagged or negative-group partial sites): a subset of the partial
  atoms of the full unit cell is chosen, and the result has no symmetry until
  :meth:`~crystalatte.core.crystal.Crystal.detect_symmetry` recovers it
"""

import itertools
from collections import Counter
from typing import NamedTuple

import numpy as np
import qcelemental as qcel
from numpy.typing import NDArray

from ..core.crystal import expand_cell
from ..core.sym_ops import SymOp

# occupancy at or above this counts as fully occupied
FULL_OCCUPANCY = 0.999

# an assembly whose group weights sum below this also admits "no group present"
_ABSENT_BELOW = 0.98


def normalize_tag(value) -> str | None:
    """Normalise a disorder assembly or group value

    :param value: Raw value from the cleaned block (numbers arrive as floats)
    :returns: ``None`` for an untagged site, integral numbers without a decimal point, else the text
    """
    if value is None:
        return None
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    return None if text in ("", ".", "?") else text


class ExplicitChoice(NamedTuple):
    """One group chosen per explicit disorder assembly

    :param keep: Mask of the ASU sites present in this choice
    :param settled: Mask of the ASU sites whose occupancy this choice resolved
    :param groups: ``(assembly, group)`` pairs; ``group`` is ``"-"`` when the assembly is absent
    :param label: e.g. ``"A1 B2"``; empty when the block has no explicit groups
    :param weight: Product of the chosen groups' occupancies
    """

    keep: NDArray[np.bool_]
    settled: NDArray[np.bool_]
    groups: tuple[tuple[str, str], ...]
    label: str
    weight: float


def _group_order(group: str) -> tuple:
    return (0, int(group), "") if group.isdigit() else (1, 0, group)


def explicit_choices(
    occupancies, assemblies: list[str | None], groups: list[str | None]
) -> list[ExplicitChoice]:
    """Enumerate one group per explicit disorder assembly, taking assemblies as independent

    A site takes part when it has a positive group and is partially occupied. A tagged but
    fully occupied site is kept in every choice; negative groups and untagged partial sites
    are left untouched for the symmetry stage.

    :param occupancies: Per-site occupancies
    :param assemblies: Per-site normalised assembly names (``None`` when untagged or absent)
    :param groups: Per-site normalised group names (``None`` when untagged)
    :returns: Every combination of choices, in assembly then group order
    """
    occupancies = np.asarray(occupancies, dtype=float)
    n = len(occupancies)
    selectable = np.array(
        [
            g is not None and not g.startswith("-") and o < FULL_OCCUPANCY
            for g, o in zip(groups, occupancies)
        ],
        dtype=bool,
    )
    if not selectable.any():
        return [ExplicitChoice(np.ones(n, dtype=bool), np.zeros(n, dtype=bool), (), "", 1.0)]

    # options for each assembly: its groups and their weights
    assembly_of = [a or "" for a in assemblies]
    members = np.flatnonzero(selectable)
    options: dict[str, list[tuple[str, float]]] = {}
    for name in sorted({assembly_of[i] for i in members}):
        mine = [i for i in members if assembly_of[i] == name]
        opts = [
            (g, float(np.mean([occupancies[i] for i in mine if groups[i] == g])))
            for g in sorted({groups[i] for i in mine}, key=_group_order)
        ]
        total = sum(w for _, w in opts)
        if total < _ABSENT_BELOW:
            opts.append(("-", 1.0 - total))
        options[name] = opts

    choices = []
    for combo in itertools.product(*options.values()):
        picked = dict(zip(options, (g for g, _ in combo)))
        keep = np.array(
            [not selectable[i] or picked[assembly_of[i]] == groups[i] for i in range(n)],
            dtype=bool,
        )
        pairs = tuple(picked.items())
        label = " ".join(f"{a}{g}" for a, g in pairs)
        weight = float(np.prod([w for _, w in combo]))
        choices.append(ExplicitChoice(keep, selectable.copy(), pairs, label, weight))
    return choices


_BOHR_TO_ANGSTROM = qcel.constants.bohr2angstroms


class SymmetryChoice(NamedTuple):
    """One ordered state of symmetry-generated disorder, as the atoms of the full unit cell

    :param frac: Fractional coordinates of the atoms present, in [0, 1)
    :param symbols: Their atomic symbols
    :param label: ``"config k"``, numbered by decreasing weight
    :param weight: Share of all valid configurations of this cell that are symmetry copies of this one
    """

    frac: NDArray[np.float64]
    symbols: NDArray[np.str_]
    label: str
    weight: float


def _periodic_distances(frac: NDArray[np.float64], lattice: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum-image distances in Angstroms between every pair of atoms of the cell"""
    delta = frac[:, None, :] - frac[None, :, :]
    delta -= np.round(delta)
    best = np.full(delta.shape[:2], np.inf)
    for shift in itertools.product((-1, 0, 1), repeat=3):
        best = np.minimum(best, np.linalg.norm((delta + shift) @ lattice.T, axis=2))
    return best


def contacts(symbols, distances: NDArray[np.float64], tol: float = 1.2):
    """Classify atom pairs as bonded or clashing

    Bonded follows :meth:`~crystalatte.core.crystal.Crystal._bfs`, ``d <= tol * (r_cov,i + r_cov,j)``,
    except that two hydrogens are never bonded: the two H sites on one O...O contact in ice sit
    0.76 A apart, a hair above the 0.74 A H-H cutoff. A clash is a non-bonded pair closer than half
    the sum of their van der Waals radii.

    :param symbols: Atomic symbols, shape ``(n,)``
    :param distances: Pairwise distances in Angstroms, shape ``(n, n)``
    :param tol: Bonding tolerance factor
    :returns: ``(bonded, clash)`` boolean matrices
    """
    symbols = np.asarray(symbols)
    r_cov = np.array([qcel.covalentradii.get(s) for s in symbols]) * _BOHR_TO_ANGSTROM
    r_vdw = np.array([qcel.vdwradii.get(s) for s in symbols]) * _BOHR_TO_ANGSTROM
    hydrogen = symbols == "H"
    bonded = distances <= tol * (r_cov[:, None] + r_cov[None, :])
    bonded &= ~(hydrogen[:, None] & hydrogen[None, :])
    np.fill_diagonal(bonded, False)
    clash = ~bonded & (distances < 0.5 * (r_vdw[:, None] + r_vdw[None, :]))
    np.fill_diagonal(clash, False)
    return bonded, clash


def _components(bonded: NDArray[np.bool_], members) -> list[list[int]]:
    """Connected components of ``members`` in the bond graph (union-find)"""
    members = [int(a) for a in members]
    index = {a: k for k, a in enumerate(members)}
    parent = list(range(len(members)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in members:
        for b in np.flatnonzero(bonded[a]):
            if int(b) in index:
                parent[find(index[a])] = find(index[int(b)])
    groups: dict[int, list[int]] = {}
    for a in members:
        groups.setdefault(find(index[a]), []).append(a)
    return list(groups.values())


def _site_permutations(frac: NDArray[np.float64], sym_ops: list[SymOp]) -> list[NDArray[np.intp]]:
    """For each operation, the index of the cell atom that every cell atom maps onto"""
    perms = []
    for op in sym_ops:
        images = np.array([op.apply(f) for f in frac]) % 1.0
        delta = images[:, None, :] - frac[None, :, :]
        delta -= np.round(delta)
        perms.append(np.argmin(np.linalg.norm(delta, axis=2), axis=1))
    return perms


# a state's bonds and angles may depart from the asymmetric unit as written by at most this much
_BOND_TOL = 0.15
_ANGLE_TOL = 10.0


def _angle(d_ac: float, d_bc: float, d_ab: float) -> float:
    """Angle at c in degrees, from the three side lengths"""
    cos = (d_ac**2 + d_bc**2 - d_ab**2) / (2 * d_ac * d_bc)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class _Geometry:
    """Bond graph of the full unit cell, judged against the asymmetric unit as written

    The asymmetric unit of a refined structure lists one consistent alternative, so its bond
    lengths and angles are the ones every ordered state must reproduce.

    :ivar frac: Fractional coordinates of every cell atom
    :ivar symbols: Their atomic symbols
    :ivar site: The ASU site each came from
    :ivar partial: Whether each is partially occupied
    :ivar bonded: Bond matrix
    :ivar conflict: Pairs (with a partial atom) that can never both be present
    """

    def __init__(self, positions, symbols, partial_site, sym_ops: list[SymOp], lattice):
        positions = np.asarray(positions, dtype=float)
        self.frac, self.symbols, self.site = expand_cell(positions, symbols, sym_ops)
        self.partial = np.asarray(partial_site, dtype=bool)[self.site]
        self.dist = _periodic_distances(self.frac, lattice)
        self.bonded, clash = contacts(self.symbols, self.dist)

        # bond lengths and angles of the asymmetric unit as written
        ref_dist = _periodic_distances(positions % 1.0, lattice)
        ref_bonded, _ = contacts(symbols, ref_dist)
        self.lengths = {(int(i), int(j)): ref_dist[i, j] for i, j in zip(*np.nonzero(ref_bonded))}
        self.angles = {
            (int(a), c, int(b)): _angle(ref_dist[a, c], ref_dist[b, c], ref_dist[a, b])
            for c in range(len(positions))
            for a, b in itertools.permutations(np.flatnonzero(ref_bonded[c]), 2)
        }

        # which operations generate each cell atom; two bonded partial atoms must share one
        generated = np.zeros((len(sym_ops), len(self.frac)), dtype=bool)
        for k, op in enumerate(sym_ops):
            images = np.array([op.apply(p) for p in positions]) % 1.0
            delta = images[:, None, :] - self.frac[None, :, :]
            delta -= np.round(delta)
            generated[k, np.argmin(np.linalg.norm(delta, axis=2), axis=1)] = True
        shared_op = (generated.T.astype(int) @ generated.astype(int)) > 0

        either = self.partial[:, None] | self.partial[None, :]
        both = self.partial[:, None] & self.partial[None, :]
        wrong_length = np.zeros_like(self.bonded)
        for a, b in zip(*np.nonzero(self.bonded & either)):
            ref = self.lengths.get((int(self.site[a]), int(self.site[b])))
            if ref is not None and abs(self.dist[a, b] - ref) > _BOND_TOL:
                wrong_length[a, b] = True
        self.conflict = either & (clash | wrong_length | (self.bonded & both & ~shared_op))

    def bad_angles(self, present) -> bool:
        """Whether an angle among ``present`` atoms that involves a partial atom departs from the ASU"""
        inside = np.zeros(len(self.frac), dtype=bool)
        inside[list(present)] = True
        for c in present:
            for a, b in itertools.combinations(np.flatnonzero(self.bonded[c] & inside), 2):
                if not (self.partial[a] or self.partial[b] or self.partial[c]):
                    continue
                ref = self.angles.get((int(self.site[a]), int(self.site[c]), int(self.site[b])))
                d = self.dist
                if ref is not None and abs(_angle(d[a, c], d[b, c], d[a, b]) - ref) > _ANGLE_TOL:
                    return True
        return False


def overlapping_assemblies(
    positions, symbols, occupancies, assemblies: list[str | None], groups: list[str | None],
    sym_ops: list[SymOp], lattice,
) -> set[str]:
    """Explicit assemblies with a group whose atoms collide with their own symmetry images

    Such a group is disorder across a symmetry element written with a positive group number, so
    choosing it as a whole (and expanding it by the full group) cannot give an ordered state.

    :returns: The names of those assemblies (``""`` for the unnamed one)
    """
    occupancies = np.asarray(occupancies, dtype=float)
    full = np.flatnonzero(occupancies >= FULL_OCCUPANCY)
    keys = {
        (a or "", g) for a, g, o in zip(assemblies, groups, occupancies)
        if g is not None and not g.startswith("-") and o < FULL_OCCUPANCY
    }
    overlapping = set()
    for assembly, group in sorted(keys):
        members = [
            i for i, (a, g) in enumerate(zip(assemblies, groups))
            if (a or "") == assembly and g == group and occupancies[i] < FULL_OCCUPANCY
        ]
        sub = np.concatenate([full, members]).astype(int)
        geometry = _Geometry(
            np.asarray(positions)[sub], [symbols[i] for i in sub],
            np.isin(sub, members), sym_ops, lattice,
        )
        present = range(len(geometry.frac))
        if geometry.conflict.any() or geometry.bad_angles(present):
            overlapping.add(assembly)
    return overlapping


def symmetry_choices(
    positions, symbols, occupancies, labels: list[str], sym_ops: list[SymOp], lattice
) -> list[SymmetryChoice]:
    """Enumerate the ordered states of partial sites whose alternatives are symmetry images

    A configuration chooses ``occupancy x images`` atoms of every partial site in the cell such
    that no two chosen atoms clash, bonds and angles keep the values of the asymmetric unit as
    written, bonded partial atoms come from the same symmetry copy, and every molecule (the
    fully occupied atoms that the partial sites connect) gets the same formula as its symmetry
    copies. For water these are the Bernal-Fowler ice rules. Configurations related by an
    operation of the declared group are one state.

    :param positions: Fractional coordinates of the ASU sites
    :param symbols: Their atomic symbols
    :param occupancies: Their occupancies; sites at or above :data:`FULL_OCCUPANCY` are always present
    :param labels: Site labels, for error messages
    :param sym_ops: Operations of the declared space group
    :param lattice: Fractional-to-Cartesian matrix
    :returns: The states, heaviest first
    :raises ValueError: If no ordered state of this cell exists, alternatives are written side by
        side without disorder groups, or a partial atom belongs to no fully occupied molecule
    """
    occupancies = np.asarray(occupancies, dtype=float)
    partial_site = occupancies < FULL_OCCUPANCY
    geo = _Geometry(positions, symbols, partial_site, sym_ops, lattice)
    frac, cell_symbols, site = geo.frac, geo.symbols, geo.site
    images = np.bincount(site, minlength=len(occupancies))

    # each partial site must put a whole number of atoms into the cell
    wanted: dict[int, int] = {}
    for i in np.flatnonzero(partial_site):
        count = occupancies[i] * images[i]
        if abs(count - round(count)) > 0.05:
            raise ValueError(
                f"site {labels[i]} (occupancy {occupancies[i]:g}, {images[i]} image(s) in the cell) "
                "does not contribute a whole number of atoms, so this cell has no ordered state"
            )
        wanted[int(i)] = int(round(count))

    # alternatives written side by side cannot be told apart without disorder groups
    _, asu_clash = contacts(symbols, _periodic_distances(np.asarray(positions, dtype=float) % 1.0, lattice))
    for i, j in zip(*np.nonzero(np.triu(asu_clash))):
        if partial_site[i] and partial_site[j]:
            raise ValueError(
                f"sites {labels[i]} and {labels[j]} are alternative positions written side by side in "
                "the asymmetric unit; give them a disorder group so they can be told apart"
            )

    # molecules: the fully occupied atoms that the partial atoms connect
    molecules = _components(geo.bonded, range(len(frac)))
    for comp in molecules:
        if geo.partial[comp].all():
            names = sorted({labels[site[a]] for a in comp})
            raise ValueError(
                f"site(s) {', '.join(names)} are not bonded to any fully occupied atom; disorder of "
                "a whole molecule over symmetry images is not supported"
            )

    perms = _site_permutations(frac, sym_ops)
    molecule_of = {a: k for k, comp in enumerate(molecules) for a in comp}
    orbit_of = [min(molecule_of[int(p[comp[0]])] for p in perms) for comp in molecules]
    orbit_size = Counter(orbit_of)
    full = np.flatnonzero(~geo.partial)
    blocked = geo.conflict[:, full].any(axis=1)
    partial_names = ", ".join(labels[i] for i in sorted(wanted))

    # every molecule of an orbit takes the same share of each element
    need: dict[int, Counter] = {}
    for i, count in wanted.items():
        orbit = orbit_of[molecule_of[int(np.flatnonzero(site == i)[0])]]
        need.setdefault(orbit, Counter())[str(symbols[i])] += count
    for orbit, counts in need.items():
        for element, count in counts.items():
            if count % orbit_size[orbit]:
                raise ValueError(
                    f"partial sites ({partial_names}) put {count} {element} atom(s) on "
                    f"{orbit_size[orbit]} equivalent molecules, so they cannot all get the same formula"
                )
            counts[element] = count // orbit_size[orbit]

    def options(k: int) -> list[list[int]]:
        """Every valid way to give molecule ``k`` its share of partial atoms"""
        comp = molecules[k]
        own_full = [a for a in comp if not geo.partial[a]]
        candidates = sorted(a for a in comp if geo.partial[a] and not blocked[a])
        target = need.get(orbit_of[k], Counter())
        found: list[list[int]] = []

        def grow(idx: int, chosen: list[int], missing: Counter) -> None:
            if not +missing:
                present = own_full + chosen
                if not geo.bad_angles(present) and len(_components(geo.bonded, present)) == 1:
                    found.append(chosen)
                return
            if idx == len(candidates):
                return
            a = candidates[idx]
            element = str(cell_symbols[a])
            if missing[element] > 0 and not geo.conflict[a, chosen].any():
                grow(idx + 1, [*chosen, a], missing - Counter({element: 1}))
            grow(idx + 1, chosen, missing)

        grow(0, [], Counter(target))
        return found

    choices_per_molecule = [options(k) for k in range(len(molecules))]
    if not all(choices_per_molecule):
        raise ValueError(
            f"no ordered state of this cell: partial sites ({partial_names}) cannot give every "
            "molecule its share without clashes or distorted bonds and angles"
        )

    # how many atoms of each site a molecule can still contribute, to prune the search
    n_sites = len(occupancies)
    contribution = [
        [np.bincount(site[opt], minlength=n_sites) for opt in opts] for opts in choices_per_molecule
    ]
    reachable = np.zeros((len(molecules) + 1, n_sites), dtype=int)
    for k in range(len(molecules) - 1, -1, -1):
        reachable[k] = reachable[k + 1] + np.max(contribution[k], axis=0)
    target_counts = np.zeros(n_sites, dtype=int)
    for i, count in wanted.items():
        target_counts[i] = count

    valid: list[tuple[int, ...]] = []

    def assign(k: int, chosen: list[int], counts: NDArray[np.int_]) -> None:
        if np.any(counts > target_counts) or np.any(counts + reachable[k] < target_counts):
            return
        if k == len(molecules):
            valid.append(tuple(sorted(chosen)))
            return
        for opt, add in zip(choices_per_molecule[k], contribution[k]):
            if opt and chosen and geo.conflict[np.ix_(opt, chosen)].any():
                continue
            assign(k + 1, [*chosen, *opt], counts + add)

    assign(0, [], np.zeros(n_sites, dtype=int))
    if not valid:
        raise ValueError(
            f"no ordered state of this cell: partial sites ({partial_names}) cannot give every "
            "molecule its share with the occupancies of the CIF"
        )

    # configurations related by a declared operation are the same state
    seen: set[tuple[int, ...]] = set()
    states: list[tuple[tuple[int, ...], int]] = []
    for selection in valid:
        if selection in seen:
            continue
        orbit = {tuple(sorted(int(p[a]) for a in selection)) for p in perms}
        seen |= orbit
        states.append((min(orbit), len(orbit)))
    states.sort(key=lambda s: (-s[1], s[0]))

    choices = []
    for k, (selection, count) in enumerate(states, start=1):
        present = np.concatenate([full, np.array(selection, dtype=np.intp)])
        choices.append(
            SymmetryChoice(frac[present], cell_symbols[present], f"config {k}", count / len(valid))
        )
    return choices
