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


def symmetry_choices(
    positions, symbols, occupancies, labels: list[str], sym_ops: list[SymOp], lattice
) -> list[SymmetryChoice]:
    """Enumerate the ordered states of partial sites whose alternatives are symmetry images

    A configuration chooses ``occupancy x images`` atoms of every partial site in the cell such
    that no two chosen atoms clash, every molecule holds exactly one skeleton (connected
    component of fully occupied atoms), and symmetry-equivalent skeletons end up with the same
    formula. For water these are the Bernal-Fowler ice rules. Configurations related by an
    operation of the declared group are one state.

    :param positions: Fractional coordinates of the ASU sites
    :param symbols: Their atomic symbols
    :param occupancies: Their occupancies; sites at or above :data:`FULL_OCCUPANCY` are always present
    :param labels: Site labels, for error messages
    :param sym_ops: Operations of the declared space group
    :param lattice: Fractional-to-Cartesian matrix
    :returns: The states, heaviest first
    :raises ValueError: If no ordered state of this cell exists, or a partial atom has no skeleton
    """
    frac, cell_symbols, site = expand_cell(positions, symbols, sym_ops)
    occupancies = np.asarray(occupancies, dtype=float)
    partial_site = occupancies < FULL_OCCUPANCY
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

    full = np.flatnonzero(~partial_site[site])
    partial = np.flatnonzero(partial_site[site])
    bonded, clash = contacts(cell_symbols, _periodic_distances(frac, lattice))

    # every partial atom must hang off a fully occupied skeleton
    for comp in _components(bonded, range(len(frac))):
        if all(partial_site[site[a]] for a in comp):
            names = sorted({labels[site[a]] for a in comp})
            raise ValueError(
                f"site(s) {', '.join(names)} are not bonded to any fully occupied atom; disorder of "
                "a whole molecule over symmetry images is not supported"
            )

    perms = _site_permutations(frac, sym_ops)
    skeletons = _components(bonded, full)
    skeleton_of = {a: k for k, comp in enumerate(skeletons) for a in comp}
    orbit_of = {k: min(skeleton_of[int(p[comp[0]])] for p in perms) for k, comp in enumerate(skeletons)}
    candidates = {i: [int(a) for a in partial if site[a] == i] for i in wanted}
    order = sorted(wanted)
    valid: list[tuple[int, ...]] = []

    def is_valid(chosen: list[int]) -> bool:
        formulas: dict[int, tuple] = {}
        for comp in _components(bonded, [*full, *chosen]):
            owners = {skeleton_of[a] for a in comp if a in skeleton_of}
            if len(owners) != 1:
                return False
            formula = tuple(sorted(Counter(cell_symbols[comp].tolist()).items()))
            if formulas.setdefault(orbit_of[owners.pop()], formula) != formula:
                return False
        return True

    def extend(k: int, chosen: list[int]) -> None:
        if k == len(order):
            if is_valid(chosen):
                valid.append(tuple(sorted(chosen)))
            return
        for combo in itertools.combinations(candidates[order[k]], wanted[order[k]]):
            if any(clash[a, b] for a in combo for b in (*full, *chosen, *combo)):
                continue
            extend(k + 1, [*chosen, *combo])

    extend(0, [])
    if not valid:
        raise ValueError("no ordered state of this cell satisfies the occupancies without clashes")

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
