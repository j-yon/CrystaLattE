# Disorder → Ordered States Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `from_cif` yields one `Crystal` per ordered state of a disordered CIF block (by default, with a warning), each carrying a `DisorderState` (label + weight).

**Architecture:** A new pure module `io/disorder.py` enumerates states from per-site arrays in two stages. Explicit stage: one group per `_atom_site_disorder_assembly`, keeping the declared space group. Symmetry stage: for partial sites left untagged (or with negative groups), choose atoms of the full P1 cell; the resulting P1 `Crystal` gets its true symmetry from the existing `Crystal.detect_symmetry`. `core/crystal.py` gains `DisorderState`, a `Crystal._state` field that survives `detect_symmetry`, and a shared `expand_cell`. `io/cif.py` wires the two stages in under a new default policy `"enumerate"`.

**Tech Stack:** Python 3.13, NumPy, qcelemental (radii), spglib (via `detect_symmetry`), PyCifRW, pytest. Conda env `cle`.

**Spec:** `docs/superpowers/specs/2026-10-15-disorder-states-design.md`

## Global Constraints

- **Run tests from the worktree with `PYTHONPATH=src`.** The `cle` env has an editable install pointing at the *main checkout's* `src/`, so a bare `python -m pytest` in a worktree silently tests the wrong code. Command: `PYTHONPATH=src python -m pytest tests/<file> -q`.
- Fractional translations in `SymOp` are integers in **twelfths**; `SymOp("-x,-y,-z")` parses a string op; `SymOp.identity()`.
- qcelemental `covalentradii.get` / `vdwradii.get` return **bohr**; multiply by `qcel.constants.bohr2angstroms`.
- `from_cif` keeps yielding `(Crystal, Monomer)` tuples (decision 1a). One extra yield per state; ordered blocks unchanged.
- Default `disorder="enumerate"` must warn (`RuntimeWarning`, message contains `"ordered state"`). `"error"`, `"ignore"`, `"drop"` keep today's behaviour exactly.
- No weighting/combination logic, no supercells, no state cap, no changes to `work/setup.py` or writers.
- Commit format `<type>(<scope>): <subject>`, imperative, ≤50 chars, no period.
- **Prerequisite:** the v2 checkout has an uncommitted docstring/type-hint pass over `core/crystal.py`, `io/cif.py` and others. This branch (`feat/disorder-states`, from `de713c3`) edits the same files. Commit that pass on v2 and rebase this branch onto it **before Task 1**, otherwise the merge will conflict on nearly every docstring.

## Review Focus

1. **Multi-block CIF mixing a disordered and an ordered block:** every block's states are yielded in file order, and the ordered block's crystal has `state is None`. → test in Task 4.
2. **Explicit groups and symmetry-generated disorder in one block:** labels combine as `"A1 / config 1"`, weights multiply, and explicitly chosen atoms are *not* re-treated as partial by the symmetry stage. → test in Task 4 (synthetic P-1, expected 4 states, weights 0.3/0.3/0.2/0.2, verified in prototype).
3. **An assembly whose groups don't sum to 1** (single half-occupied group): an extra "absent" choice with the remaining weight. → test in Task 2.
4. **A tagged site with occupancy 1** (ritonavir H11): kept in every state; the formula stays C37H48N6O5S2. → tests in Tasks 2 and 4.
5. **Disorder that has no ordered state in this cell:** a clear `ValueError` naming the site, plus a hint to use `'ignore'`/`'drop'`, never a silently wrong crystal. → tests in Tasks 3 and 4.

---

### Task 1: `DisorderState`, `Crystal.state`, shared `expand_cell`

**Files:**
- Modify: `src/crystalatte/core/crystal.py`
- Test: `tests/test_disorder.py` (create)

**Interfaces:**
- Produces:
  - `DisorderState(label: str, weight: float, groups: tuple[tuple[str, str], ...] = ())`: frozen dataclass.
  - `Crystal(vectors, angles, space_group, asu, state: DisorderState | None = None)`, with `Crystal.state -> DisorderState | None`.
  - `expand_cell(positions, symbols, sym_ops) -> tuple[NDArray[float64] (n,3) in [0,1), NDArray[str_] (n,), NDArray[intp] (n,) ASU site index]`.
  - `Crystal.detect_symmetry()` returns a crystal with the same `state`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_disorder.py`:

```python
"""Ordered-state enumeration for disordered CIFs (see docs/superpowers/specs/2026-10-15-disorder-states-design.md)"""
import numpy as np
import pytest

from conftest import load
from crystalatte.core.crystal import Crystal, DisorderState, expand_cell
from crystalatte.core.multimer import ASU
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

P1 = SpaceGroup.from_sym_ops([SymOp.identity()], number=1)


def _benzene_cell():
    crystal, _ = load("benzene")
    return crystal, expand_cell(crystal._asu.positions, crystal._asu.atoms, crystal.space_group.sym_ops)


def test_expand_cell_returns_every_atom_of_the_cell_once():
    _, (frac, symbols, site) = _benzene_cell()
    # Pbca, 6-atom ASU (half a molecule on an inversion centre) -> 4 molecules x 12 atoms
    assert len(frac) == 48
    assert symbols.tolist().count("C") == 24
    assert sorted(set(site.tolist())) == list(range(6))
    assert np.all((frac >= 0.0) & (frac < 1.0))


def test_ordered_crystal_has_no_state():
    crystal, _ = load("benzene")
    assert crystal.state is None


def test_state_survives_symmetry_detection():
    crystal, (frac, symbols, _) = _benzene_cell()
    state = DisorderState("config 1", 0.5)
    p1 = Crystal(*crystal.lattice_parameters, P1, ASU([str(s) for s in symbols], frac), state)
    detected = p1.detect_symmetry()
    assert len(detected.space_group.sym_ops) == 8, "P1 benzene must be lifted back to Pbca"
    assert detected.state is state
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py -q`
Expected: collection error, `ImportError: cannot import name 'DisorderState'`.

- [ ] **Step 3: Implement in `src/crystalatte/core/crystal.py`**

Add after the imports, before `class Crystal`:

```python
@dataclass(frozen=True)
class DisorderState:
    """One ordered state of a disordered CIF block

    :param label: Human-readable name, e.g. ``"A1 B2"`` or ``"config 1"``
    :param weight: Fraction of the disordered crystal this state represents, as implied by the CIF
    :param groups: ``(assembly, group)`` choices for explicit disorder groups; empty if none
    """

    label: str
    weight: float
    groups: tuple[tuple[str, str], ...] = ()


def expand_cell(
    positions: NDArray[np.float64], symbols: list[str], sym_ops: list[SymOp]
) -> tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.intp]]:
    """Expand an asymmetric unit into every distinct atom of the unit cell

    :param positions: Fractional coordinates of the ASU sites, shape ``(n, 3)``
    :param symbols: Atomic symbols of the ASU sites
    :param sym_ops: Operations of the space group
    :returns: Fractional coordinates wrapped into [0, 1), their symbols, and the ASU site each came from
    """
    frac, syms, site = [], [], []
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        for op in sym_ops:
            r = op.apply(np.asarray(pos, dtype=float)) % 1.0
            r[r > 1.0 - 1e-6] = 0.0  # wrap 0.999999 -> 0
            if frac:
                d = (np.asarray(frac) - r + 0.5) % 1.0 - 0.5  # periodic distance
                if np.any(np.linalg.norm(d, axis=1) < 1e-3):
                    continue
            frac.append(r)
            syms.append(sym)
            site.append(i)
    return np.array(frac), np.array(syms), np.array(site, dtype=np.intp)
```

In `class Crystal`, add the field **after** `_frac_to_cart` (an `init=False` field, so the positional order stays `vectors, angles, space_group, asu, state`), document it in the class docstring, and add the property next to `space_group`:

```python
    # computed matrices for coordinate transformations
    _frac_to_cart: NDArray[np.float64] = field(init=False, repr=False)

    # which ordered state of a disordered CIF this is, if any
    _state: DisorderState | None = None
```

```python
    :param _state: :class:`DisorderState` this crystal represents, ``None`` for an ordered CIF
```

```python
    @property
    def state(self) -> DisorderState | None:
        return self._state
```

In `detect_symmetry`, replace the block from `# expand ASU to full cell using declared symmetry ops` through `numbers = nums[keep].tolist()` with:

```python
        # expand ASU to full cell using declared symmetry ops
        positions, symbols, _ = expand_cell(
            self._asu.positions, self._asu.atoms, self._space_group.sym_ops
        )
        numbers = [qcel.periodictable.to_Z(s) for s in symbols]
```

and change its final line to carry the state:

```python
        return Crystal(self._vectors, self._angles, new_sg, self._asu, self._state)
```

- [ ] **Step 4: Run the tests to verify they pass, plus the symmetry suite as a regression guard for the `detect_symmetry` refactor**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py tests/test_symmetry_detect.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/crystal.py tests/test_disorder.py
git commit -m "feat(crystal): add DisorderState and expand_cell"
```

---

### Task 2: Explicit disorder groups

**Files:**
- Create: `src/crystalatte/io/disorder.py`
- Test: `tests/test_disorder.py`

**Interfaces:**
- Consumes: nothing from Task 1 yet.
- Produces:
  - `FULL_OCCUPANCY = 0.999`
  - `normalize_tag(value) -> str | None`: `1.0 → "1"`, `-1.0 → "-1"`, `"."`/`"?"`/`""`/`None → None`, other strings unchanged.
  - `ExplicitChoice(keep: NDArray[bool], settled: NDArray[bool], groups: tuple[tuple[str, str], ...], label: str, weight: float)`: a `NamedTuple`. `settled` marks the sites whose occupancy this choice resolved (every selectable site, chosen or not); a later stage treats them as fully occupied.
  - `explicit_choices(occupancies, assemblies: list[str | None], groups: list[str | None]) -> list[ExplicitChoice]`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_disorder.py`; add `from crystalatte.io.disorder import explicit_choices, normalize_tag` to the imports)

```python
@pytest.mark.parametrize("raw,expected", [
    (1.0, "1"), (-1.0, "-1"), (2.5, "2.5"), (".", None), ("?", None), ("", None), (None, None), ("A", "A"),
])
def test_normalize_tag(raw, expected):
    assert normalize_tag(raw) == expected


def test_explicit_choices_take_one_group_per_assembly():
    occ = [1.0, 0.7, 0.3, 0.6, 0.4]
    asm = [None, "A", "A", "B", "B"]
    grp = [None, "1", "2", "1", "2"]
    choices = explicit_choices(occ, asm, grp)
    assert [c.label for c in choices] == ["A1 B1", "A1 B2", "A2 B1", "A2 B2"]
    assert [round(c.weight, 3) for c in choices] == [0.42, 0.28, 0.18, 0.12]
    assert choices[1].keep.tolist() == [True, True, False, False, True]
    assert choices[1].groups == (("A", "1"), ("B", "2"))
    assert choices[1].settled.tolist() == [False, True, True, True, True]


def test_tagged_but_fully_occupied_site_is_in_every_choice():
    # ritonavir's H11 is tagged B/1 with occupancy 1; selecting by tag alone loses it
    choices = explicit_choices([0.5, 0.5, 1.0], ["B", "B", "B"], ["1", "2", "1"])
    assert len(choices) == 2
    assert all(c.keep[2] for c in choices)


def test_unnamed_assembly_is_labelled_by_group_only():
    choices = explicit_choices([0.6, 0.4], [None, None], ["1", "2"])
    assert [c.label for c in choices] == ["1", "2"]


def test_groups_sort_numerically():
    choices = explicit_choices([0.5, 0.5], ["A", "A"], ["10", "2"])
    assert [c.label for c in choices] == ["A2", "A10"]


def test_single_partial_group_also_admits_absence():
    choices = explicit_choices([1.0, 0.5], [None, "S"], [None, "1"])
    assert [(c.label, c.weight) for c in choices] == [("S1", 0.5), ("S-", 0.5)]
    assert choices[1].keep.tolist() == [True, False]


def test_untagged_and_negative_partial_sites_are_left_for_the_symmetry_stage():
    choices = explicit_choices([0.5, 0.5, 0.6, 0.4], [None, None, "A", "A"], [None, "-1", "1", "2"])
    assert len(choices) == 2
    assert all(c.keep[0] and c.keep[1] for c in choices)
    assert all(not c.settled[0] and not c.settled[1] for c in choices)


def test_block_without_groups_is_a_single_trivial_choice():
    (only,) = explicit_choices([1.0, 0.5], [None, None], [None, None])
    assert (only.label, only.weight, only.groups) == ("", 1.0, ())
    assert only.keep.all() and not only.settled.any()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'crystalatte.io.disorder'`.

- [ ] **Step 3: Implement `src/crystalatte/io/disorder.py`**

```python
"""Enumerate the ordered states a disordered CIF block admits

Two kinds of disorder are handled:

* explicit groups (``_atom_site_disorder_assembly`` / ``_atom_site_disorder_group``): one group
  is chosen per assembly, and the declared space group still holds
* symmetry-generated disorder (untagged or negative-group partial sites): a subset of the partial
  atoms of the full unit cell is chosen, and the result has no symmetry until
  :meth:`~crystalatte.core.crystal.Crystal.detect_symmetry` recovers it
"""

import itertools
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

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
        [g is not None and not g.startswith("-") and o < FULL_OCCUPANCY for g, o in zip(groups, occupancies)],
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
            [not selectable[i] or picked[assembly_of[i]] == groups[i] for i in range(n)], dtype=bool
        )
        pairs = tuple(picked.items())
        label = " ".join(f"{a}{g}" for a, g in pairs)
        weight = float(np.prod([w for _, w in combo]))
        choices.append(ExplicitChoice(keep, selectable.copy(), pairs, label, weight))
    return choices
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/io/disorder.py tests/test_disorder.py
git commit -m "feat(disorder): enumerate explicit disorder groups"
```

---

### Task 3: Symmetry-generated disorder

**Files:**
- Modify: `src/crystalatte/io/disorder.py`
- Test: `tests/test_disorder.py`

**Interfaces:**
- Consumes: `expand_cell` (Task 1), `FULL_OCCUPANCY` (Task 2).
- Produces:
  - `SymmetryChoice(frac: NDArray[float64], symbols: NDArray[str_], label: str, weight: float)`: a `NamedTuple` of the atoms present in the full cell.
  - `symmetry_choices(positions, symbols, occupancies, labels: list[str], sym_ops, lattice: NDArray (3,3) frac→cart) -> list[SymmetryChoice]`, ordered by weight descending, labelled `"config 1"`, `"config 2"`, …; raises `ValueError` when the cell has no ordered state.
  - `contacts(symbols, distances, tol=1.2) -> (bonded, clash)` boolean matrices.

- [ ] **Step 1: Write the failing tests** (append; extend the import to `from crystalatte.io.disorder import _periodic_distances, contacts, explicit_choices, normalize_tag, symmetry_choices`)

```python
def _ice_inputs():
    crystal, _ = load("ice")  # conftest reads ice with disorder="ignore"
    asu = crystal._asu
    occupancies = [1.0 if s == "O" else 0.5 for s in asu.atoms]
    labels = [f"site {i}" for i in range(len(asu.atoms))]
    return asu.positions, asu.atoms, occupancies, labels, crystal.space_group.sym_ops, crystal.lattice_vectors


def test_two_hydrogens_are_never_bonded():
    # ice: the two H sites on one O...O contact are 0.76 A apart, just above the 0.74 A H-H cutoff
    bonded, clash = contacts(["H", "H"], np.array([[0.0, 0.7], [0.7, 0.0]]))
    assert not bonded[0, 1] and clash[0, 1]


def test_ice_has_two_ordered_states_in_its_own_cell():
    choices = symmetry_choices(*_ice_inputs())
    # 18 Bernal-Fowler configurations of the 4-molecule cell fall into 12 + 6 under P6_3/mmc
    assert [c.label for c in choices] == ["config 1", "config 2"]
    assert [c.weight for c in choices] == pytest.approx([12 / 18, 6 / 18])
    for c in choices:
        assert sorted(c.symbols.tolist()) == ["H"] * 8 + ["O"] * 4


def test_ice_states_obey_the_ice_rules():
    *_, lattice = _ice_inputs()
    for c in symmetry_choices(*_ice_inputs()):
        d = _periodic_distances(c.frac, lattice)
        oxygen, hydrogen = c.symbols == "O", c.symbols == "H"
        assert ((d[np.ix_(oxygen, hydrogen)] < 1.2).sum(axis=1) == 2).all(), "two H per O"
        hh = d[np.ix_(hydrogen, hydrogen)] + np.eye(hydrogen.sum()) * 9
        assert hh.min() > 1.2, "one H per O...O contact"


def test_ice_states_recover_cc_and_cmc21():
    lattice_parameters = load("ice")[0].lattice_parameters
    numbers = []
    for c in symmetry_choices(*_ice_inputs()):
        crystal = Crystal(*lattice_parameters, P1, ASU([str(s) for s in c.symbols], c.frac)).detect_symmetry()
        refs = crystal.get_references()
        assert len(refs) == 2 and all(sorted(r.symbols) == ["H", "H", "O"] for r in refs)
        numbers.append(crystal.space_group.number)
    assert numbers == [9, 36]


def test_half_an_atom_per_cell_has_no_ordered_state():
    with pytest.raises(ValueError, match="H1.*whole number"):
        symmetry_choices(
            np.array([[0.10, 0.20, 0.30], [0.15, 0.25, 0.35]]), ["O", "H"], [1.0, 0.5],
            ["O1", "H1"], [SymOp.identity()], np.diag([5.0, 6.0, 7.0]),
        )


def test_whole_molecule_disorder_is_refused():
    ops = [SymOp.identity(), SymOp("-x,-y,-z")]
    with pytest.raises(ValueError, match="not bonded to any fully occupied atom"):
        symmetry_choices(np.array([[0.1, 0.1, 0.1]]), ["O"], [0.5], ["O1"], ops, np.diag([5.0, 5.0, 5.0]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py -q`
Expected: collection error, `ImportError: cannot import name 'symmetry_choices'`.

- [ ] **Step 3: Implement** (append to `src/crystalatte/io/disorder.py`; add `from collections import Counter`, `import qcelemental as qcel`, `from ..core.crystal import expand_cell` and `from ..core.sym_ops import SymOp` to the imports)

```python
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
        choices.append(SymmetryChoice(frac[present], cell_symbols[present], f"config {k}", count / len(valid)))
    return choices
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/io/disorder.py tests/test_disorder.py
git commit -m "feat(disorder): enumerate symmetry-generated disorder"
```

---

### Task 4: `from_cif` enumerates by default

**Files:**
- Modify: `src/crystalatte/io/cif.py`
- Modify: `tests/test_cif_parsing.py` (three disorder tests, lines ~546–604)
- Create: `tests/other/ritonavir_c2/ritonavir_c2.cif` (trimmed copy of `data/cif/2207870.cif`)
- Test: `tests/test_disorder.py`

**Interfaces:**
- Consumes: `DisorderState` (Task 1), `explicit_choices`, `normalize_tag`, `FULL_OCCUPANCY` (Task 2), `symmetry_choices` (Task 3).
- Produces: `DISORDER_POLICIES = ("enumerate", "error", "ignore", "drop")`; `from_cif(cif_file, disorder="enumerate")`.

- [ ] **Step 1: Add the ritonavir fixture**

`data/cif/2207870.cif` is untracked in the main checkout and 1.4 MB, almost all of it embedded SHELX `.res`/`.hkl` text. Save this one-off script **outside the repo** at `$CLAUDE_JOB_DIR/tmp/trim_cif.py`:

```python
"""Copy a CIF without its embedded SHELX text fields."""
import sys

src, dst = sys.argv[1], sys.argv[2]
out, skipping, in_text = [], False, False
for line in open(src):
    if line.split() and line.split()[0] in ("_shelx_res_file", "_shelx_hkl_file", "_shelx_fab_file"):
        skipping, in_text = True, False
        continue
    if skipping:
        if line.startswith(";"):
            if in_text:
                skipping = False
            in_text = not in_text
        continue
    out.append(line)
open(dst, "w").writelines(out)
```

Run:
```bash
mkdir -p tests/other/ritonavir_c2
python "$CLAUDE_JOB_DIR/tmp/trim_cif.py" /Users/jaden/gits/CrystaLattE/data/cif/2207870.cif tests/other/ritonavir_c2/ritonavir_c2.cif
```
Expected: a file of about 27 KB.

- [ ] **Step 2: Write the failing integration tests** (append to `tests/test_disorder.py`; add `import warnings`, `from collections import Counter`, `import crystalatte as cle` and `from conftest import cif` to the imports)

```python
def _states(path, **kwargs):
    with pytest.warns(RuntimeWarning, match="ordered state"):
        return list(cle.from_cif(path, **kwargs))


def test_ice_yields_its_two_ordered_states_by_default():
    pairs = _states(cif("ice"))
    assert [c.state.label for c, _ in pairs] == ["config 1", "config 2"]
    assert [c.space_group.number for c, _ in pairs] == [9, 36]
    assert sum(c.state.weight for c, _ in pairs) == pytest.approx(1.0)
    assert all(sorted(ref.symbols) == ["H", "H", "O"] for _, ref in pairs)


def test_ritonavir_yields_four_states_in_c2():
    pairs = _states(cif("ritonavir_c2"))
    assert [c.state.label for c, _ in pairs] == ["A1 B1", "A1 B2", "A2 B1", "A2 B2"]
    assert [round(c.state.weight, 3) for c, _ in pairs] == [0.358, 0.330, 0.162, 0.150]
    assert pairs[1][0].state.groups == (("A", "1"), ("B", "2"))
    for crystal, ref in pairs:
        assert len(crystal.space_group.sym_ops) == 4
        # H11 is tagged B/1 at full occupancy: losing it would give H47
        assert Counter(ref.symbols) == {"C": 37, "H": 48, "N": 6, "O": 5, "S": 2}


def test_ordered_cif_yields_one_stateless_crystal_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pairs = list(cle.from_cif(cif("benzene")))
    assert len(pairs) == 1 and pairs[0][0].state is None


def test_generate_runs_on_an_ordered_ice_state():
    crystal, _ = _states(cif("ice"))[1]
    assert len(cle.generate(crystal, 2, 8.0, verbose=0)) == 53


_CUBE = """_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""

_MIXED = """data_mixed
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
-x,-y,-z
""" + _CUBE + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
_atom_site_disorder_assembly
_atom_site_disorder_group
O1 O 0.0 0.0 0.0 1.0 . .
H1 H 0.096 0.0 0.0 0.5 . .
H2 H -0.024 0.0929 0.0 0.5 . .
O2 O 0.40 0.30 0.60 1.0 . .
H3 H 0.496 0.30 0.60 1.0 . .
H4a H 0.376 0.3929 0.60 0.6 A 1
H4b H 0.376 0.30 0.6929 0.4 A 2
"""

_ORDERED_WATER = """data_ordered
_symmetry_int_tables_number 1
""" + _CUBE + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
O1 O 0.40 0.30 0.60
H1 H 0.496 0.30 0.60
H2 H 0.376 0.3929 0.60
"""


def test_explicit_and_symmetry_disorder_in_one_block(tmp_path):
    # water on an inversion centre (H disorder generated by -1) next to a water with an explicit A1/A2 H
    path = tmp_path / "mixed.cif"
    path.write_text(_MIXED)
    pairs = _states(str(path))
    assert [c.state.label for c, _ in pairs] == ["A1 / config 1", "A1 / config 2", "A2 / config 1", "A2 / config 2"]
    assert [c.state.weight for c, _ in pairs] == pytest.approx([0.3, 0.3, 0.2, 0.2])
    assert all(len(c.get_references()) == 3 for c, _ in pairs)


def test_multi_block_cif_yields_each_blocks_states_in_order(tmp_path):
    path = tmp_path / "two_blocks.cif"
    path.write_text(_MIXED + "\n" + _ORDERED_WATER)
    pairs = _states(str(path))
    assert [c.state.label if c.state else None for c, _ in pairs] == [
        "A1 / config 1", "A1 / config 2", "A2 / config 1", "A2 / config 2", None,
    ]


def test_no_ordered_state_names_the_site_and_suggests_a_policy(tmp_path):
    path = tmp_path / "half.cif"
    path.write_text("data_half\n_symmetry_int_tables_number 1\n" + _CUBE + """loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
O1 O 0.10 0.20 0.30 1.0
H1 H 0.196 0.20 0.30 0.5
""")
    with pytest.raises(ValueError, match=r"H1.*whole number.*disorder='ignore'"):
        next(cle.from_cif(str(path)))
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py -q`
Expected: the new integration tests FAIL (the default still raises "partially occupied … policy is set to 'error'"); Tasks 1–3 tests still pass.

- [ ] **Step 4: Implement in `src/crystalatte/io/cif.py`**

Imports: replace `from ..core.crystal import Crystal` with `from ..core.crystal import Crystal, DisorderState` and add `from .disorder import FULL_OCCUPANCY, explicit_choices, normalize_tag, symmetry_choices`. Make sure `from numpy.typing import NDArray` is present (it arrives with the v2 polish pass; add it if the rebase hasn't happened).

Replace the `_FULL_OCCUPANCY` constant and the policy tuple with:

```python
# what `from_cif` may be told to do about partially occupied sites
DISORDER_POLICIES = ("enumerate", "error", "ignore", "drop")

# the trivial group that states with symmetry-generated disorder start from
_P1 = SpaceGroup.from_sym_ops([SymOp.identity()], number=1)
```

Pull the site description out of `_apply_disorder_policy` so both paths share it:

```python
def _describe_partial(block: dict, occupancies: NDArray[np.float64]) -> str:
    """Name every partially occupied site and its occupancy, e.g. ``"H1=0.5, H2=0.5"``"""
    labels = _first(block, ("_atom_site_label",)) or []
    labels = labels if isinstance(labels, list) else [labels]
    return ", ".join(
        f"{labels[i] if i < len(labels) else f'site {i}'}={occupancies[i]:g}"
        for i in np.flatnonzero(occupancies < FULL_OCCUPANCY)
    )
```

In `_apply_disorder_policy`, replace `_FULL_OCCUPANCY` with `FULL_OCCUPANCY` and replace the `labels = …` / `named = …` lines with `named = _describe_partial(block, occupancies)`. Nothing else in that function changes.

Add the tag reader and the state builder after `_resolve_asu`:

```python
def _tags(block: dict, field: str, n_sites: int) -> list[str | None]:
    """Return a per-site disorder tag column, all ``None`` if absent or the wrong length"""
    raw = _first(block, (field,))
    if raw is None:
        return [None] * n_sites
    values = raw if isinstance(raw, list) else [raw]
    return [normalize_tag(v) for v in values] if len(values) == n_sites else [None] * n_sites


def _ordered_states(
    block: dict,
    cell: tuple[tuple[float, float, float], tuple[float, float, float]],
    space_group: SpaceGroup,
    coords: NDArray[np.float64],
    occupancies: NDArray[np.float64],
) -> list[Crystal]:
    """Build one :class:`Crystal` per ordered state of a disordered block

    Explicit groups are chosen first and keep ``space_group``; any partial sites left over are
    resolved over the full cell and their crystals get their symmetry back from spglib.

    :param block: Cleaned data block
    :param cell: Lattice lengths and angles
    :param space_group: Declared space group
    :param coords: Fractional coordinates of every site, shape ``(n, 3)``
    :param occupancies: Occupancy of every site
    :returns: The state crystals, each carrying its :class:`DisorderState`
    :raises ValueError: If a leftover partial site has no ordered state in this cell
    """
    n = len(coords)
    labels = _first(block, ("_atom_site_label",)) or []
    labels = labels if isinstance(labels, list) else [labels]
    labels = [str(labels[i]) if i < len(labels) else f"site {i}" for i in range(n)]
    assemblies = _tags(block, "_atom_site_disorder_assembly", n)
    groups = _tags(block, "_atom_site_disorder_group", n)

    crystals = []
    for choice in explicit_choices(occupancies, assemblies, groups):
        asu = _resolve_asu(block, coords, choice.keep)
        residual = np.where(choice.settled, 1.0, occupancies)[choice.keep]
        if not (residual < FULL_OCCUPANCY).any():
            state = DisorderState(choice.label, choice.weight, choice.groups)
            crystals.append(Crystal(*cell, space_group, asu, state))
            continue

        lattice = Crystal(*cell, space_group, asu).lattice_vectors
        kept_labels = [labels[i] for i in np.flatnonzero(choice.keep)]
        for sym in symmetry_choices(
            asu.positions, asu.atoms, residual, kept_labels, space_group.sym_ops, lattice
        ):
            label = f"{choice.label} / {sym.label}" if choice.label else sym.label
            state = DisorderState(label, choice.weight * sym.weight, choice.groups)
            p1 = Crystal(*cell, _P1, ASU([str(s) for s in sym.symbols], sym.frac), state)
            crystals.append(p1.detect_symmetry())
    return crystals
```

Rewrite `from_cif`. The signature default becomes `disorder: str = "enumerate"`; the docstring `:param disorder:` line becomes:

```
    :param disorder: Policy for partially occupied sites: "enumerate" (default) yields one crystal per
        ordered state, each with :attr:`Crystal.state` set; "error", "ignore" or "drop" yield one crystal
```

Validate the policy before reading the file:

```python
    if disorder not in DISORDER_POLICIES:
        raise ValueError(
            f"unknown disorder policy {disorder!r}; expected one of {DISORDER_POLICIES}"
        )
```

Replace everything from `# grab the fractional coordinates and apply the disorder policy` to the end of the loop with:

```python
        # grab the fractional coordinates
        coords = np.array(
            [
                block["_atom_site_fract_x"],
                block["_atom_site_fract_y"],
                block["_atom_site_fract_z"],
            ]
        ).T
        occupancies = _site_occupancies(block, len(coords))
        cell = ((a, b, c), (alpha, beta, gamma))

        # build one crystal per ordered state, or one crystal by the requested policy
        if disorder == "enumerate" and (occupancies < FULL_OCCUPANCY).any():
            try:
                crystals = _ordered_states(block, cell, space_group, coords, occupancies)
            except ValueError as exc:
                raise ValueError(
                    f"{exc}. Pass disorder='ignore' or disorder='drop' to read this block anyway."
                ) from exc
            warnings.warn(
                f"CIF contains partially occupied site(s) ({_describe_partial(block, occupancies)}); "
                f"yielding {len(crystals)} ordered state(s), one Crystal each (see Crystal.state). "
                "Pass disorder='error', 'ignore' or 'drop' to opt out.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            keep = _apply_disorder_policy(block, coords, disorder)
            crystals = [Crystal(*cell, space_group, _resolve_asu(block, coords, keep))]

        # cross-check the declared volume against the one the cell implies
        declared = _first(block, ("_cell_volume",))
        if declared is not None and abs(crystals[0].volume - float(declared)) > 0.1:
            raise ValueError(
                f"Discordant unit cell volumes:\nDeclared in CIF: {float(declared):.2f} A^3\n"
                f"Calculated from primitive vectors: {crystals[0].volume:.2f} A^3"
            )

        # since the CIF format allows multiple data blocks, yield each crystal and its reference monomer
        for crystal in crystals:
            yield crystal, crystal.get_reference()
```

`_apply_disorder_policy` keeps its own unknown-policy check and its `occupancies` computation unchanged; it is only reached for `"error"`, `"ignore"` and `"drop"`, or for an ordered block (which returns all-True).

- [ ] **Step 5: Update the three existing tests whose premise the new default reverses** (`tests/test_cif_parsing.py`)

Replace `test_partial_occupancy_is_refused_by_default`:

```python
def test_default_refuses_a_cell_with_no_ordered_state(disordered):
    # The default enumerates ordered states; this synthetic P1 cell holds half a hydrogen
    # per site, which no ordered state can have, so it must still refuse rather than guess.
    with pytest.raises(ValueError, match="whole number"):
        next(cle.from_cif(disordered))
```

In `test_partial_occupancy_error_names_the_offending_sites`, change both calls to `next(cle.from_cif(disordered, disorder="error"))`.

Replace `test_ice_is_the_real_motivating_case`:

```python
def test_ice_is_the_real_motivating_case():
    # Not a synthetic construction: ice Ih is proton-disordered, both hydrogen
    # sites carry occupancy 0.5, and reading them all produces one oxygen with
    # four hydrogens at bonding distance. The default resolves it into ordered
    # states; 'error' still refuses and 'ignore' still builds the impossible molecule.
    with pytest.warns(RuntimeWarning, match="2 ordered state"):
        assert len(list(cle.from_cif(cif("ice")))) == 2

    with pytest.raises(ValueError, match="partially occupied"):
        next(cle.from_cif(cif("ice"), disorder="error"))

    with pytest.warns(RuntimeWarning):
        crystal, reference = next(cle.from_cif(cif("ice"), disorder="ignore"))
    assert reference.symbols.count("H") == 4, (
        "if this is no longer 4, the unphysical reading has changed and the "
        "disorder guard may no longer be guarding anything"
    )
```

- [ ] **Step 6: Run the parsing and disorder tests**

Run: `PYTHONPATH=src python -m pytest tests/test_disorder.py tests/test_cif_parsing.py -q`
Expected: all pass.

- [ ] **Step 7: Run the full suite**

Run: `PYTHONPATH=src python -m pytest -q -m "not slow"`
Expected: same pass/xfail counts as the baseline plus the new tests. Every existing ice test loads through `conftest.load`, which forces `disorder="ignore"`, so none should change. If one does, stop and report rather than adjusting expected values.

- [ ] **Step 8: Commit**

```bash
git add src/crystalatte/io/cif.py tests/test_cif_parsing.py tests/test_disorder.py tests/other/ritonavir_c2/ritonavir_c2.cif
git commit -m "feat(cif): enumerate ordered states by default" \
  -m "Disordered blocks now yield one Crystal per ordered state, each with
Crystal.state (label, weight). Explicit disorder groups keep the declared
space group; symmetry-generated disorder (ice) is resolved over the full
cell and its symmetry recovered with detect_symmetry. 'error', 'ignore'
and 'drop' remain available."
```

---

## Not in this plan (by decision)
- Combining per-state results (decision 4a), supercells (3a), state caps (6), `write_inputs` changes (7).
- `src/crystalatte/TODO.md` ("ice, disordered water molecules") is left alone: it has uncommitted edits on v2. Tick it off after the prerequisite rebase.
