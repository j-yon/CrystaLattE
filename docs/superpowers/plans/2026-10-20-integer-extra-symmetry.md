# Integer Extra Molecular Symmetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold exact extra molecular symmetry into the integer orbit key as a per-reference dressing group `D`, and delete the Cartesian congruence pass together with its point-group detection.

**Architecture:** A new pure module `work/dedup/dressing.py` computes the working cell's integer symmetry `H` (space-group lookup via a generic invariant metric, plus a metric test on the actual cell, both in a Niggli-reduced basis) and each reference's `D = W ∪ {h ∈ H : exact self-isometry, twelfths translation}`. `context.bind` stores `D` as `dress_stacked_by_ref`, which `invariants` reads in place of `W_stacked_by_ref`. `congruence.py`, `matchers.py`, the gate and the dead helpers are removed.

**Tech Stack:** Python 3.11+, numpy, spglib (`niggli_reduce`), pytest.

**Spec:** `docs/superpowers/specs/2026-10-20-integer-extra-symmetry-design.md`

## Global Constraints

- Run everything from the worktree with `PYTHONPATH=src`. The editable install points at the main checkout, so without it the tests import the wrong code.
- `EXACT_ATOL = 1e-6` Å (monomer test), `METRIC_RTOL = 1e-9` (metric tests), `TWELFTHS_ATOL = 1e-9` (translation integrality).
- `W` always enters `D` unconditionally (ice's site ops hold only to about 2.7e-4 Å).
- Per-cluster work stays exact `int64`; no tolerance is added to the orbit key.
- Commit format `<type>(<scope>): <subject>`, imperative, ≤ 50 chars, no period.

## Review Focus

1. Non-standard CIF settings (P2₁/a naphthalene, unique-axis variants): `H` must still contain `W`. Covered by the `W ⊆ H` assertion run over every fixture in Task 2.
2. Cells promoted by `detect_symmetry` (ammonia P1 → P2₁3, `crystal_system=None`): the lookup must work from the ops alone. Covered by ammonia `|H| = 48` in Task 2.
3. A slightly inconsistent cell where the metric test disagrees with the lookup: the closure must stay finite, or fall back to the lookup. Covered by a perturbed-cell unit test in Task 2.
4. A monomer whose site ops are not exact (ice): `W` must survive in `D`. Covered by ice `|D| = |W|` in Task 2.
5. Z' > 1 (fomepizole, pyrazole): one `D` per reference, built from that reference's own atoms. Covered by the multiref binding test in Task 3.

---

### Task 1: Golden regression rows (pre-change)

**Files:**
- Modify: `tests/test_generate.py` (append a new parametrized test)

- [ ] **Step 1: Add the regression test.** Values were recorded from clean v2 (`ddc4c41`); CO₂ is added in Task 4 once its new value is known.

```python
# Extra molecular symmetry beyond the site symmetry. Recorded from the exact +
# Cartesian-congruence pipeline (ddc4c41) before the congruence pass was
# folded into the integer orbit key; unique count, total multiplicity and the
# multiplicity histogram must all survive that change.
@pytest.mark.parametrize("name,N,R,n_unique,total,histogram", [
    ("triazine", 2, 20.0, 61, 418, {2: 5, 6: 44, 12: 12}),
    pytest.param("triazine", 2, 35.0, 266, 2082, {2: 9, 6: 178, 12: 75, 24: 4},
                 marks=pytest.mark.slow),
    ("triazine", 3, 12.0, 123, 2103, {3: 1, 6: 2, 9: 14, 18: 103, 36: 3}),
    ("ammonia", 2, 20.0, 183, 1075, {2: 2, 3: 5, 6: 176}),
    ("benzene", 2, 20.0, 116, 366, {2: 49, 4: 67}),
])
def test_extra_symmetry_regression(name, N, R, n_unique, total, histogram):
    mers = cle.generate(crystal(name), N, R, **GENERATE_KW)
    assert len(mers) == n_unique
    assert sum(m.multiplicity for m in mers) == total
    assert dict(Counter(m.multiplicity for m in mers)) == histogram
```

Add `from collections import Counter` to the imports.

- [ ] **Step 2: Run it; it passes on the current code.**
Run: `PYTHONPATH=src python -m pytest tests/test_generate.py -k extra_symmetry -q`
Expected: 5 passed.

- [ ] **Step 3: Commit** `test(dedup): pin extra-symmetry regression counts`

### Task 2: `dressing.py`, cell symmetry and dressing group

**Files:**
- Create: `src/crystalatte/work/dedup/dressing.py`
- Create: `tests/test_dressing.py`

**Interfaces:**
- Produces: `cell_symmetry(rotations: NDArray[int] (k,3,3), lattice: NDArray (3,3) columns = frac_to_cart) -> NDArray[int64] (h,3,3)`
- Produces: `dressing_group(frac_coords (n,3), centroid_frac (3,), Z (n,), lattice (3,3), W: list[NDArray], W_tr: list[NDArray], H: NDArray) -> tuple[NDArray[int64] (d,3,3), NDArray[int64] (d,3)]`
- Constants: `EXACT_ATOL`, `METRIC_RTOL`, `TWELFTHS_ATOL`

- [ ] **Step 1: Write the failing tests** (`tests/test_dressing.py`):

```python
import numpy as np
import pytest

from crystalatte.work.dedup import context, dressing
from conftest import crystal

FIXTURES = ["14_cyclohexanedione", "acetic_acid", "adamantane", "ammonia", "anthracene",
            "benzene", "carbon_dioxide", "cyanamide", "cytosine", "ethyl_carbamate",
            "formamide", "hexamine", "ice", "imidazole", "naphthalene", "oxalic_acid_alpha",
            "oxalic_acid_beta", "pyrazine", "pyrazole", "succinic_acid", "triazine",
            "trioxane", "uracil", "urea", "fomepizole", "hydroxyurea"]
H_ORDER = {"ethyl_carbamate": 2, "ammonia": 48, "carbon_dioxide": 48, "hexamine": 48,
           "adamantane": 16, "urea": 16, "ice": 24, "triazine": 24, "trioxane": 24}
MONOCLINIC = {"14_cyclohexanedione", "anthracene", "formamide", "imidazole", "naphthalene",
              "oxalic_acid_beta", "succinic_acid", "uracil", "hydroxyurea"}
EXTRA = {"triazine": 12, "carbon_dioxide": 12}

def _H(xtal):
    rots = np.stack([op.rot for op in xtal.space_group.sym_ops]).astype(np.int64)
    return dressing.cell_symmetry(rots, np.asarray(xtal.lattice_vectors, float))

@pytest.mark.parametrize("name", FIXTURES)
def test_cell_symmetry_order_and_contains_site_symmetry(name):
    xtal = crystal(name).detect_symmetry(1e-3)
    H = _H(xtal)
    expected = H_ORDER.get(name, 4 if name in MONOCLINIC else 8)
    assert len(H) == expected
    for op in xtal.space_group.sym_ops:
        assert any(np.array_equal(op.rot.astype(np.int64), h) for h in H)

def test_cell_symmetry_is_the_box_not_the_bravais_lattice():
    # R-3c on hexagonal axes: the R-centred lattice has -3m (12), the hexagonal box 6/mmm (24).
    # Triazine's extra ops live outside -3m, so Bravais-lattice symmetry would lose its merges.
    xtal = crystal("triazine").detect_symmetry(1e-3)
    H = _H(xtal)
    cen = np.array([2 / 3, 1 / 3, 1 / 3])
    def keeps_R(h):
        y = h @ cen
        return any(np.allclose(y - t, np.rint(y - t)) for t in (np.zeros(3), cen, 2 * cen))
    assert sum(keeps_R(h) for h in H) == 12

def test_cell_symmetry_falls_back_to_lookup_on_inconsistent_cell():
    # A tetragonal lookup on a cell whose a and b disagree by 1e-3: the metric test sees only
    # orthorhombic symmetry, the union is still a group, and the lookup ops are all present.
    rots = np.stack([np.eye(3), np.diag([-1, -1, 1]), [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                     [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]]).astype(np.int64)
    H = dressing.cell_symmetry(rots, np.diag([5.0, 5.005, 7.0]))
    assert len(H) == 16

@pytest.mark.parametrize("name", FIXTURES)
def test_dressing_group(name):
    xtal = crystal(name).detect_symmetry(1e-3)
    refs = xtal.get_references()
    context.bind(xtal, refs)
    H = _H(xtal)
    A = np.asarray(xtal.lattice_vectors, float)
    for i, mon in enumerate(refs):
        W, W_tr = context.W_lin_by_ref[i], context.W_tr_by_ref[i]
        D, D_tr = dressing.dressing_group(mon.frac_coords, mon.centroid_frac,
                                          context.Z_by_ref[i], A, W, W_tr, H)
        assert len(D) == EXTRA.get(name, len(W))
        for w in W:  # W is always kept, even when it is inexact (ice)
            assert any(np.array_equal(w, d) for d in D)
        for d in D:  # D is a subgroup of H
            assert any(np.array_equal(d, h) for h in H)
        c = np.asarray(mon.centroid_frac, float)
        for d, t in zip(D, D_tr):  # every dressing fixes the centroid exactly, in twelfths
            assert np.allclose(d @ c + t / 12.0, c, atol=1e-6)
        prods = {(a @ b).tobytes() for a in D for b in D}
        assert prods == {d.tobytes() for d in D}  # closed
```

- [ ] **Step 2: Run them; they fail** with `ImportError: cannot import name 'dressing'`.
Run: `PYTHONPATH=src python -m pytest tests/test_dressing.py -q`

- [ ] **Step 3: Implement `dressing.py`.**

```python
"""Integer dressing groups: the site symmetry plus every exact cell-symmetry self-isometry of a monomer.

A dressing only has to be an isometry that maps its monomer onto itself and is integer in the working
basis, so the orbit key can use any operation of the cell's own symmetry ``H`` that fixes the monomer,
whether or not the space group contains it. ``H`` is the symmetry of the box spanned by the cell vectors,
not the point group of the Bravais lattice: triazine's extra operations on hexagonal axes lie in 6/mmm
but outside the R lattice's -3m.
"""

import itertools
import warnings
from functools import cache
from typing import Final

import numpy as np
import spglib
from numpy.typing import NDArray

#: Cartesian tolerance (Angstroms) for accepting a cell operation as an exact self-isometry
EXACT_ATOL: Final[float] = 1e-6
#: relative tolerance on metric preservation
METRIC_RTOL: Final[float] = 1e-9
#: absolute tolerance on a dressing translation being a whole number of twelfths
TWELFTHS_ATOL: Final[float] = 1e-9
#: order of the largest cell symmetry group (m-3m)
_MAX_ORDER: Final[int] = 48

# fixed generic metric; averaging it over a point group leaves no accidental extra symmetry
_M0 = np.array([[1.0, 0.3141592653589793, 0.2718281828459045],
                [0.0, 1.4142135623730951, 0.5772156649015329],
                [0.0, 0.0, 1.7320508075688772]])
_G0: Final[NDArray] = _M0 @ _M0.T


def _int_inv(M: NDArray) -> NDArray:
    """Return the exact inverse of a unimodular integer matrix or stack of them"""
    return np.rint(np.linalg.inv(M)).astype(np.int64)


@cache
def _candidates() -> NDArray:
    """Return every 3x3 matrix with entries in {-1, 0, 1} and determinant +-1

    Complete for the symmetry of a Niggli-reduced basis.
    """
    E = np.array(list(itertools.product((-1, 0, 1), repeat=9)), dtype=np.int64).reshape(-1, 3, 3)
    return E[np.abs(np.rint(np.linalg.det(E.astype(float)))) == 1]


def _preserving(G: NDArray, E: NDArray) -> NDArray:
    """Return the candidates ``h`` with ``h^T G h == G`` to :data:`METRIC_RTOL`"""
    Gs = np.einsum("nji,jk,nkl->nil", E, G, E)
    return E[np.abs(Gs - G).max(axis=(1, 2)) <= METRIC_RTOL * np.abs(G).max()]


def _closure(ops: NDArray) -> NDArray | None:
    """Return the group generated by ``ops``, or ``None`` if it outgrows :data:`_MAX_ORDER`"""
    group = {o.tobytes(): o for o in ops}
    frontier = list(group.values())
    while frontier:
        fresh = []
        for a in frontier:
            for b in list(group.values()):
                for p in (a @ b, b @ a):
                    if p.tobytes() not in group:
                        group[p.tobytes()] = p
                        fresh.append(p)
                        if len(group) > _MAX_ORDER:
                            return None
        frontier = fresh
    return np.stack(list(group.values()))


def cell_symmetry(rotations: NDArray, lattice: NDArray) -> NDArray:
    """Return the integer symmetry of the working cell: lookup from the space group plus a metric test

    The lookup averages a generic metric over the space group's linear parts, so it yields the
    crystal family's cell symmetry in whatever setting the cell uses, from the operations alone.
    The metric test adds symmetry the actual cell shape has beyond that. Both run in a
    Niggli-reduced basis, where the candidates are complete, and are mapped back.

    :param rotations: ``(k, 3, 3)`` integer linear parts of the space group, in the cell's basis
    :param lattice: ``(3, 3)`` fractional-to-Cartesian matrix, cell vectors as columns
    :returns: ``(h, 3, 3)`` int64 operations, a group containing every row of ``rotations``
    """
    A = np.asarray(lattice, dtype=float)
    reduced = spglib.niggli_reduce(A.T)
    Q = np.eye(3, dtype=np.int64) if reduced is None else np.rint(np.linalg.solve(A, reduced.T)).astype(np.int64)
    if abs(round(np.linalg.det(Q))) != 1:
        Q = np.eye(3, dtype=np.int64)
    Q_inv = _int_inv(Q)

    E = _candidates()
    P = Q_inv @ np.asarray(rotations, dtype=np.int64) @ Q
    lookup = _preserving(np.einsum("nji,jk,nkl->il", P, _G0, P) / len(P), E)
    A_red = A @ Q
    metric = _preserving(A_red.T @ A_red, E)

    H = _closure(np.concatenate((lookup, metric)))
    if H is None:
        warnings.warn("the cell metric is inconsistent with its space group; using the space-group "
                      "lookup alone for the cell symmetry", RuntimeWarning, stacklevel=2)
        H = lookup
    return Q @ H @ Q_inv


def dressing_group(frac_coords: NDArray, centroid_frac: NDArray, Z: NDArray, lattice: NDArray,
                   W: list[NDArray], W_tr: list[NDArray], H: NDArray) -> tuple[NDArray, NDArray]:
    """Return a monomer's dressing group: its site symmetry plus every exact cell-symmetry self-isometry

    An operation ``h`` of ``H`` outside ``W`` joins when ``x -> h (x - c) + c`` puts every atom within
    :data:`EXACT_ATOL` of an atom of the same element, and its translation ``(I - h) c`` is a whole
    number of twelfths. ``W`` is kept unconditionally, exact or not.

    :param frac_coords: ``(n, 3)`` fractional atom positions of the reference monomer
    :param centroid_frac: the centroid ``c`` the site symmetry fixes
    :param Z: ``(n,)`` atomic numbers
    :param lattice: ``(3, 3)`` fractional-to-Cartesian matrix
    :param W: site-symmetry linear parts, int64
    :param W_tr: their translations, in twelfths
    :param H: the cell symmetry from :func:`cell_symmetry`
    :returns: ``(D, D_tr)``, int64 arrays of shapes ``(d, 3, 3)`` and ``(d, 3)``
    """
    X = np.asarray(frac_coords, dtype=float)
    c = np.asarray(centroid_frac, dtype=float)
    A = np.asarray(lattice, dtype=float)
    same = Z[:, None] == Z[None]

    D, D_tr = list(W), list(W_tr)
    skipped = 0
    for h in H:
        if any(np.array_equal(h, d) for d in D):
            continue
        dist = np.linalg.norm((((X - c) @ h.T + c)[:, None] - X[None]) @ A.T, axis=2)
        if np.where(same, dist, np.inf).min(axis=1).max() > EXACT_ATOL:
            continue
        t = 12.0 * (c - h @ c)
        if np.abs(t - np.rint(t)).max() > TWELFTHS_ATOL:
            skipped += 1
            continue
        D.append(h)
        D_tr.append(np.rint(t).astype(np.int64))

    if skipped:
        warnings.warn(f"{skipped} exact self-isometr{'y' if skipped == 1 else 'ies'} skipped: the "
                      "translation is not a whole number of twelfths", RuntimeWarning, stacklevel=2)
    if len(D) > len(W) and _closure(np.stack(D)).shape[0] != len(D):
        # only possible when an inexact site operation composes with an exact extra one
        warnings.warn("site symmetry and exact extra symmetry do not close into a group; "
                      "using the site symmetry alone", RuntimeWarning, stacklevel=2)
        D, D_tr = list(W), list(W_tr)
    return np.stack(D).astype(np.int64), np.stack(D_tr).astype(np.int64)
```


- [ ] **Step 4: Run the tests; they pass.**
Run: `PYTHONPATH=src python -m pytest tests/test_dressing.py -q`

- [ ] **Step 5: Commit** `feat(dedup): add integer cell-symmetry dressing groups`

### Task 3: Wire `D` into `bind` and the orbit key

**Files:**
- Modify: `src/crystalatte/work/dedup/context.py` (`bind`, globals)
- Modify: `src/crystalatte/work/dedup/invariants.py:109-110,165` (read `dress_stacked_by_ref`)
- Modify: `tests/test_dedup.py` (`_same_label` helper, about line 68: `context.W_stacked_by_ref` → `context.dress_stacked_by_ref`)

**Interfaces:**
- Consumes: `dressing.cell_symmetry`, `dressing.dressing_group`
- Produces: `context.dress_stacked_by_ref: list[tuple[D, D_inv, D_tr]]`; `W_lin_by_ref` and `W_tr_by_ref` unchanged

- [ ] **Step 1:** In `bind`, compute `H` once from `crystal.space_group.sym_ops` and `frac_to_cart`. Per reference, after `W` / `W_tr`, call `dressing_group(mon.frac_coords, c_frac, Z_by_ref[-1], frac_to_cart, W, W_tr, H)` and append `(D, int_inv(D), D_tr)` to `dress_stacked_by_ref`. Replace `W_stacked_by_ref` throughout.
- [ ] **Step 2:** `invariants._dressed_minima` and `_group_keys` read `context.dress_stacked_by_ref`. Docstrings say "dressing group".
- [ ] **Step 3: Run** `PYTHONPATH=src python -m pytest -q -x`. Expected: green. The congruence pass still runs but has nothing left to merge on triazine; the golden rows hold.
- [ ] **Step 4: Commit** `feat(dedup): dress orbit key with exact extra symmetry`

### Task 4: Delete the Cartesian congruence pass

**Files:**
- Delete: `src/crystalatte/work/dedup/congruence.py`, `src/crystalatte/work/dedup/matchers.py`
- Modify: `src/crystalatte/work/neighbors.py` (import and congruence block)
- Modify: `src/crystalatte/work/dedup/context.py` (drop `self_iso_by_ref`, `self_iso_tr_by_ref`, `check_congruence_by_ref`, the `point_group_seeded` call, the `Monomer` import if unused, `SITE_ATOL` comment block about the gate)
- Modify: `src/crystalatte/work/dedup/invariants.py` (delete `fingerprints_match_many`, `is_mono_reference`, `base_reference`, and the `integer` parameter and float branch of `relative_transforms`)
- Modify: `src/crystalatte/work/generate.py` (linear-molecule warning and docstring mention of the Cartesian pass, if any)
- Modify tests:
  - `test_dedup.py`:
    - `test_bind_sets_site_symmetry_and_gate` → `test_bind_sets_site_symmetry`: keep the W assertions, drop the S and gate lines.
    - Delete `test_congruence_gate_opens_iff_point_group_exceeds_site_symmetry`.
    - `test_bind_sets_dressing_translations`: assert on the `dress_stacked_by_ref` translations instead of `self_iso_tr_by_ref`.
    - `test_bind_dressing_translations_vanish_for_benzene`: assert all `dress_stacked_by_ref[0][2] == 0`.
    - `test_every_dressing_is_a_symmetry_of_its_monomer`: check every `D` op (Cartesian `A d A⁻¹`) displaces no atom more than 1e-3 Å (the same bound as `W`, since `D ⊇ W` and ice's W is the loosest at 2.7e-4). Drop the 0.05 Å S bound.
    - `test_bound_dressing_groups_satisfy_the_containment_theorem`: `W ⊆ D` for every reference.
    - `relative_transforms(g, integer=True)` → `relative_transforms(g)`.
  - `test_multiref.py::test_bind_multi_reference_state`: `len(context.dress_stacked_by_ref) == 4`.
  - `conftest.py`: remove `test_full_congruence_check_is_unconditional` from `_SLOW_TESTS`.
  - `test_generate.py`: add the CO₂ row to `test_extra_symmetry_regression` with the value measured after this task. Re-measure the CO₂ R=15 row and `test_linear_monomer_dedup_is_complete`; if either now meets its target, remove the strict xfail, otherwise update the reason's counts.

- [ ] **Step 1:** Make the source deletions and edits above.
- [ ] **Step 2:** `grep -rn "congruence\|self_iso\|check_congruence\|W_stacked\|fingerprints_match_many\|is_mono_reference\|base_reference\|anchored_match" src tests` must show only historical comments, each reworded or removed.
- [ ] **Step 3:** Make the test edits; measure CO₂ (`R=20` regression row, `R=15` row, linear test) and record the values.
- [ ] **Step 4: Run** `PYTHONPATH=src python -m pytest -q`. Expected: all pass, with the xfails reduced only where CO₂ now meets its target.
- [ ] **Step 5: Commit** `refactor(dedup): remove Cartesian congruence pass` (body: what is removed and why, citing the spec).

### Task 5: Docs

**Files:**
- Modify: `tests/README.md` (suite map rows for `test_dedup.py`, add `test_dressing.py`, totals, timings)
- Modify: `src/crystalatte/work/dedup/context.py` module docstring (if not already done in Task 4)

- [ ] **Step 1:** Update the README rows and the test count from a final `pytest -q` run.
- [ ] **Step 2: Commit** `docs(tests): update suite map for dressing groups`
