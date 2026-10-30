# Linear-Monomer Dot-Product Key Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make multimer deduplication complete for linear (and single-atom) monomers, exactly, by comparing dot-product tables written as integer formulas.

**Architecture:** Two new modules in `work/dedup/`: `linear.py` (once-per-run setup: which references are linear, the cell-shape pieces, and each linear reference's centroid/axis as fixed part plus free unknowns) and `dotkey.py` (per-cluster canonical dot-product key). `context.bind` stores the setup; `invariants.orbit_labels` anchors mixed clusters on a non-linear member and, for all-linear reference groups, merges existing key classes whose representatives share a dot-product key.

**Tech Stack:** Python, NumPy (int64), `fractions.Fraction` / Python ints for the small exact linear algebra, pytest.

**Spec:** `docs/dedup_correctness.tex`, section "Planned: a dot-product key for linear monomers" (`\label{sec:linear}`). Read it before starting; it explains every quantity below.

## Global Constraints

- Every per-cluster comparison is exact integer equality. No tolerance enters per cluster.
- Tolerances appear only once per reference: linearity uses the existing rank test (`matrix_rank(cart - mean, tol=1e-6)`); end-to-end symmetry uses `dressing.EXACT_ATOL` (1e-6 Å).
- A merge must never join non-congruent clusters. Any doubt in the per-reference setup falls back to the existing frame key for that reference, with a `RuntimeWarning`; it never raises.
- The existing orbit key is unchanged for groups with a non-linear member, except that members are ordered with non-linear references first.
- `context` globals are rebound by `bind`; consumers read them as `context.<name>`, never `from .context import <name>`.
- Docstrings use the repo's `:param:` / `:returns:` style. Commits follow `<type>(<scope>): <subject>`, imperative, ≤ 50 chars.

## Review Focus

1. **A linear reference whose site operations only approximately keep its axis** (CIF rounding). Expect a warning and the frame key for that reference, not a crash. Test: `test_line_params_rejects_an_axis_the_dressings_do_not_keep` (Task 2).
2. **Several independent linear references** (Z' > 1, e.g. CO₂ with detection off). Each reference needs its own unknowns, and no distinct dimer may be lost. Test: `test_zprime_linear_references_lose_no_class` (Task 5).
3. **Mixed cluster whose reference 0 is linear.** It must anchor on the non-linear member. Tests: `test_anchor_rank_puts_nonlinear_first` (Task 3) and `test_mixed_cocrystal_is_complete_and_sound` (Task 5).
4. **Wide coefficient rows** (triclinic or monoclinic cell with free centroid and axis) push `lex_codes` past 63 bits and onto its rank fallback. Must still be exact. Test: `test_general_position_polar_linear_is_sound` (Task 5) runs a monoclinic cell with 3 + 3 free unknowns.
5. **Single atoms in an all-linear group.** They have no axis, so their axis rows must be absent from the table layout, not zero-filled. Tests: `test_single_atom_crystal_is_unchanged` (Task 5) and `test_dot_key_single_atoms` (Task 4).

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/crystalatte/work/dedup/linear.py` | Exact helpers and per-reference setup: `integer_nullspace`, `metric_basis`, `is_linear`, `symmetric_ends`, `LineParams`, `line_params` |
| Create `src/crystalatte/work/dedup/dotkey.py` | `dot_keys(cands, refs)`: canonical dot-product key rows for clusters sharing one reference tuple |
| Modify `src/crystalatte/work/dedup/context.py` | Store `linear_by_ref`, `line_by_ref`, `metric_pieces`, `anchor_rank` in `bind` |
| Modify `src/crystalatte/work/dedup/invariants.py` | Order members by `anchor_rank` in `_group_keys`/`orbit_labels`; route all-linear groups through `dotkey` |
| Modify `src/crystalatte/work/generate.py` | Remove the linear warning and the docstring `.. warning::` |
| Create `tests/test_linear.py` | Unit and integration tests for the new modules |
| Modify `tests/test_generate.py`, `tests/test_dedup.py` | CO₂ R = 20 regression row; remove warning tests; add referee rows |
| Modify `tests/README.md`, `docs/dedup_correctness.tex`, `src/crystalatte/TODO.md` | Status updates |

---

### Task 1: Exact linear algebra and cell-shape pieces

**Files:**
- Create: `src/crystalatte/work/dedup/linear.py`
- Test: `tests/test_linear.py`

**Interfaces:**
- Produces:
  - `integer_nullspace(M: NDArray) -> NDArray`: for an integer `(r, m)` matrix, returns int64 `(k, m)` rows spanning `{x : M x = 0}` over the rationals. Each row is primitive (gcd 1) and `k = m - rank`. If `r == 0`, returns the identity.
  - `metric_basis(H: NDArray) -> NDArray`: int64 `(k, 3, 3)` symmetric matrices `G_k` spanning the metrics every `h` in `H` preserves (`hᵀ G_k h = G_k`).

- [ ] **Step 1: Write the failing tests**

Later tasks extend this import block with `dotkey`, `invariants`, `neighbors`, `generate`, `SymOpList`, `SpaceGroup`, `Crystal`, `ASU` and `count_chemistry_classes`, each imported from where `tests/test_dedup.py` and `tests/test_disorder.py` import it.

```python
import numpy as np
import pytest
from crystalatte.work.dedup import context, dressing, linear
from conftest import GENERATE_KW, crystal, load

def _H(cry):
    rots = np.stack([op.rot for op in cry.space_group.sym_ops]).astype(np.int64)
    return dressing.cell_symmetry(rots, np.asarray(cry.lattice_vectors, float))

def test_integer_nullspace():
    N = linear.integer_nullspace(np.array([[1, 1, 1], [0, 2, -2]]))
    assert N.shape == (1, 3) and abs(np.gcd.reduce(N[0])) == 1
    assert np.array_equal(np.array([[1, 1, 1], [0, 2, -2]]) @ N.T, np.zeros((2, 1)))
    assert np.array_equal(linear.integer_nullspace(np.zeros((0, 4), np.int64)), np.eye(4, dtype=np.int64))

@pytest.mark.parametrize("name,k", [
    ("carbon_dioxide", 1), ("triazine", 2), ("adamantane", 2),
    ("benzene", 3), ("naphthalene", 4), ("ethyl_carbamate", 6),
])
def test_metric_basis_dimension(name, k):
    cry = crystal(name)
    B = linear.metric_basis(_H(cry))
    assert B.shape == (k, 3, 3) and B.dtype == np.int64
    for h in _H(cry):
        assert np.array_equal(np.einsum("ji,njk,kl->nil", h, B, h), B)

@pytest.mark.parametrize("name", ["carbon_dioxide", "triazine", "naphthalene", "ethyl_carbamate"])
def test_actual_metric_lies_in_the_basis(name):
    cry = crystal(name)
    A = np.asarray(cry.lattice_vectors, float)
    B = linear.metric_basis(_H(cry)).reshape(-1, 9).T.astype(float)
    g, *_ = np.linalg.lstsq(B, (A.T @ A).ravel(), rcond=None)
    assert np.abs(B @ g - (A.T @ A).ravel()).max() < 1e-8
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_linear.py -v`
Expected: FAIL (`ImportError: cannot import name 'linear'`).

- [ ] **Step 3: Implement `integer_nullspace` and `metric_basis` in `linear.py`**

`integer_nullspace`: reduce to row echelon form with `fractions.Fraction`, read off one basis vector per free column, then scale each to integers (lcm of denominators) and divide by its gcd. `metric_basis`: for each of the 6 elementary symmetric matrices `E_ij` (`E_ii`, and `E_ij + E_ji` for i < j), form `S = Σ_{h∈H} hᵀ E h`, which is an integer matrix. Keep a maximal linearly independent subset (`np.linalg.matrix_rank` on the flattened stack is safe at these magnitudes), then divide each by the gcd of its entries.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_linear.py -v`
Expected: PASS (all 12).

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/dedup/linear.py tests/test_linear.py
git commit -m "feat(dedup): add exact nullspace and metric basis"
```

---

### Task 2: Per-reference line parameters

**Files:**
- Modify: `src/crystalatte/work/dedup/linear.py`
- Test: `tests/test_linear.py`

**Interfaces:**
- Consumes: `integer_nullspace` (Task 1).
- Produces:
  - `is_linear(cart_coords: NDArray) -> bool`: the existing rank test from `generate.py`, `matrix_rank(pos - pos.mean(0), tol=1e-6) <= 1`. True for single atoms.
  - `symmetric_ends(frac_coords, centroid_frac, Z, lattice) -> bool`: inversion through the centroid puts every atom within `dressing.EXACT_ATOL` of a same-element atom.
  - A frozen dataclass:
    ```python
    @dataclass(frozen=True)
    class LineParams:
        c0: NDArray          # (3,) int64, fixed part of the centroid times q
        q: int               # > 0; centroid = c0 / q + B @ xi
        B: NDArray           # (3, n_xi) int64, free centroid directions (n_xi may be 0)
        U: NDArray | None    # (3, n_eta) int64, free axis directions; None for a single atom
        symmetric_ends: bool
    ```
  - `line_params(frac_coords, centroid_frac, Z, lattice, D, D_tr) -> LineParams | None`. Returns `None`, with a `RuntimeWarning` whose message contains `"treating it as non-linear"`, when the checks below fail.

Algorithm for `line_params`. These are the decisions an implementer cannot make alone:
1. **Centroid.** Stack the rows `[12 (I − h) | −t]` for every `(h, t)` in `(D, D_tr)` into a `(3|D|, 4)` integer matrix and take its `integer_nullspace`. Pick a row `v*` with `v*[3] != 0`. Replace every other row `w` with `v*[3]·w − w[3]·v*` and drop zero rows. Then `c0 = sign·v*[:3]`, `q = |v*[3]|`, and `B` = the remaining rows' first 3 entries, stacked as columns. If no row has `v*[3] != 0`, return `None`.
2. **Axis.** `u0` = fractional vector from the atom with the smallest projection onto the principal Cartesian direction to the atom with the largest. Skip this step for a single atom (`U = None`). For each `h` in `D`, set `ε = sign((A h u0)·(A u0))` and require `|A (h u0 − ε u0)| ≤ 1e-3·|A u0|`, otherwise return `None`. Then `U` = columns of `integer_nullspace` of the stacked `(h − ε I)` rows.
3. **Self-check.** The actual centroid must equal `c0/q + B ξ` and `u0` must equal `U η` for some real `ξ`, `η` (least-squares residual ≤ 1e-6 in fractional units), otherwise return `None`.

- [ ] **Step 1: Write the failing tests**

```python
def _ref_setup(name):
    cry = crystal(name).detect_symmetry(1e-3)
    mons = cry.get_references()
    context.bind(cry, mons)
    return cry, mons

def test_is_linear():
    _, (co2,) = _ref_setup("carbon_dioxide")
    _, (bz,) = _ref_setup("benzene")
    assert linear.is_linear(co2.cart_coords) and not linear.is_linear(bz.cart_coords)
    assert linear.is_linear(np.zeros((1, 3)))

def test_line_params_carbon_dioxide():
    cry, (co2,) = _ref_setup("carbon_dioxide")
    D, _, D_tr = context.dress_stacked_by_ref[0]
    p = linear.line_params(co2.frac_coords, co2.centroid_frac, _Z(co2),
                           np.asarray(cry.lattice_vectors, float), D, D_tr)
    assert p.q == 1 and not p.c0.any() and p.B.shape == (3, 0)
    assert p.U.shape == (3, 1) and abs(p.U[:, 0]).tolist() == [1, 1, 1] and len(set(p.U[:, 0])) == 1
    assert p.symmetric_ends

def test_line_params_general_position_polar(hcn_crystal):
    # one HCN per asymmetric unit on a general position of P2_1/c
    mons = hcn_crystal.get_references()
    context.bind(hcn_crystal, mons)
    D, _, D_tr = context.dress_stacked_by_ref[0]
    p = linear.line_params(mons[0].frac_coords, mons[0].centroid_frac, _Z(mons[0]),
                           np.asarray(hcn_crystal.lattice_vectors, float), D, D_tr)
    assert p.B.shape == (3, 3) and p.U.shape == (3, 3) and not p.symmetric_ends

def test_line_params_rejects_an_axis_the_dressings_do_not_keep():
    # a fake 4-fold about z tilts CO2's [111] axis, so the setup must refuse it
    cry, (co2,) = _ref_setup("carbon_dioxide")
    D, _, D_tr = context.dress_stacked_by_ref[0]
    four = np.array([[[0, -1, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.int64)
    with pytest.warns(RuntimeWarning, match="treating it as non-linear"):
        assert linear.line_params(co2.frac_coords, co2.centroid_frac, _Z(co2),
                                  np.asarray(cry.lattice_vectors, float),
                                  np.concatenate((D, four)),
                                  np.concatenate((D_tr, np.zeros((1, 3), np.int64)))) is None
```

Add two fixtures to `tests/test_linear.py`, both built with `Crystal((6.0, 7.0, 10.0), (90, 100, 90), SpaceGroup(14), ASU(...))`. Detection must stay off for both so the space group is kept:
- `hcn_crystal`: an HCN molecule with its centroid at fractional `(0.23, 0.31, 0.17)` and its axis along Cartesian `(0.48, 0.60, 0.64)` normalised. Bond lengths are H–C 1.07 Å and C–N 1.16 Å.
- `mixed_crystal`: the same HCN listed first in the ASU, plus a water molecule centred at `(0.71, 0.62, 0.58)` (O–H 0.96 Å, angle 104.5°).

Also add the helper `_Z(mon)`, copied from `tests/test_dressing.py`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_linear.py -v -k "linear or line_params"`
Expected: FAIL (`AttributeError: module ... has no attribute 'is_linear'`).

- [ ] **Step 3: Implement `is_linear`, `symmetric_ends`, `LineParams`, `line_params`**

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_linear.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/dedup/linear.py tests/test_linear.py
git commit -m "feat(dedup): derive centroid and axis unknowns"
```

---

### Task 3: Bind the setup and order anchors

**Files:**
- Modify: `src/crystalatte/work/dedup/context.py`
- Modify: `src/crystalatte/work/dedup/invariants.py` (`_group_keys` member sort; `orbit_labels` grouping)
- Test: `tests/test_linear.py`

**Interfaces:**
- Consumes: everything in `linear` (Tasks 1–2).
- Produces these module globals, set in `bind`:
  - `context.linear_by_ref: list[bool]`: `is_linear(...) and line_params(...) is not None`.
  - `context.line_by_ref: list[LineParams | None]`.
  - `context.metric_pieces: NDArray`: `metric_basis(H)`, computed from the `H` that `bind` already builds.
  - `context.anchor_rank: NDArray`: int64 `(r,)`, the rank of each reference when sorted by `(linear_by_ref[i], i)`, so non-linear references come first.
- `invariants.orbit_labels` groups clusters by `tuple(sorted(refs, key=anchor_rank.__getitem__))`. `_group_keys` orders members by `np.argsort(context.anchor_rank[ref_indices], kind="stable")`. Its `refs` argument is that ranked tuple, and `refs[0]` is still the anchor reference.

- [ ] **Step 1: Write the failing tests**

```python
def test_bind_records_linear_setup():
    _ref_setup("carbon_dioxide")
    assert context.linear_by_ref == [True] and context.line_by_ref[0].symmetric_ends
    assert context.metric_pieces.shape == (1, 3, 3)
    _ref_setup("benzene")
    assert context.linear_by_ref == [False] and context.line_by_ref == [None]

def test_anchor_rank_puts_nonlinear_first(mixed_crystal):
    context.bind(mixed_crystal, mixed_crystal.get_references())
    assert context.linear_by_ref == [True, False]   # precondition: HCN is reference 0
    assert context.anchor_rank.tolist() == [1, 0]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_linear.py -v -k "bind_records or anchor_rank"`
Expected: FAIL (`AttributeError: ... 'linear_by_ref'`).

- [ ] **Step 3: Implement the `bind` additions and the ranked member order**

- [ ] **Step 4: Run the whole suite and confirm nothing else changed**

Run: `pytest tests -x -q -m "not slow"`
Expected: PASS. For every fixture without a linear reference, `anchor_rank` is the identity, so all existing counts are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/dedup/context.py src/crystalatte/work/dedup/invariants.py tests/test_linear.py
git commit -m "feat(dedup): anchor mixed clusters on non-linear refs"
```

---

### Task 4: The dot-product key

**Files:**
- Create: `src/crystalatte/work/dedup/dotkey.py`
- Test: `tests/test_linear.py`

**Interfaces:**
- Consumes: `context.line_by_ref`, `context.metric_pieces`; `invariants.lex_codes`. Import it as `from . import invariants` and call `invariants.lex_codes` at runtime: `invariants` will import `dotkey`, so a `from`-import would be circular.
- Produces: `dot_keys(cands: list[SymOpList], refs: tuple[int, ...]) -> NDArray`, an int64 `(N, L)` array. Two clusters whose sorted `ref_indices` equal `refs` get equal rows iff some reference-compatible relabelling and end-flip gives equal dot-product formula tables.

Algorithm. The layout and the expansion trick are the decisions; vectorise over clusters and loop over relabellings.
1. **Scale and unknowns.** `S = 12 · lcm(q_i)` over the references in `refs`. Global unknowns are `[1, ξ⁽ⁱ⁾…, η⁽ⁱ⁾…]` with one block per distinct reference. Member `a` of reference `i` with placement `(R, t)` has:
   - centroid form `P_a` (3 × (1+Ξ)): constant column `(S/q_i) R c0_i + (S/12) t`, and `S · R B_i` in reference `i`'s ξ block.
   - axis form `U_a` (3 × Hη): `R U_i` in reference `i`'s η block. It has no columns if the member is a single atom.
2. **Pairwise products, once per cluster.** For each metric piece `G_k`: `PP[a,b] = P_aᵀ G_k P_b`, `PU[a,b] = P_aᵀ G_k U_b`, `UU[a,b] = U_aᵀ G_k U_b`.
3. **Per relabelling `(π, ε)`.** π permutes positions within each reference block; ε ∈ {±1} per member, fixed at +1 unless its reference has `symmetric_ends`. Offsets are `o_b = P_{πb} − P_{π0}` for b ≥ 1. Expand bilinearly, never re-multiplying matrices:
   - `o_b·o_c = PP[πb,πc] − PP[πb,π0] − PP[π0,πc] + PP[π0,π0]`
   - `o_b·u_c = ε_c (PU[πb,πc] − PU[π0,πc])`
   - `u_b·u_c = ε_b ε_c UU[πb,πc]`
4. **Collapse equal monomials.** For o·o and u·u entries, replace `C` with the upper triangle of `C + Cᵀ`, but keep the diagonal as `C_ii` (ξᵢξⱼ = ξⱼξᵢ). Leave o·u entries as they are.
5. **Row layout.** Concatenate, k-major: o·o for 1 ≤ b ≤ c; then o·u for all b ≥ 1 and members c that have an axis; then u·u for b ≤ c among members that have an axis. The layout depends only on `refs`.
6. **Canonical row.** Use `lex_codes` over the `(N, V, L)` stack, take the argmin over V, and return the winning rows.

- [ ] **Step 1: Write the failing tests**

```python
def _dimer(t1_twelfths, rot1=np.eye(3, dtype=np.int16)):
    return SymOpList.from_components(
        np.stack([np.eye(3, dtype=np.int16), rot1]),
        np.stack([np.zeros(3, np.int16), np.array(t1_twelfths, np.int16)]), 1, np.zeros(2, np.int64))

def test_dot_key_merges_the_surviving_co2_pair():
    _ref_setup("carbon_dioxide")
    k = dotkey.dot_keys([_dimer((-36, 0, 0)), _dimer((-24, -24, 12))], (0, 0))
    assert np.array_equal(k[0], k[1])

def test_dot_key_separates_equal_length_offsets_at_different_angles():
    _ref_setup("carbon_dioxide")
    k = dotkey.dot_keys([_dimer((12, 12, 12)), _dimer((12, 12, -12))], (0, 0))
    assert not np.array_equal(k[0], k[1])

def test_dot_key_ignores_member_order_and_inversion():
    _ref_setup("carbon_dioxide")
    a = _dimer((-36, 0, 0))
    swapped = SymOpList.from_components(a.rot_cache[::-1].copy(), a.tr_cache[::-1].copy(), 1, a.ref_indices)
    inverted = _dimer((-36, 0, 0), -np.eye(3, dtype=np.int16))    # second molecule flipped end to end
    k = dotkey.dot_keys([a, swapped, inverted], (0, 0))
    assert np.array_equal(k[0], k[1]) and np.array_equal(k[0], k[2])

def test_dot_key_agrees_with_every_existing_merge():
    # each existing-key class must have one dot key: the spec's consistency argument
    cry = crystal("carbon_dioxide").detect_symmetry(1e-3)
    refs = cry.get_references()
    cands = _candidates(cry, refs, R=15.0)   # capture via monkeypatched orbit_labels input
    frame = invariants.orbit_labels(cands)
    dk = dotkey.dot_keys(cands, (0, 0))
    for lab in np.unique(frame):
        rows = dk[frame == lab]
        assert (rows == rows[0]).all()

def test_dot_key_single_atoms(argon_crystal):
    context.bind(argon_crystal, argon_crystal.get_references())
    k = dotkey.dot_keys([_dimer((6, 6, 0)), _dimer((0, 6, 6)), _dimer((12, 0, 0))], (0, 0))
    assert k.shape[1] == 1                      # one o·o entry, no axis rows
    assert np.array_equal(k[0], k[1]) and not np.array_equal(k[0], k[2])
```

`argon_crystal` fixture: `Crystal((5.26,) * 3, (90,) * 3, SpaceGroup(225), ASU(["Ar"], np.zeros((1, 3))))`.

`_candidates` helper: monkeypatch `neighbors.orbit_labels` with a wrapper that records its argument and returns `np.arange(len(cands))`, then run `neighbors.generate_neighbors(cry, refs, np.array([[-2, 2]] * 3), R, 2, **GENERATE_KW)`. The bounds match the CO₂ R = 15 entry of `_REFEREE_BOX`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_linear.py -v -k dot_key`
Expected: FAIL (`ImportError: cannot import name 'dotkey'`).

- [ ] **Step 3: Implement `dot_keys` in `dotkey.py`**

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest tests/test_linear.py -v -k dot_key`
Expected: PASS (5).

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/dedup/dotkey.py tests/test_linear.py
git commit -m "feat(dedup): add exact dot-product key"
```

---

### Task 5: Route all-linear groups through the key

**Files:**
- Modify: `src/crystalatte/work/dedup/invariants.py` (`orbit_labels`)
- Modify: `tests/test_generate.py`, `tests/test_dedup.py`
- Test: `tests/test_linear.py`

**Interfaces:**
- Consumes: `dotkey.dot_keys` (Task 4), `context.linear_by_ref` (Task 3).
- Produces: `orbit_labels` keeps its signature. For a group where every reference in `refs` is in `linear_by_ref`, it:
  1. computes the existing frame labels,
  2. picks the first cluster of each frame class,
  3. computes `dot_keys` on those representatives only,
  4. labels every cluster by its representative's dot-key class.

- [ ] **Step 1: Write and update the failing tests**

In `tests/test_generate.py`, change the `test_extra_symmetry_regression` CO₂ row and its comment to:
```python
("carbon_dioxide", 2, 20.0, 80, 832, {2: 2, 6: 18, 12: 60}),
```

In `tests/test_dedup.py`:
- Rename `test_linear_monomer_dedup_is_complete` to `test_linear_monomer_dedup_is_complete_at_20`, run it at `R = 20.0`, and rewrite its comment to point at `dotkey`.
- Add `("carbon_dioxide", 2, 20.0)` (marked `slow`) and `("carbon_dioxide", 3, 10.0)` to `test_no_over_merge_every_discarded_cluster_matches_a_survivor`.
- Add `_REFEREE_BOX` entries: `("carbon_dioxide", 2, 20.0): 4`, `("carbon_dioxide", 3, 10.0): 2`.

In `tests/test_linear.py`:
```python
def test_co2_trimers_are_complete():
    mers = generate.generate(crystal("carbon_dioxide"), 3, 10.0, **GENERATE_KW)
    assert count_chemistry_classes(mers, atol=1e-8)[0] == len(mers)

def test_general_position_polar_linear_is_sound(hcn_crystal):
    mers = generate.generate(hcn_crystal, 2, 9.0, detect_symmetry=False, **GENERATE_KW)
    assert count_chemistry_classes(mers, atol=1e-8)[0] == len(mers)

def test_mixed_cocrystal_is_complete_and_sound(mixed_crystal):
    mers = generate.generate(mixed_crystal, 2, 9.0, detect_symmetry=False, **GENERATE_KW)
    assert count_chemistry_classes(mers, atol=1e-8)[0] == len(mers)

def test_single_atom_crystal_is_unchanged(argon_crystal):
    mers = generate.generate(argon_crystal, 2, 8.0, detect_symmetry=False, **GENERATE_KW)
    assert sorted(m.multiplicity for m in mers) == [6, 12, 12, 24]

def test_zprime_linear_references_lose_no_class():
    # CO2 expanded to P1 with detection off: 4 independent linear references
    sym = generate.generate(crystal("carbon_dioxide"), 2, 8.0, **GENERATE_KW)
    p1 = _expand_to_p1(crystal("carbon_dioxide"))
    mers = generate.generate(p1, 2, 8.0, detect_symmetry=False, **GENERATE_KW)
    assert count_chemistry_classes(mers, atol=1e-8)[0] == len(sym)
```

`_expand_to_p1`: build a P1 `Crystal` from the expanded cell, the same way `tests/test_disorder.py:47` does.

- [ ] **Step 2: Run them and confirm the CO₂ rows fail**

Run: `pytest tests/test_generate.py tests/test_dedup.py tests/test_linear.py -v -k "carbon_dioxide or linear or co2 or polar or mixed or single_atom or zprime"`
Expected: the CO₂ R = 20 rows FAIL with 81 ≠ 80. The rest may already pass.

- [ ] **Step 3: Implement the routing in `orbit_labels`**

- [ ] **Step 4: Run the full suite**

Run: `pytest tests -q` (include the slow tests once)
Expected: all PASS, with the existing ice xfails still xfail.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/dedup/invariants.py tests/
git commit -m "feat(dedup): merge all-linear clusters by dot key"
```

---

### Task 6: Remove the warning and update the docs

**Files:**
- Modify: `src/crystalatte/work/generate.py` (delete the linear `.. warning::` and the rank-test warning loop)
- Modify: `tests/test_dedup.py` (delete `test_linear_monomer_emits_a_warning` and `test_nonlinear_monomer_emits_no_warning`)
- Modify: `tests/README.md` (rewrite the "Linear monomers" bullet as resolved; the remaining xfails are ice only)
- Modify: `docs/dedup_correctness.tex` (section `sec:linear`: change the status line to "implemented" and drop "Planned:" from the title; Caveat C5: state that all-linear clusters use the dot-product key and that only parameter-specific coincidences are missed; update the abstract's caveat sentence)
- Modify: `src/crystalatte/TODO.md` (remove the "carbon dioxide, linear molecules dont work" line)

- [ ] **Step 1: Make the edits**

- [ ] **Step 2: Check that nothing still references the warning**

Run: `grep -rn "linear molecule" src tests`
Expected: no hits from `generate.py` and none in any test that checks for a warning.

- [ ] **Step 3: Build the doc**

Run: `latexmk -pdf -interaction=nonstopmode -halt-on-error -cd docs/dedup_correctness.tex`
Expected: exit 0; `grep -E "^!|undefined" docs/dedup_correctness.log` prints nothing.

- [ ] **Step 4: Run the suite**

Run: `pytest tests -q -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/generate.py tests/ docs/dedup_correctness.tex src/crystalatte/TODO.md
git commit -m "docs(dedup): mark linear monomers as supported"
```
