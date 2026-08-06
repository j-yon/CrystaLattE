# Exact-Integer Multimer Deduplication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two `_is_bijection` branches with one exact-integer relative-transform matcher (dressing = crystallographic site symmetry) plus a gated float fallback that catches all remaining congruences.

**Architecture:** Relative transforms `Δ_ab = T_a⁻¹ T_b` are computed in the integer lattice basis (`rot_a⁻¹ rot_b`, `rot_a⁻¹(tr_b−tr_a)` in twelfths). An anchored search (enumerate node-0 image + dressing, force the permutation, solve each per-node dressing, verify all pairs) decides equivalence with `np.array_equal`. The integer matcher, dressed by the site-symmetry group `W`, is complete for all crystallographic merges; a float `_same_multimer` with the full molecular self-isometry group runs only when the integer check fails and `S ⊋ W`.

**Tech Stack:** Python 3, NumPy. Tests via `pytest` under the conda env `cle`.

## Global Constraints

- Fractional translations are stored as integers in **twelfths** (`tr/12`). Keep integer arithmetic in these units.
- Use `np.int64` for all relative-transform products (avoid `int16` overflow).
- Integer matrix inverse is the **exact adjugate** via `np.rint(np.linalg.inv(M)).astype(np.int64)` — never `.T` (lattice-basis rotations are not orthogonal). Do NOT modify the existing `SymOpList.invert_list` (pinned for separate cleanup).
- Run tests with: `conda run -n cle python -m pytest Tests/<file> -v` — note the **capital `Tests/`** path; `pyproject.toml` sets `testpaths = ["Tests"]`, and pytest will not collect a lowercase `tests/...` argument even though macOS resolves the directory case-insensitively.
- **Space-group data:** tests may only use space groups present in the tracked `src/crystalatte/core/space_groups.json` — namely **1 (P1), 4 (P2₁), 61 (Pbca)**. The full table (`space_groups_all.json`) is an untracked local WIP file, so tests must not depend on it. Use **SG 61 (Pbca, centrosymmetric, benzene)** for the inversion/site-symmetry cases; construct `SymOpList`s directly (no data file) for the matcher unit tests.
- **Commit scoping:** the repo carries unrelated pre-existing WIP in several files. This feature builds on a committed baseline that already contains the dedup pipeline (`_is_bijection`, `_same_multimer`, `_relative_transforms`, `_distance_fingerprint`, `_fingerprints_match`, and the `SymOpList` geometry caches). Stage only this feature's changes; do not `git add` whole pre-modified files blindly.
- The dedup target is **energetic equivalence**: the float fallback uses the full molecular `S`, deliberately merging all congruences (proper molecular-symmetry congruences and mirror-image enantiomers), not enantiomers only.

---

### Task 1: `SpaceGroup.get_site_symmetry`

**Files:**
- Modify: `src/crystalatte/core/space_group.py` (add method to `SpaceGroup`, after `get_unique_sym_ops`, ~line 164)
- Test: `tests/test_dedup.py` (create)

**Interfaces:**
- Produces: `SpaceGroup.get_site_symmetry(coords: NDArray) -> list[SymOp]` — the ops fixing `coords` modulo the lattice (the site-symmetry stabilizer `W`). Always includes the identity.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dedup.py`:

```python
import numpy as np

from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp, SymOpList
from crystalatte.work import generate


def test_get_site_symmetry_inversion_center():
    # Pbca (space group 61): origin is an inversion centre => {E, i}.
    # SG 61 is in the tracked space_groups.json, so no dependency on the
    # untracked full table.
    sg = SpaceGroup(61)
    w0 = sg.get_site_symmetry(np.array([0.0, 0.0, 0.0]))
    rots = [op.rot for op in w0]
    assert len(w0) == 2
    assert any(np.array_equal(r, np.eye(3, dtype=r.dtype)) for r in rots)
    assert any(np.array_equal(r, -np.eye(3, dtype=r.dtype)) for r in rots)


def test_get_site_symmetry_general_position():
    sg = SpaceGroup(61)
    wg = sg.get_site_symmetry(np.array([0.13, 0.21, 0.37]))
    assert len(wg) == 1
    assert np.array_equal(wg[0].rot, np.eye(3, dtype=wg[0].rot.dtype))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_get_site_symmetry_inversion_center -v`
Expected: FAIL with `AttributeError: 'SpaceGroup' object has no attribute 'get_site_symmetry'`

- [ ] **Step 3: Write minimal implementation**

Add to `SpaceGroup` in `src/crystalatte/core/space_group.py`:

```python
    def get_site_symmetry(self, coords: NDArray) -> list[SymOp]:
        """Return the site-symmetry stabilizer of ``coords``: the space-group
        operations that map ``coords`` onto itself modulo a lattice translation.

        :param coords: Fractional coordinates of the monomer centroid
        :returns: The subset of symmetry operations fixing ``coords`` (always
            includes the identity)
        """
        coords = np.asarray(coords, dtype=float)
        result: list[SymOp] = []
        for op in self._sym_ops:
            delta = op.apply(coords) - coords
            if np.allclose(delta - np.round(delta), 0.0, atol=1e-6):
                result.append(op)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py -v`
Expected: PASS (both site-symmetry tests)

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/space_group.py tests/test_dedup.py
git commit -m "feat: add SpaceGroup.get_site_symmetry for site-symmetry stabilizer"
```

---

### Task 2: Integer relative transforms + cache field

**Files:**
- Modify: `src/crystalatte/core/sym_ops.py` (add `_rel_int_cache` field to `SymOpList`; init in `from_components`)
- Modify: `src/crystalatte/work/generate.py` (add `_int_inv` and `_relative_transforms_int`)
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `SymOpList.rot_cache`, `SymOpList.tr_cache` (integer arrays).
- Produces:
  - `generate._int_inv(M: NDArray) -> NDArray` — exact integer inverse of a unimodular integer matrix (`int64`).
  - `generate._relative_transforms_int(g_N: SymOpList) -> tuple[NDArray, NDArray]` — `(L, v)` with shapes `(n, n, 3, 3)` and `(n, n, 3)`, integer (`v` in twelfths). Cached on `g_N._rel_int_cache`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dedup.py`:

```python
def _translate(dx12):
    return SymOp.identity().translate(np.array(dx12, dtype=np.int16))


def test_relative_transforms_int_pure_translation():
    g = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    L, v = generate._relative_transforms_int(g)
    assert np.array_equal(L[0, 1], np.eye(3, dtype=np.int64))
    assert np.array_equal(v[0, 1], np.array([12, 0, 0]))
    assert np.array_equal(v[1, 0], np.array([-12, 0, 0]))


def test_relative_transforms_int_nonorthogonal_inverse():
    # hexagonal 6-fold: rot = [[1,-1,0],[1,0,0],[0,0,1]] is NOT orthogonal;
    # its inverse differs from its transpose.
    six = SymOp("x-y,x,z")
    g = SymOpList([SymOp.identity(), six], 1)
    L, v = generate._relative_transforms_int(g)
    R = six.rot.astype(np.int64)
    # L[1,0] = rot_inv[1] @ rot[0] = inv(R); verify it is the exact inverse
    assert np.array_equal(L[1, 0] @ R, np.eye(3, dtype=np.int64))
    # and that the wrong (transpose) formula would have differed
    assert not np.array_equal(L[1, 0], R.T)


def test_relative_transforms_int_is_cached():
    g = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    first = generate._relative_transforms_int(g)
    second = generate._relative_transforms_int(g)
    assert first[0] is second[0]  # same cached array object
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_relative_transforms_int_pure_translation -v`
Expected: FAIL with `AttributeError: module 'crystalatte.work.generate' has no attribute '_relative_transforms_int'`

- [ ] **Step 3a: Add the cache field to `SymOpList`**

In `src/crystalatte/core/sym_ops.py`, next to `_rel_cache` (in the `SymOpList` dataclass field block):

```python
    _rel_int_cache: tuple[NDArray, NDArray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
```

And in `SymOpList.from_components` (the `__new__` path), next to `instance._rel_cache = None`:

```python
        instance._rel_int_cache = None
```

- [ ] **Step 3b: Add the integer helpers to `generate.py`**

In `src/crystalatte/work/generate.py`, after `_relative_transforms` (~line 138):

```python
def _int_inv(M: NDArray) -> NDArray:
    """Return the exact integer inverse of a unimodular integer matrix.

    Lattice-basis rotations are integer with ``det = ±1`` but are not
    orthogonal, so the inverse is recovered via the adjugate (``round`` of the
    float inverse), never the transpose.

    :param M: A ``(3, 3)`` integer matrix with ``det = ±1``
    :returns: The exact ``(3, 3)`` ``int64`` inverse
    """
    return np.rint(np.linalg.inv(M)).astype(np.int64)


def _relative_transforms_int(g_N: SymOpList) -> tuple[NDArray, NDArray]:
    """Return the pairwise relative transforms of the cluster in the integer
    lattice basis: ``L[a,b] = rot_a^-1 rot_b`` (integer) and
    ``v[a,b] = rot_a^-1 (tr_b - tr_a)`` (integer, in twelfths).

    Exact at all distances (no Cartesian conversion, no roundoff). Cached on the
    :class:`~crystalatte.core.sym_ops.SymOpList` since the matcher re-reads them
    across many comparisons.

    :param g_N: The cluster whose integer relative transforms are requested
    :returns: A tuple ``(L, v)`` of shapes ``(n, n, 3, 3)`` and ``(n, n, 3)``
    """
    cached = getattr(g_N, "_rel_int_cache", None)
    if cached is not None:
        return cached
    rot = g_N.rot_cache.astype(np.int64)          # (n, 3, 3)
    tr = g_N.tr_cache.astype(np.int64)            # (n, 3), twelfths
    rot_inv = np.empty_like(rot)
    for i in range(len(rot)):
        rot_inv[i] = _int_inv(rot[i])
    L = np.einsum("aij,bjk->abik", rot_inv, rot)           # (n, n, 3, 3)
    dt = tr[None, :, :] - tr[:, None, :]                   # (n, n, 3): tr_b - tr_a
    v = np.einsum("aij,abj->abi", rot_inv, dt)            # (n, n, 3)
    g_N._rel_int_cache = (L, v)
    return L, v
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py -v`
Expected: PASS (all Task 1 + Task 2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/sym_ops.py src/crystalatte/work/generate.py tests/test_dedup.py
git commit -m "feat: integer lattice-basis relative transforms for multimer dedup"
```

---

### Task 3: `_same_multimer_int` exact-integer matcher

**Files:**
- Modify: `src/crystalatte/work/generate.py` (add module globals `_W_lin`, `_check_congruence`; add `_same_multimer_int`)
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `generate._relative_transforms_int`, `generate._int_inv`, module global `generate._W_lin` (list of integer `(3,3)` dressing matrices).
- Produces: `generate._same_multimer_int(g_N: SymOpList, h_N: SymOpList) -> bool` — True iff the clusters are the same multimer under a crystallographic isometry, modding out the site-symmetry dressing `_W_lin`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dedup.py`:

```python
def test_same_multimer_int_translation_swap():
    # Same dimer counted from each monomer: [I, T(+1)] ~ [I, T(-1)]. W = {I}.
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)
    assert generate._same_multimer_int(G, H) is True


def test_same_multimer_int_non_match():
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)
    assert generate._same_multimer_int(G, H) is False


def test_same_multimer_int_inversion_needs_W():
    # G = [I, T(+1)];  H = [I, i0.T(+1)] with partner rot = -I, tr = (-1,0,0).
    # Same multimer only via s0 = -I, so it merges iff -I is in the dressing W.
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    h_partner = SymOp.from_components(
        -np.eye(3, dtype=np.int16), np.array([-12, 0, 0], dtype=np.int16)
    )
    H = SymOpList([SymOp.identity(), h_partner], 1)

    generate._W_lin = [np.eye(3, dtype=np.int64)]
    assert generate._same_multimer_int(G, H) is False

    generate._W_lin = [np.eye(3, dtype=np.int64), -np.eye(3, dtype=np.int64)]
    assert generate._same_multimer_int(G, H) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_same_multimer_int_translation_swap -v`
Expected: FAIL with `AttributeError: module 'crystalatte.work.generate' has no attribute '_same_multimer_int'`

- [ ] **Step 3a: Declare the new module globals**

In `src/crystalatte/work/generate.py`, in the module-global block (~lines 48-53), add:

```python
_W_lin: list[NDArray] = [np.eye(3, dtype=np.int64)]  # integer site-symmetry dressings
_check_congruence: bool = False  # True iff molecular S is richer than site symmetry W
```

- [ ] **Step 3b: Implement `_same_multimer_int`**

Add to `src/crystalatte/work/generate.py` (after `_relative_transforms_int`):

```python
def _same_multimer_int(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff ``g_N`` and ``h_N`` are the same multimer under a
    crystallographic isometry, working entirely in the exact integer lattice
    basis and modding out the site-symmetry dressing ``_W_lin``.

    This is the exact-integer form of the anchored relative-transform search
    (see ``docs/dedup_correctness.tex``, Thm 2). It subsumes the old exact
    matrix branch and is complete for every same-orbit (multiplicity-required)
    merge, including centrosymmetric site-inversion cases. Non-crystallographic
    congruences (dressing outside ``_W_lin``) are left to the float fallback in
    :func:`_is_bijection`.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :returns: True iff the clusters are the same multimer via a crystallographic
        isometry
    """
    n = len(g_N)
    if n != len(h_N):
        return False

    LG, vG = _relative_transforms_int(g_N)
    LH, vH = _relative_transforms_int(h_N)
    S = _W_lin

    def in_S(M: NDArray) -> bool:
        return any(np.array_equal(M, s) for s in S)

    def verify(pi: list[int], s: list[NDArray]) -> bool:
        for a in range(n):
            sa_inv = _int_inv(s[a])
            for b in range(n):
                if not np.array_equal(vG[a, b], sa_inv @ vH[pi[a], pi[b]]):
                    return False
                if not np.array_equal(
                    LG[a, b], sa_inv @ LH[pi[a], pi[b]] @ s[b]
                ):
                    return False
        return True

    for h0 in range(n):
        for s0 in S:
            pi = [-1] * n
            s: list[NDArray] = [None] * n
            used = [False] * n
            pi[0], s[0], used[h0] = h0, s0, True

            def assign(b: int) -> bool:
                if b == n:
                    return verify(pi, s)
                target = s0 @ vG[0, b]  # want vH[h0, pi(b)] == target
                for h in range(n):
                    if used[h] or not np.array_equal(vH[h0, h], target):
                        continue
                    sb = _int_inv(LH[h0, h]) @ s0 @ LG[0, b]
                    if not in_S(sb):
                        continue
                    pi[b], s[b], used[h] = h, sb, True
                    if assign(b + 1):
                        return True
                    pi[b], s[b], used[h] = -1, None, False
                return False

            if assign(1):
                return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py -v`
Expected: PASS (all tests through Task 3)

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_dedup.py
git commit -m "feat: exact-integer _same_multimer_int with site-symmetry dressing"
```

---

### Task 4: Bind `_W_lin` / `_check_congruence` in `_bind_globals`

**Files:**
- Modify: `src/crystalatte/work/generate.py` (`_bind_globals`, ~lines 56-77)
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `SpaceGroup.get_site_symmetry` (Task 1), `Monomer.self_isometries`, `Monomer.centroid_frac`.
- Produces: after `_bind_globals(crystal, monomer)`, module globals `generate._W_lin` (deduped integer site-symmetry `rot` parts, includes `I`) and `generate._check_congruence` (`len(_self_iso) > len(_W_lin)`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dedup.py` (uses the benzene fixture; benzene sits on an inversion centre, so `W = {E, i}` and its molecular symmetry is richer):

```python
from pathlib import Path

import crystalatte as cle

DATA = Path(__file__).parent


def test_bind_globals_sets_site_symmetry_and_gate():
    crystal, monomer = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    generate._bind_globals(crystal, monomer)

    # Benzene is on an inversion centre: identity + inversion in the dressing.
    assert any(np.array_equal(w, np.eye(3, dtype=np.int64)) for w in generate._W_lin)
    assert any(np.array_equal(w, -np.eye(3, dtype=np.int64)) for w in generate._W_lin)
    # Molecular point group (D6h-ish) is richer than the site symmetry.
    assert generate._check_congruence is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_bind_globals_sets_site_symmetry_and_gate -v`
Expected: FAIL with `AttributeError` on `generate._W_lin` not being populated by `_bind_globals` (still the default), i.e. the `-I` assertion fails.

- [ ] **Step 3: Extend `_bind_globals`**

In `src/crystalatte/work/generate.py`, update the `global` line and body of `_bind_globals`:

```python
    global _crystal, _monomer, _self_iso, _Z_per_monomer, _A, _A_inv
    global _W_lin, _check_congruence
    _crystal = crystal
    _monomer = monomer
    _self_iso = monomer.self_isometries
    _Z_per_monomer = np.array(
        [qcel.periodictable.to_Z(s) for s in monomer.symbols], dtype=np.int64
    )
    _A = np.asarray(crystal.lattice_vectors, dtype=float)
    _A_inv = np.linalg.inv(_A)

    # Crystallographic site-symmetry dressing (integer, lattice basis) for the
    # exact-integer matcher, plus the fallback gate (`S` richer than `W`).
    site_ops = crystal.space_group.get_site_symmetry(monomer.centroid_frac)
    _W_lin = []
    for op in site_ops:
        r = op.rot.astype(np.int64)
        if not any(np.array_equal(r, w) for w in _W_lin):
            _W_lin.append(r)
    if not _W_lin:  # defensive: identity always fixes the centroid
        _W_lin = [np.eye(3, dtype=np.int64)]
    _check_congruence = len(_self_iso) > len(_W_lin)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_bind_globals_sets_site_symmetry_and_gate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_dedup.py
git commit -m "feat: bind site-symmetry dressing and congruence gate in _bind_globals"
```

---

### Task 5: Rewrite `_is_bijection` (3 stages) + fallback gating + regression

**Files:**
- Modify: `src/crystalatte/work/generate.py` (`_is_bijection`, ~lines 293-333; module docstring note)
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `_fingerprints_match`, `_same_multimer_int`, `_same_multimer`, `_check_congruence`.
- Produces: `_is_bijection(g_N, h_N) -> bool` with the fingerprint gate, then the integer matcher, then a `_check_congruence`-gated float fallback.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dedup.py`:

```python
def test_is_bijection_fallback_gated_by_check_congruence(monkeypatch):
    # Two clusters that the integer matcher rejects; the float fallback must run
    # only when _check_congruence is True.
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([24, 0, 0])], 1)

    calls = []
    monkeypatch.setattr(generate, "_fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(
        generate, "_same_multimer", lambda a, b, **k: calls.append(1) or False
    )

    generate._check_congruence = False
    assert generate._is_bijection(G, H) is False
    assert calls == []  # gated off: float fallback not called

    generate._check_congruence = True
    assert generate._is_bijection(G, H) is False
    assert calls == [1]  # gated on: float fallback called once


def test_is_bijection_integer_path_matches_without_fallback(monkeypatch):
    # A crystallographic merge must be decided by the integer path alone.
    generate._W_lin = [np.eye(3, dtype=np.int64)]
    generate._check_congruence = True
    G = SymOpList([SymOp.identity(), _translate([12, 0, 0])], 1)
    H = SymOpList([SymOp.identity(), _translate([-12, 0, 0])], 1)

    def _boom(*a, **k):
        raise AssertionError("float fallback should not run for integer merges")

    monkeypatch.setattr(generate, "_fingerprints_match", lambda a, b, **k: True)
    monkeypatch.setattr(generate, "_same_multimer", _boom)
    assert generate._is_bijection(G, H) is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_is_bijection_fallback_gated_by_check_congruence -v`
Expected: FAIL — current `_is_bijection` runs the old branch (a)/`_same_multimer` unconditionally, so the gating assertions (`calls == []`) do not hold.

- [ ] **Step 3: Rewrite `_is_bijection`**

Replace the body of `_is_bijection` in `src/crystalatte/work/generate.py` (remove the `composed`/`aug_cache` branch-(a) block entirely):

```python
def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    """Return True iff ``g_N`` and ``h_N`` represent the same multimer.

    Three stages, cheapest first:

    0. Distance-fingerprint gate (:func:`_fingerprints_match`) — a cached,
       vectorized necessary condition rejecting most within-bucket pairs.
    1. Exact-integer matcher (:func:`_same_multimer_int`) — the complete,
       roundoff-free decision for every crystallographic (same-orbit) merge,
       modding out the site symmetry. Subsumes the former exact matrix branch.
    2. Float congruence fallback (:func:`_same_multimer`) — runs only when the
       integer matcher fails and the molecular symmetry is richer than the site
       symmetry (``_check_congruence``). It uses the FULL molecular
       self-isometry group, so it merges every energetically-equivalent
       congruence: proper molecular-symmetry congruences AND mirror-image
       enantiomers. This is intentional and goes beyond an enantiomer-only
       check; the dedup target is energetic equivalence.

    :param g_N: The first cluster
    :param h_N: The second cluster
    :returns: True iff the clusters represent the same multimer
    """
    if not _fingerprints_match(g_N, h_N):
        return False
    if _same_multimer_int(g_N, h_N):
        return True
    if _check_congruence:
        return _same_multimer(g_N, h_N)
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py -v`
Expected: PASS (all dedup unit tests)

- [ ] **Step 5: Capture the regression baseline from current behavior**

Use **benzene only** (SG 61 Pbca — the one non-trivial space group in the tracked `space_groups.json`). Benzene is centrosymmetric, so it exercises the integer path (inversion merges via `W = {E, i}`) *and* the float fallback (`_check_congruence` is True, so proper molecular-symmetry congruences route to `_same_multimer`). Ammonia (SG 198) is intentionally dropped — it needs the untracked full table; the fallback's invocation logic is already covered data-free by the gating test in Step 1.

Run (record the printed count):
```bash
conda run -n cle python -c "
import crystalatte as cle
from pathlib import Path
crystal, mon = next(cle.from_cif(str(Path('tests/x23/benzene/benzene.cif'))))
mers = cle.work.generate.generate(crystal, mon, 2, 5.0, use_origin=True, n_jobs=1)
print('benzene dimers:', len(mers))
"
```
Note the printed count. (It is unchanged by this refactor — Option C is behavior-preserving — so it is a valid post-change target.)

- [ ] **Step 6: Write the regression test with the observed count**

Append to `tests/test_dedup.py`, substituting the integer observed in Step 5 for `BENZENE_DIMERS`:

```python
def test_generate_dimer_count_regression():
    crystal, monomer = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    mers = generate.generate(crystal, monomer, 2, 5.0, use_origin=True, n_jobs=1)
    assert len(mers) == BENZENE_DIMERS   # from Step 5
```

- [ ] **Step 7: Run the regression test**

Run: `conda run -n cle python -m pytest Tests/test_dedup.py::test_generate_dimer_count_regression -v`
Expected: PASS

- [ ] **Step 8: Run the full suite to confirm no regressions**

Run: `conda run -n cle python -m pytest Tests/ -v`
Expected: PASS (or unchanged from the pre-existing baseline — note any pre-existing failures unrelated to this change)

- [ ] **Step 9: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_dedup.py
git commit -m "feat: unify _is_bijection on exact-integer matcher with gated float fallback"
```

---

## Self-Review

**Spec coverage:**
- `get_site_symmetry` → Task 1. ✓
- integer `(L, v)` + `_rel_int_cache` + adjugate inverse → Task 2. ✓
- `_same_multimer_int` anchored search with `W` dressing → Task 3. ✓
- `_W_lin` / `_check_congruence` bound in `_bind_globals` (workers inherit via `_init_bucket_worker`) → Task 4. ✓
- `_is_bijection` 3-stage rewrite, branch-(a) deletion, doc note → Task 5. ✓
- Float `_same_multimer` / `_relative_transforms` retained as fallback → unchanged, exercised in Task 5. ✓
- Regression on fixtures → Task 5 Steps 5-8. ✓
- Out-of-scope items (bucketing, neighbor gen, `get_unique_sym_ops`, multiprocessing) → untouched. ✓

**Placeholder scan:** The only runtime-observed values are the two regression counts (`BENZENE_DIMERS`, `AMMONIA_DIMERS`), captured by the explicit command in Step 5 and substituted in Step 6 — a data-dependent baseline, not a vague instruction.

**Type consistency:** `_W_lin` is a `list[NDArray]` of `int64 (3,3)` throughout; `_relative_transforms_int` returns `(L, v)` used with matching indices in `_same_multimer_int`; `_int_inv` returns `int64`; `get_site_symmetry` returns `list[SymOp]` and `_bind_globals` reads `.rot` from each — consistent across Tasks 1-5.
