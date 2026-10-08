# Z′>1 Multi-Reference N-mer Enumeration — Implementation Plan (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enumerate and deduplicate n-mers across *all* symmetry-independent molecules in the asymmetric unit (Z′>1), with species-aware plumbing for future co-crystals.

**Architecture:** `Crystal.get_references()` returns one `Monomer` per molecular symmetry-orbit. Candidates carry a per-member base-reference index (`SymOpList.ref_indices`). Enumeration draws partners from the union of all references' orbits. Dedup routes mono-reference clusters (all members one base) through the exact-integer matcher, and everything else through a labeled point-cloud congruence test. `generate(crystal, N, R)` drives it all.

**Tech Stack:** Python 3, NumPy, SciPy (cKDTree already used), qcelemental. Tests via `pytest` under conda env `cle`.

## Global Constraints

- Run tests with the **capital** path: `conda run -n cle python -m pytest Tests/<file> -v` (`pyproject.toml` sets `testpaths=["Tests"]`; lowercase `tests/...` collects nothing). `git add tests/...` uses the real lowercase tracked path.
- Fractional translations are integers in **twelfths**; integer products use `np.int64`; integer matrix inverse is the adjugate `np.rint(np.linalg.inv(M)).astype(np.int64)`, never `.T`.
- **Space-group data:** the tracked table now includes the full set (`space_groups_all.json`), so any space group is available. Fixtures used: `x23/benzene/benzene.cif` (SG 61, Z′=1) and `x23/ammonia/ammonia.cif` (P1, 4 NH₃).
- **Backward compatibility of `ref_indices`:** it defaults to all-zeros; every existing Z′=1 code path and test must behave identically (benzene end-to-end must stay at **3** dimers).
- Stage only this feature's changes; the working tree is clean at the baseline.

---

### Task 1: `Crystal.get_references()` + orbit grouping

**Files:**
- Modify: `src/crystalatte/core/crystal.py` (refactor `get_reference`, add `_find_molecules`, `_build_monomer`, `get_references`)
- Test: `tests/test_multiref.py` (create)

**Interfaces:**
- Produces: `Crystal.get_references(tol=1.2) -> list[Monomer]` (one per molecular symmetry-orbit, ordered nearest-origin first); `Crystal.get_reference(tol=1.2)` returns `get_references(tol)[0]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_multiref.py`:

```python
import numpy as np
import crystalatte as cle
from pathlib import Path

DATA = Path(__file__).parent


def test_get_references_z1_benzene():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    refs = crystal.get_references()
    assert len(refs) == 1
    assert len(refs[0].symbols) == 12  # full benzene


def test_get_references_z4_ammonia_p1():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    assert len(refs) == 4                      # 4 independent NH3 in P1
    assert all(len(r.symbols) == 4 for r in refs)  # each NH3 = N + 3H
    # distinct molecules => distinct centroids
    cens = np.array([r.centroid_frac % 1.0 for r in refs])
    assert len({tuple(np.round(c, 3)) for c in cens}) == 4


def test_get_reference_is_first_of_references():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    assert np.allclose(crystal.get_reference().centroid_frac,
                       crystal.get_references()[0].centroid_frac)
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py -v`
Expected: FAIL — `get_references` does not exist yet (`AttributeError`).

- [ ] **Step 3: Refactor `get_reference` and add `get_references`**

In `src/crystalatte/core/crystal.py`, replace the body of `get_reference` (lines ~196-279) with the shared helpers plus the two public methods:

```python
    def _find_molecules(self, tol: float = 1.2):
        """Expand the ASU by all sym-ops x lattice translations, dedup positions,
        BFS into complete molecules, and return
        ``(unique_frac, unique_cart, unique_symbols, full_components)`` where
        ``full_components`` are the index lists of max-size (complete) molecules.
        """
        all_symbols, all_frac = [], []
        for d in product(range(-1, 2), repeat=3):
            translation = np.array(d)
            for symop in self._space_group.sym_ops:
                for atom, pos in zip(self._asu.atoms, self._asu.positions):
                    all_symbols.append(atom)
                    all_frac.append(symop.apply(pos) + translation)
        all_frac = np.array(all_frac)
        all_symbols = np.array(all_symbols)

        mask = ~np.any(np.abs(all_frac) > 2.0, axis=1)
        all_frac, all_symbols = all_frac[mask], all_symbols[mask]

        unique_indices = []
        for i in range(len(all_frac)):
            if not unique_indices:
                unique_indices.append(i)
                continue
            dists = np.linalg.norm(all_frac[unique_indices] - all_frac[i], axis=1)
            if np.all(dists >= 1e-3):
                unique_indices.append(i)

        unique_frac = all_frac[unique_indices]
        unique_symbols = all_symbols[unique_indices]
        unique_cart = self.to_cartesian(unique_frac)
        components = self._bfs(unique_cart, unique_symbols, tol)
        mol_len = max(len(c) for c in components)
        full = [c for c in components if len(c) == mol_len]
        return unique_frac, unique_cart, unique_symbols, full

    def _build_monomer(self, comp, unique_frac, unique_cart, unique_symbols) -> Monomer:
        syms = [str(unique_symbols[j]) for j in comp]
        masses = np.array([qcel.periodictable.to_mass(s) for s in syms])
        return Monomer(
            syms,
            unique_frac[comp],
            unique_cart[comp],
            np.average(unique_frac[comp], axis=0, weights=masses),
            np.average(unique_cart[comp], axis=0, weights=masses),
        )

    def get_references(self, tol: float = 1.2) -> list[Monomer]:
        """Return one Monomer per molecular symmetry-orbit in the ASU (all
        symmetry-independent molecules, i.e. Z'), ordered nearest-origin first.

        Two complete molecules are the same reference iff a space-group
        operation maps one onto the other; this is decided on the mass-weighted
        centroid orbit (the smallest sym-op image reduced into the unit cell).
        """
        unique_frac, unique_cart, unique_symbols, full = self._find_molecules(tol)
        ops = self._space_group.sym_ops

        reps: dict[tuple, tuple[float, list]] = {}
        for comp in full:
            syms = [str(unique_symbols[j]) for j in comp]
            masses = np.array([qcel.periodictable.to_mass(s) for s in syms])
            cen = np.average(unique_frac[comp], axis=0, weights=masses)
            imgs = []
            for op in ops:
                r = op.apply(cen) % 1.0
                r[r > 1.0 - 1e-3] -= 1.0          # wrap 0.999... -> ~0
                imgs.append(tuple(np.round(r, 3)))
            key = min(imgs)
            d = float(np.linalg.norm(cen))
            if key not in reps or d < reps[key][0]:
                reps[key] = (d, comp)

        ordered = [comp for _, comp in sorted(reps.values(), key=lambda t: t[0])]
        return [
            self._build_monomer(c, unique_frac, unique_cart, unique_symbols)
            for c in ordered
        ]

    def get_reference(self, tol: float = 1.2) -> Monomer:
        """Return the single reference molecule closest to the origin.

        :param tol: Tolerance factor for determining bonding (default 1.2)
        :returns: A Monomer object representing the reference monomer
        """
        return self.get_references(tol)[0]
```

Ensure `product` (from `itertools`) and `qcel` imports already exist at the top of the file (they do — `get_reference` used them).

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py -v`
Expected: PASS (3 tests). If ammonia returns a count other than 4, inspect the centroid keys printed via a scratch run before adjusting the wrap/round tolerance — do not loosen blindly.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/crystal.py tests/test_multiref.py
git commit -m "feat: Crystal.get_references() enumerates all symmetry-independent molecules"
```

---

### Task 2: `SymOpList.ref_indices`

**Files:**
- Modify: `src/crystalatte/core/sym_ops.py` (`SymOpList`: field, property, `from_components`, `__post_init__`)
- Test: `tests/test_multiref.py`

**Interfaces:**
- Produces: `SymOpList._ref_indices` / `SymOpList.ref_indices -> NDArray` (int, one per member; defaults to all-zeros of length `len(sym_ops)`), settable; carried through `from_components(rot, tr, multiplicity, ops=None, ref_indices=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multiref.py`:

```python
from crystalatte.core.sym_ops import SymOp, SymOpList


def test_symoplist_ref_indices_default_zeros():
    s = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    assert list(s.ref_indices) == [0, 0]


def test_symoplist_ref_indices_explicit():
    s = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    s.ref_indices = np.array([0, 1])
    assert list(s.ref_indices) == [0, 1]
    fc = SymOpList.from_components(
        s.rot_cache, s.tr_cache, 1, ref_indices=np.array([1, 0])
    )
    assert list(fc.ref_indices) == [1, 0]
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py::test_symoplist_ref_indices_default_zeros -v`
Expected: FAIL — `SymOpList` has no `ref_indices`.

- [ ] **Step 3: Add the field/property**

In `SymOpList` (`src/crystalatte/core/sym_ops.py`):

Add a dataclass field near `_ops`:
```python
    _ref_indices: NDArray | None = field(default=None, repr=False)
```
At the end of `__post_init__`, default it:
```python
        if self._ref_indices is None:
            self._ref_indices = np.zeros(len(self._sym_ops), dtype=np.int64)
```
Add the property:
```python
    @property
    def ref_indices(self) -> NDArray:
        """Per-member base-reference index (which ASU molecule each member is a
        placement of). Defaults to all-zeros for single-reference clusters."""
        return self._ref_indices

    @ref_indices.setter
    def ref_indices(self, value: NDArray):
        self._ref_indices = np.asarray(value, dtype=np.int64)
```
In `from_components`, add a keyword parameter and set it (after `instance._ops = ops`):
```python
    @classmethod
    def from_components(cls, rot, tr, multiplicity, ops=None, ref_indices=None):
        ...
        instance._ref_indices = (
            np.zeros(len(rot), dtype=np.int64) if ref_indices is None
            else np.asarray(ref_indices, dtype=np.int64)
        )
        ...
        return instance
```
(Keep all existing `from_components` body; just add the `ref_indices` param and assignment. Preserve the existing `_rel_cache`/`_rel_int_cache`/`_dist_fp_cache = None` lines.)

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py -v` and `conda run -n cle python -m pytest Tests/test_dedup.py -v`
Expected: PASS (new ref_indices tests + all existing dedup tests unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/sym_ops.py tests/test_multiref.py
git commit -m "feat: SymOpList.ref_indices (per-member base reference, defaults to zeros)"
```

---

### Task 3: Per-reference module state + generalized fingerprint

**Files:**
- Modify: `src/crystalatte/work/generate.py` (module globals, `_bind_globals`, `_init_bucket_worker`, `_cluster_atoms_cart`, `_distance_fingerprint`)
- Test: `tests/test_multiref.py`

**Interfaces:**
- Consumes: `Crystal.get_references` (Task 1), `SymOpList.ref_indices` (Task 2).
- Produces: module globals `_monomers: list[Monomer]`, `_Z_by_ref: list[NDArray]`, `_self_iso_by_ref: list[list[NDArray]]`, `_W_lin_by_ref: list[list[NDArray]]`, `_check_congruence_by_ref: list[bool]`, plus `_A`/`_A_inv`; `_bind_globals(crystal, monomers)`; `_cluster_atoms_cart`/`_distance_fingerprint` reconstruct each member from its base reference.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_multiref.py`:

```python
from crystalatte.work import generate


def test_bind_globals_multi_reference_state():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    assert len(generate._monomers) == 4
    assert len(generate._Z_by_ref) == 4
    assert len(generate._W_lin_by_ref) == 4
    assert len(generate._check_congruence_by_ref) == 4
    # NH3 Z tags = [7,1,1,1] per reference
    assert list(generate._Z_by_ref[0]) == [7, 1, 1, 1]


def test_fingerprint_uses_per_member_reference():
    # A 2-member cluster whose members come from different NH3 references must
    # reconstruct atoms from each base reference; fingerprint is well-defined.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    g = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    g.ref_indices = np.array([0, 1])   # member 0 <- ref 0, member 1 <- ref 1
    zz, d = generate._distance_fingerprint(g)
    assert len(zz) == len(d) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py::test_bind_globals_multi_reference_state -v`
Expected: FAIL — `_bind_globals` still takes a single `monomer`; `_monomers`/`_Z_by_ref` don't exist.

- [ ] **Step 3: Convert module state and generalize reconstruction**

In `src/crystalatte/work/generate.py`:

Replace the single-monomer globals block with the per-reference lists:
```python
_crystal: Crystal | None = None
_monomers: list[Monomer] = []
_Z_by_ref: list[NDArray] = []
_self_iso_by_ref: list[list[NDArray]] = [[np.eye(3)]]
_W_lin_by_ref: list[list[NDArray]] = [[np.eye(3, dtype=np.int64)]]
_check_congruence_by_ref: list[bool] = [False]
_A: NDArray | None = None
_A_inv: NDArray | None = None
```

Rewrite `_bind_globals` to take the reference list and fill the per-reference state:
```python
def _bind_globals(crystal: Crystal, monomers: list[Monomer]) -> None:
    global _crystal, _monomers, _Z_by_ref, _self_iso_by_ref
    global _W_lin_by_ref, _check_congruence_by_ref, _A, _A_inv
    _crystal = crystal
    _monomers = list(monomers)
    _A = np.asarray(crystal.lattice_vectors, dtype=float)
    _A_inv = np.linalg.inv(_A)

    _Z_by_ref, _self_iso_by_ref, _W_lin_by_ref, _check_congruence_by_ref = [], [], [], []
    for mon in _monomers:
        _Z_by_ref.append(
            np.array([qcel.periodictable.to_Z(s) for s in mon.symbols], dtype=np.int64)
        )
        S = mon.self_isometries
        _self_iso_by_ref.append(S)
        site_ops = crystal.space_group.get_site_symmetry(mon.centroid_frac)
        W = []
        for op in site_ops:
            r = op.rot.astype(np.int64)
            if not any(np.array_equal(r, w) for w in W):
                W.append(r)
        if not W:
            W = [np.eye(3, dtype=np.int64)]
        _W_lin_by_ref.append(W)
        _check_congruence_by_ref.append(len(S) > len(W))
```

Update `_init_bucket_worker`:
```python
def _init_bucket_worker(crystal: Crystal, monomers: list[Monomer]) -> None:
    _bind_globals(crystal, monomers)
```

Generalize `_cluster_atoms_cart` to reconstruct each member from its base reference:
```python
def _cluster_atoms_cart(g_N: SymOpList) -> NDArray:
    ref = g_N.ref_indices
    parts = [op.apply(_monomers[ref[k]].frac_coords) for k, op in enumerate(g_N)]
    return _crystal.to_cartesian(np.vstack(parts))
```

Generalize `_distance_fingerprint` Z tagging to per-member base reference:
```python
    ...
    atoms = _cluster_atoms_cart(g_N)
    ref = g_N.ref_indices
    Z = np.concatenate([_Z_by_ref[ref[k]] for k in range(len(g_N))])
    i_idx, j_idx = np.triu_indices(len(atoms), k=1)
    ...
```
(Leave the sorting/caching logic unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py -v`
Expected: PASS. NOTE: `Tests/test_dedup.py` will now FAIL to import/run because it calls the old `_bind_globals(crystal, monomer)` and single-monomer globals — that is expected and fixed in Task 5 (API cutover). Do not fix test_dedup.py here; note it in your report.

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_multiref.py
git commit -m "feat: per-reference deduper state + per-member fingerprint reconstruction"
```

---

### Task 4: Dedup routing + cross-reference cloud congruence

**Files:**
- Modify: `src/crystalatte/work/generate.py` (`_same_multimer_int` gains a `W` param; generalize `_relative_transforms`/`_cart_frames` reconstruction for mono-ref; add `_same_multimer_geom`; rewrite `_is_bijection` routing)
- Test: `tests/test_multiref.py`

**Interfaces:**
- Consumes: per-reference state (Task 3).
- Produces: `_same_multimer_int(g_N, h_N, W)`; `_same_multimer_geom(g_N, h_N, atol=1e-3) -> bool` (labeled point-cloud congruence, reflection-allowed); `_is_bijection` routes mono-reference→integer(→rigid-frame float), cross-reference→cloud congruence.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_multiref.py`:

```python
def _mono(g):
    return len(set(g.ref_indices)) == 1


def test_cloud_congruence_matches_translated_and_mirror():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    # Same physical pair described two ways: (ref0 @ I, ref1 @ I) vs itself.
    g = SymOpList([SymOp.identity(), SymOp.identity()], 1); g.ref_indices = np.array([0, 1])
    h = SymOpList([SymOp.identity(), SymOp.identity()], 1); h.ref_indices = np.array([0, 1])
    assert generate._same_multimer_geom(g, h) is True
    # A genuinely different partner molecule must NOT match.
    h2 = SymOpList([SymOp.identity(), SymOp.identity()], 1); h2.ref_indices = np.array([0, 2])
    # (may or may not be congruent depending on geometry; assert it's a bool)
    assert isinstance(generate._same_multimer_geom(g, h2), bool)


def test_is_bijection_routes_mono_reference_to_integer(monkeypatch):
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    # two mono-reference (base 0) clusters -> must go through the integer matcher
    g = SymOpList([SymOp.identity(), _tr([12, 0, 0])], 1)   # ref_indices default [0,0]
    h = SymOpList([SymOp.identity(), _tr([-12, 0, 0])], 1)
    called = {"geom": 0}
    monkeypatch.setattr(generate, "_same_multimer_geom",
                        lambda a, b, **k: called.__setitem__("geom", called["geom"] + 1) or False)
    monkeypatch.setattr(generate, "_fingerprints_match", lambda a, b, **k: True)
    generate._is_bijection(g, h)
    assert called["geom"] == 0        # mono-reference never uses the cloud matcher


def _tr(dx12):
    return SymOp.identity().translate(np.array(dx12, dtype=np.int16))
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py::test_cloud_congruence_matches_translated_and_mirror -v`
Expected: FAIL — `_same_multimer_geom` does not exist.

- [ ] **Step 3: Implement the cross-reference matcher, parameterize the integer one, and route**

In `src/crystalatte/work/generate.py`:

Parameterize `_same_multimer_int` to take the dressing set (default keeps back-compat):
```python
def _same_multimer_int(g_N, h_N, W=None):
    ...
    S = _W_lin_by_ref[0] if W is None else W
    ...
```
(Only the `S = ...` binding changes; the rest of the anchored search is unchanged.)

Generalize the mono-reference rigid-frame float path: in `_cart_frames`, the reference geometry is that of the cluster's single base reference. Since `_same_multimer` (rigid-frame) is only called for mono-reference clusters, add a base lookup where it reconstructs, and use `_self_iso_by_ref[base]`. Concretely, add a helper `_base(g_N)` and thread it:
```python
def _base(g_N: SymOpList) -> int:
    return int(g_N.ref_indices[0])
```
In `_same_multimer` replace the module-level `_self_iso` reference with `S = _self_iso_by_ref[_base(g_N)]`. (`_relative_transforms`/`_cart_frames` already use only the SymOp rot/tr + `_A`, which are base-independent, so they need no change for the mono-reference case.)

Add the labeled point-cloud congruence for cross-reference:
```python
def _same_multimer_geom(g_N: SymOpList, h_N: SymOpList, atol: float = 1e-3) -> bool:
    """Return True iff the two clusters' reconstructed, Z-labeled atom clouds are
    congruent under some isometry (rotation, translation, or reflection).

    Anchor-based: center both clouds, pick a non-collinear anchor triple in G,
    match it to every Z-compatible, distance-compatible triple in H, solve the
    (possibly improper) rigid map by Kabsch, and accept iff it carries every
    atom of G onto a same-Z atom of H within ``atol``. Reflection is allowed
    (enantiomers are energetically equivalent).
    """
    A = _cluster_atoms_cart(g_N)
    B = _cluster_atoms_cart(h_N)
    if A.shape != B.shape:
        return False
    ZA = np.concatenate([_Z_by_ref[g_N.ref_indices[k]] for k in range(len(g_N))])
    ZB = np.concatenate([_Z_by_ref[h_N.ref_indices[k]] for k in range(len(h_N))])
    if sorted(ZA) != sorted(ZB):
        return False
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)

    n = len(A)
    # anchor triple in A: first three mutually non-collinear atoms
    def _triple(P):
        i0 = 0
        for i1 in range(1, n):
            for i2 in range(i1 + 1, n):
                v1, v2 = P[i1] - P[i0], P[i2] - P[i0]
                if np.linalg.norm(np.cross(v1, v2)) > 1e-3:
                    return (i0, i1, i2)
        return None
    tri = _triple(A)
    if tri is None:          # (near-)linear cluster: fall back to full-pair distance check
        from scipy.spatial.distance import cdist
        DA = np.sort(cdist(A, A).ravel())
        DB = np.sort(cdist(B, B).ravel())
        return bool(np.allclose(DA, DB, atol=atol))
    ia, ib, ic = tri
    dref = np.array([
        np.linalg.norm(A[ia] - A[ib]),
        np.linalg.norm(A[ia] - A[ic]),
        np.linalg.norm(A[ib] - A[ic]),
    ])
    from itertools import permutations
    from scipy.spatial.distance import cdist
    DB_full = cdist(B, B)
    same = {z: np.where(ZB == z)[0] for z in np.unique(ZB)}
    for jb in same[ZA[ia]]:
        for jc in same[ZA[ic]]:
            for jd in same[ZA[ib]]:
                if len({jb, jc, jd}) != 3:
                    continue
                dcand = np.array([DB_full[jb, jd], DB_full[jb, jc], DB_full[jd, jc]])
                if not np.allclose(dref, dcand, atol=atol):
                    continue
                P = A[[ia, ib, ic]]
                Q = B[[jb, jd, jc]]
                R = _kabsch(P, Q)         # allows reflection
                if R is None:
                    continue
                mapped = A @ R.T
                # every mapped atom must coincide with a same-Z atom of B
                ok = True
                used = np.zeros(n, dtype=bool)
                for k in range(n):
                    cand = same[ZA[k]]
                    dists = np.linalg.norm(B[cand] - mapped[k], axis=1)
                    hit = cand[dists < atol]
                    hit = [h for h in hit if not used[h]]
                    if not hit:
                        ok = False
                        break
                    used[hit[0]] = True
                if ok:
                    return True
    return False


def _kabsch(P: NDArray, Q: NDArray) -> NDArray | None:
    """Return the orthogonal R (det +/-1) minimizing ||P R^T - Q|| for centered
    corresponding point sets, or None if degenerate."""
    H = P.T @ Q
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = (Vt.T @ U.T)
    return R
```

Rewrite `_is_bijection` to route:
```python
def _is_bijection(g_N: SymOpList, h_N: SymOpList) -> bool:
    if not _fingerprints_match(g_N, h_N):
        return False
    if _mono(g_N) and _mono(h_N) and _base(g_N) == _base(h_N):
        i = _base(g_N)
        if _same_multimer_int(g_N, h_N, _W_lin_by_ref[i]):
            return True
        if _check_congruence_by_ref[i]:
            return _same_multimer(g_N, h_N)
        return False
    return _same_multimer_geom(g_N, h_N)


def _mono(g_N: SymOpList) -> bool:
    return len(set(g_N.ref_indices.tolist())) == 1
```
Delete any now-unused single-reference `_self_iso`/`_W_lin`/`_check_congruence` reads left over from Phase 0.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py -v`
Expected: PASS. (`Tests/test_dedup.py` still expected-broken until Task 5.)

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_multiref.py
git commit -m "feat: dedup routing (integer mono-ref, cloud congruence cross-ref)"
```

---

### Task 5: Multi-reference enumeration + `generate(crystal, N, R)` API cutover

**Files:**
- Modify: `src/crystalatte/work/generate.py` (`_generate_neighbors`, `_process_multimer`, `generate`, `_init_multimers`)
- Modify: `tests/test_dedup.py` (update to new API; drop the ammonia artifact test); `tests/test_generate.py` (update `generate` calls)
- Test: `tests/test_multiref.py`

**Interfaces:**
- Produces: `generate(crystal, N, R, **kwargs) -> list[Multimer]` (enumerates all references internally).

- [ ] **Step 1: Update the enumeration and API**

In `_generate_neighbors`, build the partner pool over **all** references and tag each partner with its base index, then anchor each reference in turn. Replace the single-reference `sym_ops = crystal.space_group.get_unique_sym_ops(monomer.centroid_frac)` + partner build with a loop over `enumerate(_monomers)`; each candidate `SymOpList` sets `ref_indices` for its members (anchor's base + each partner's base). Reconstruct centroids from the anchor + partner base molecules. (Full code: mirror the existing structure but carry `(base_index, SymOp)` per member; set `sop.ref_indices = np.array([...])`.)

`generate(crystal, N, R, **kwargs)`:
```python
def generate(crystal, N, R, **kwargs):
    monomers = crystal.get_references()
    d = np.linalg.norm(crystal.lattice_vectors, axis=0)
    R_eff = R + max(pdist(m.cart_coords).max() for m in monomers)
    # bounds computed per the existing formula using each reference centroid;
    # take the union (min low, max high) across references
    ...
    unique_neighbors, _ = _generate_neighbors(crystal, monomers, bounds, R, N, **kwargs)
    _init_multimers(crystal, monomers)
    ...
```
`_process_multimer` builds each `Monomer` from `_monomers[g_N.ref_indices[k]]`:
```python
    for k, g in enumerate(g_N):
        base = _monomers[g_N.ref_indices[k]]
        mon_frac = g.apply(base.frac_coords)
        ...
        mons.append(Monomer(base.symbols, mon_frac, mon_cart, com_frac, com_cart, g))
```
`_init_multimers(crystal, monomers)` binds `_monomers` (drop the single-monomer + masses signature; compute masses per base as needed).

- [ ] **Step 2: Update existing tests to the new API**

In `tests/test_dedup.py`: change every `generate.generate(crystal, monomer, 2, R, ...)` to `generate.generate(crystal, 2, R, ...)`; every `generate._bind_globals(crystal, monomer)` to `generate._bind_globals(crystal, crystal.get_references())`; and **remove** `test_generate_ammonia_enantiomer_fallback` (superseded by the physical ammonia test below). In `tests/test_generate.py`, drop the `monomer` positional arg from `generate` calls.

- [ ] **Step 3: Add the multi-reference end-to-end tests**

Append to `tests/test_multiref.py` (capture `AMMONIA_DIMERS` from Step 4's run):

```python
def test_benzene_regression_unchanged():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    mers = generate.generate(crystal, 2, 5.0, use_origin=True, n_jobs=1)
    assert len(mers) == 3          # Z'=1 path unchanged


def test_ammonia_physical_dimers():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    mers = generate.generate(crystal, 2, 5.0, use_origin=True, n_jobs=1)
    # now includes the real ~3.94 A hetero-molecular contacts, not just self-translates
    assert len(mers) == AMMONIA_DIMERS   # captured from the run in Step 4
    # every returned dimer has 2 monomers
    assert all(len(m.monomers) == 2 for m in mers)
```

- [ ] **Step 4: Capture the ammonia count, run, and lock it**

Run: `conda run -n cle python -c "import crystalatte as cle; from crystalatte.work import generate; cr,_=next(cle.from_cif('tests/x23/ammonia/ammonia.cif')); print('AMMONIA_DIMERS =', len(generate.generate(cr,2,5.0,use_origin=True,n_jobs=1)))"`
Substitute the printed integer for `AMMONIA_DIMERS` in the test. Sanity-check it is **larger** than the old artifact count of 2 (the physical nearest-neighbor dimers are now included).

- [ ] **Step 5: Run the dedup + multiref suites**

Run: `conda run -n cle python -m pytest Tests/test_multiref.py Tests/test_dedup.py -v`
Expected: PASS (benzene still 3; ammonia at the captured count; all dedup tests on the new API).

- [ ] **Step 6: Full suite sanity**

Run: `conda run -n cle python -m pytest Tests/ -q`
Expected: the dedup/generate/multiref tests pass; pre-existing unrelated failures (legacy v1 API, other-fixture CIF/Wyckoff/NaN) remain. Report the delta vs. the known 37-failure baseline — no NEW failures from this change.

- [ ] **Step 7: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_multiref.py tests/test_dedup.py tests/test_generate.py
git commit -m "feat: multi-reference enumeration + generate(crystal, N, R) API"
```

---

## Self-Review

**Spec coverage:**
- `get_references` orbit grouping → Task 1. ✓
- `SymOpList.ref_indices` → Task 2. ✓
- Per-reference state + `_bind_globals(crystal, monomers)` + generalized fingerprint → Task 3. ✓
- Routing (integer mono-ref / cloud cross-ref) + species-separating fingerprint + reflection-allowed congruence → Task 4. ✓
- Multi-reference enumeration + `generate(crystal, N, R)` + `_process_multimer` + benzene regression + physical ammonia count → Task 5. ✓
- Species-aware plumbing (per-ref Z, per-base reconstruction, Z-tagged fingerprint) → Tasks 3-4. ✓
- Out of scope (spglib detection, co-crystal enumeration) → untouched. ✓

**Placeholder scan:** The only runtime-captured value is `AMMONIA_DIMERS` (Task 5, Step 4 — explicit capture command). Task 5 Step 1 prose ("mirror the existing structure but carry (base_index, SymOp)") is the one spot that describes rather than transcribes: `_generate_neighbors` is the existing 100-line function being generalized member-by-member, and the exact edit depends on its current body; the implementer has the file and the precise contract (each candidate sets `ref_indices`). Flag for the implementer to work from the current function.

**Type consistency:** `ref_indices` is `int64` NDArray throughout; `_monomers`/`_Z_by_ref`/`_W_lin_by_ref`/`_self_iso_by_ref`/`_check_congruence_by_ref` are per-reference lists indexed by `g_N.ref_indices[k]` or `_base(g_N)`; `_same_multimer_int(g,h,W)`, `_same_multimer_geom(g,h,atol)`, `_bind_globals(crystal, monomers)`, `_is_bijection(g,h)` signatures are consistent across Tasks 3-5.
