# spglib Symmetry-Detection Front-End — Implementation Plan (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the true space group (spglib), collapse apparent Z′>1 to Z′=1 where possible, and route through the fast exact-integer path; fall back to Phase-1 multi-reference otherwise.

**Architecture:** `Crystal.detect_symmetry(symprec)` runs spglib on the ASU, converts the input-basis operations to `SymOp`s, and returns a new `Crystal` carrying a `SpaceGroup.from_sym_ops(...)` when strictly higher symmetry is found. `generate()` calls it automatically (opt-out via `detect_symmetry=False`). `get_unique_sym_ops` is refactored to dedup by centroid image so the detected (DB-table-less) group is handled correctly.

**Tech Stack:** Python 3, NumPy, spglib, qcelemental. Tests via `pytest` under conda env `cle`.

## Global Constraints

- Run tests with the **capital** path: `conda run -n cle python -m pytest Tests/<file> -v` (`testpaths=["Tests"]`; lowercase collects nothing). `git add tests/...` uses the real lowercase path.
- Fractional translations are integers in **twelfths**; `SymOp` rotations are `int16`.
- spglib `get_symmetry` returns ops in the **input cell basis**: integer rotations, translations that are multiples of 1/12. Convert with `SymOp.from_components(rot.astype(np.int16), np.round(t*12).astype(np.int16))`.
- **Behavior preservation:** the `get_unique_sym_ops` refactor must not change results for DB-backed groups — benzene `generate(..., 2, 5.0)` must stay **3** dimers.
- **Detection is conservative + graceful:** accept only strictly higher symmetry; any spglib/exception path falls back to the declared symmetry (never raises out of `detect_symmetry`). Default `symprec=1e-3`.
- Stage only this feature's changes; working tree is clean at the baseline (HEAD `dc80658`).

---

### Task 1: spglib dependency + `SpaceGroup.from_sym_ops` (with pickle safety)

**Files:**
- Modify: `pyproject.toml` (add `spglib`)
- Modify: `src/crystalatte/core/space_group.py` (`from_sym_ops`, `__reduce__`)
- Test: `tests/test_symmetry_detect.py` (create)

**Interfaces:**
- Produces: `SpaceGroup.from_sym_ops(sym_ops: list[SymOp], number: int|None=None, symbol=None) -> SpaceGroup` — a non-registry-cached instance with the given ops and empty `_special_positions`; picklable so it round-trips its explicit ops (not a DB reload).

- [ ] **Step 1: Write the failing test**

Create `tests/test_symmetry_detect.py`:

```python
import pickle
import numpy as np
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp


def test_from_sym_ops_holds_explicit_ops():
    ops = [SymOp.identity(), SymOp("-x,-y,-z")]
    sg = SpaceGroup.from_sym_ops(ops, number=2)
    assert len(sg.sym_ops) == 2
    assert sg.number == 2
    # not cached in the registry (detected groups are per-structure)
    assert SpaceGroup(2) is not sg


def test_from_sym_ops_pickles_with_its_ops():
    # detected groups travel to ProcessPool workers via pickle; they must keep
    # their explicit input-basis ops, NOT reload the DB standard setting.
    ops = [SymOp.identity(), SymOp("-x,-y+1/2,-z")]
    sg = SpaceGroup.from_sym_ops(ops, number=14)
    round_tripped = pickle.loads(pickle.dumps(sg))
    got = {op.augment.tobytes() for op in round_tripped.sym_ops}
    want = {op.augment.tobytes() for op in ops}
    assert got == want
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py -v`
Expected: FAIL — `SpaceGroup.from_sym_ops` does not exist.

- [ ] **Step 3a: Add `spglib` to `pyproject.toml`**

Add `"spglib"` to the `dependencies` list (match the existing formatting/quoting style in that file).

- [ ] **Step 3b: Add `from_sym_ops` and pickle-safe `__reduce__`**

In `src/crystalatte/core/space_group.py`, add the classmethod (near the other constructors) — it bypasses the registry-caching `__new__` via `object.__new__`:

```python
    @classmethod
    def from_sym_ops(cls, sym_ops, number=None, symbol=None) -> "SpaceGroup":
        """Build a SpaceGroup from an explicit operation list (e.g. detected by
        spglib in the input cell basis), bypassing the DB/registry. Has no
        special-position table; ``get_unique_sym_ops`` dedups by centroid image.

        :param sym_ops: The symmetry operations (already in the target basis)
        :param number: Optional international number (metadata/reporting only)
        :param symbol: Optional symbol dict (metadata only)
        :returns: A fresh, uncached SpaceGroup carrying exactly these ops
        """
        instance = object.__new__(cls)
        instance._number = number
        instance._hall = None
        instance._symbol = symbol if symbol is not None else {}
        instance._point_group = None
        instance._crystal_system = None
        instance._lattice_system = None
        instance._sym_ops = list(sym_ops)
        instance._lattice_constraints = {}
        instance._special_positions = {}
        instance._detected = True
        instance._initialized = True  # guard against __init__ re-running
        return instance
```

Replace the existing `__reduce__` so detected instances pickle their explicit ops:

```python
    def __reduce__(self):
        if getattr(self, "_detected", False):
            return (SpaceGroup._reconstruct_detected, (self._sym_ops, self._number, self._symbol))
        return (self.__class__, (self._number, self._hall))

    @classmethod
    def _reconstruct_detected(cls, sym_ops, number, symbol):
        return cls.from_sym_ops(sym_ops, number=number, symbol=symbol)
```

Ensure `_detected` defaults to False for normal instances: add `instance._detected = False` in `__new__` right after `instance = super().__new__(cls)` (so registry-created instances have the attribute), or rely on `getattr(..., False)` in `__reduce__` (already does). The `getattr` default is sufficient — no other change needed.

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py -v`
Expected: PASS (both tests). Also confirm no import breakage: `conda run -n cle python -c "import crystalatte"`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/crystalatte/core/space_group.py tests/test_symmetry_detect.py
git commit -m "feat: SpaceGroup.from_sym_ops (uncached, pickle-safe) + spglib dep"
```

---

### Task 2: `get_unique_sym_ops` refactor (always dedup by centroid image)

**Files:**
- Modify: `src/crystalatte/core/space_group.py` (`get_unique_sym_ops`)
- Test: `tests/test_symmetry_detect.py`

**Interfaces:**
- Produces: `get_unique_sym_ops(coords)` now always returns one op per distinct `op.apply(coords) mod 1` image; validates against `special_positions` multiplicity only when `coords` matches a DB special position. Works for detected groups (empty `special_positions`).

- [ ] **Step 1: Write the failing/guard tests**

Append to `tests/test_symmetry_detect.py`:

```python
def test_get_unique_sym_ops_detected_group_dedups_by_centroid():
    # A group with a 2-fold that fixes the origin: at the origin the 2-fold
    # shares the identity's centroid image, so unique ops = 1 (deduped).
    ops = [SymOp.identity(), SymOp("-x,-y,z")]  # 2-fold about z through origin
    sg = SpaceGroup.from_sym_ops(ops)
    at_origin = sg.get_unique_sym_ops(np.array([0.0, 0.0, 0.0]))
    assert len(at_origin) == 1
    # at a general point both ops give distinct centroid images -> both kept
    at_general = sg.get_unique_sym_ops(np.array([0.11, 0.23, 0.37]))
    assert len(at_general) == 2


def test_get_unique_sym_ops_db_behavior_preserved():
    # DB benzene (SG 61) at the origin special position: same op count as before.
    sg = SpaceGroup(61)
    ops_origin = sg.get_unique_sym_ops(np.array([0.0, 0.0, 0.0]))
    ops_general = sg.get_unique_sym_ops(np.array([0.13, 0.21, 0.37]))
    assert len(ops_general) == len(sg.sym_ops)          # general position: all ops
    assert 1 <= len(ops_origin) <= len(sg.sym_ops)       # special position: deduped
```

- [ ] **Step 2: Run to verify the detected-group test fails**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py::test_get_unique_sym_ops_detected_group_dedups_by_centroid -v`
Expected: FAIL — current `get_unique_sym_ops` returns all ops (2) for a group with no `special_positions`, not the deduped 1.

- [ ] **Step 3: Refactor `get_unique_sym_ops`**

Replace the body in `src/crystalatte/core/space_group.py`:

```python
    def get_unique_sym_ops(self, coords: NDArray) -> list[SymOp]:
        """Return one representative operation per distinct image of ``coords``
        (mod lattice) — the coset representatives modulo the site symmetry, so a
        monomer at ``coords`` is generated exactly once.

        When ``coords`` matches a DB special position, the deduped count is
        validated against the known Wyckoff multiplicity.

        :param coords: Fractional coordinates of the monomer centroid
        :returns: One SymOp per distinct centroid image
        :raises ValueError: If a matched special position's multiplicity disagrees
        """
        unique: dict = {}
        for op in self._sym_ops:
            res = tuple(np.round(op.apply(coords), 8) % 1)
            if res not in unique:
                unique[res] = op
        result = list(unique.values())

        for letter, pos in self._special_positions.items():
            for site in pos["coordinates"]:
                if np.allclose(coords, site):
                    if len(result) != pos["multiplicity"]:
                        raise ValueError(
                            f"Expected {pos['multiplicity']} unique symmetry "
                            f"operations for position {letter}, but found {len(result)}."
                        )
                    return result
        return result
```

- [ ] **Step 4: Run to verify pass + regressions**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py Tests/test_multiref.py Tests/test_dedup.py -v`
Expected: PASS (new dedup tests + all Phase-1 tests, including benzene behavior).

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/space_group.py tests/test_symmetry_detect.py
git commit -m "refactor: get_unique_sym_ops dedups by centroid image (supports detected groups)"
```

---

### Task 3: `Crystal.detect_symmetry`

**Files:**
- Modify: `src/crystalatte/core/crystal.py` (import `SymOp`; add `detect_symmetry`)
- Test: `tests/test_symmetry_detect.py`

**Interfaces:**
- Produces: `Crystal.detect_symmetry(symprec: float = 1e-3) -> Crystal` — a new higher-symmetry `Crystal` if spglib finds strictly more ops (whose rotations contain the declared rotations); else `self`; never raises.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_symmetry_detect.py`:

```python
import crystalatte as cle
from pathlib import Path
DATA = Path(__file__).parent


def test_detect_symmetry_ammonia_p1_to_p213():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    assert len(crystal.space_group.sym_ops) == 1          # declared P1
    detected = crystal.detect_symmetry()
    assert len(detected.space_group.sym_ops) == 12        # P2_13
    assert detected.space_group.number == 198
    # symmetry now relates the 4 NH3 -> a single reference
    assert len(detected.get_references()) == 1


def test_detect_symmetry_benzene_no_change():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    detected = crystal.detect_symmetry()
    # benzene's declared SG 61 is already correct -> not strictly higher -> self
    assert detected is crystal


def test_detect_symmetry_graceful_on_failure(monkeypatch):
    import spglib
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    monkeypatch.setattr(spglib, "get_symmetry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert crystal.detect_symmetry() is crystal            # falls back, no raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py::test_detect_symmetry_ammonia_p1_to_p213 -v`
Expected: FAIL — `Crystal` has no `detect_symmetry`.

- [ ] **Step 3: Implement `detect_symmetry`**

In `src/crystalatte/core/crystal.py`, add `from .sym_ops import SymOp` to the imports, and add the method:

```python
    def detect_symmetry(self, symprec: float = 1e-3) -> "Crystal":
        """Attempt to detect a higher true space group with spglib and return a
        new Crystal using it (input-basis operations, no coordinate transform).

        Returns ``self`` unchanged if spglib is unavailable, errors, or does not
        find strictly higher symmetry (more operations whose rotations contain
        the declared group's rotations).

        :param symprec: spglib symmetry tolerance (tight default recovers only
            genuine symmetry, not pseudo-symmetry)
        :returns: A higher-symmetry Crystal, or ``self``
        """
        try:
            import spglib

            lattice = self._frac_to_cart.T
            positions = np.asarray(self._asu.positions, dtype=float)
            numbers = [qcel.periodictable.to_Z(s) for s in self._asu.atoms]
            sym = spglib.get_symmetry((lattice, positions, numbers), symprec=symprec)
            if sym is None:
                return self

            detected = [
                SymOp.from_components(
                    np.asarray(r, dtype=np.int16),
                    np.round(np.asarray(t) * 12).astype(np.int16),
                )
                for r, t in zip(sym["rotations"], sym["translations"])
            ]
            declared = self._space_group.sym_ops
            if len(detected) <= len(declared):
                return self
            det_rots = [op.rot for op in detected]
            if not all(
                any(np.array_equal(op.rot, dr) for dr in det_rots) for op in declared
            ):
                return self

            ds = spglib.get_symmetry_dataset((lattice, positions, numbers), symprec=symprec)
            number = int(ds.number) if ds is not None else None
            new_sg = SpaceGroup.from_sym_ops(detected, number=number)
            return Crystal(self._vectors, self._angles, new_sg, self._asu)
        except Exception:
            return self
```

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py -v`
Expected: PASS (ammonia→P2₁3/1-reference; benzene→self; graceful-failure→self).

- [ ] **Step 5: Commit**

```bash
git add src/crystalatte/core/crystal.py tests/test_symmetry_detect.py
git commit -m "feat: Crystal.detect_symmetry (spglib, input-basis ops, graceful fallback)"
```

---

### Task 4: Automatic activation in `generate` + the detect-invariant

**Files:**
- Modify: `src/crystalatte/work/generate.py` (`generate` signature + first lines)
- Test: `tests/test_symmetry_detect.py`

**Interfaces:**
- Produces: `generate(crystal, N, R, *, detect_symmetry=True, symprec=1e-3, **kwargs)` — attempts detection first (unless disabled), then enumerates on the resulting crystal.

- [ ] **Step 1: Write the failing/invariant tests**

Append to `tests/test_symmetry_detect.py`:

```python
from crystalatte.work import generate


def test_generate_detection_invariant_ammonia():
    # Detection changes HOW (fast integer path), not WHAT (the unique n-mer set).
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    for kw in (dict(use_origin=True), dict()):  # use_origin and default conventions
        n_on = len(generate.generate(crystal, 2, 5.0, n_jobs=1, detect_symmetry=True, **kw))
        n_off = len(generate.generate(crystal, 2, 5.0, n_jobs=1, detect_symmetry=False, **kw))
        assert n_on == n_off, f"detection changed the count for kwargs={kw}: {n_on} != {n_off}"


def test_generate_detection_uses_single_reference_path_ammonia():
    # With detection, ammonia collapses to Z'=1 (one reference), so the integer
    # matcher (not the float cloud congruence) handles it.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    detected = crystal.detect_symmetry()
    assert len(detected.get_references()) == 1


def test_generate_benzene_regression_with_detection():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    assert len(generate.generate(crystal, 2, 5.0, use_origin=True, n_jobs=1)) == 3
```

- [ ] **Step 2: Run to verify they fail**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py::test_generate_detection_invariant_ammonia -v`
Expected: FAIL — `generate` does not accept `detect_symmetry` (TypeError).

- [ ] **Step 3: Add activation to `generate`**

In `src/crystalatte/work/generate.py`, update `generate`'s signature and first lines:

```python
def generate(crystal, N, R, *, detect_symmetry=True, symprec=1e-3, **kwargs):
    if detect_symmetry:
        crystal = crystal.detect_symmetry(symprec)
    monomers = crystal.get_references()
    ...  # rest unchanged
```
(Keep the remaining body exactly as in Phase 1.)

- [ ] **Step 4: Run to verify pass**

Run: `conda run -n cle python -m pytest Tests/test_symmetry_detect.py -v`
Expected: PASS — the ammonia detect-on count equals the detect-off count for both conventions; benzene still 3.

- [ ] **Step 5: Full-suite sanity**

Run: `conda run -n cle python -m pytest Tests/ -q`
Expected: the dedup/multiref/symmetry-detect tests pass; pre-existing unrelated failures (legacy v1 API + other-fixture CIF/Wyckoff/NaN) unchanged from the ~37-failure baseline. Report the delta and confirm no NEW failures.

- [ ] **Step 6: Commit**

```bash
git add src/crystalatte/work/generate.py tests/test_symmetry_detect.py
git commit -m "feat: automatic spglib symmetry detection in generate() with fallback"
```

---

## Self-Review

**Spec coverage:**
- spglib dep + `from_sym_ops` (uncached, pickle-safe `__reduce__`) → Task 1. ✓
- `get_unique_sym_ops` refactor (centroid-image dedup, DB-behavior-preserving) → Task 2. ✓
- `Crystal.detect_symmetry` (input-basis ops, strictly-higher guard, graceful fallback) → Task 3. ✓
- Automatic activation in `generate` (`detect_symmetry`/`symprec`) → Task 4. ✓
- The detect-on == detect-off invariant, single-reference path check, benzene regression, fallback → Tasks 3-4. ✓
- Out of scope (DB-reload injection, loose-symprec, co-crystals) → untouched. ✓

**Placeholder scan:** none — all steps carry concrete code; no runtime-captured constants (the invariant test compares two live runs rather than asserting a magic number).

**Type consistency:** `from_sym_ops(sym_ops, number, symbol)` and `_reconstruct_detected` match; `detect_symmetry(symprec) -> Crystal` and `generate(..., detect_symmetry, symprec)` are consistent; `SymOp.from_components(int16 rot, int16 tr)` used uniformly; the refactored `get_unique_sym_ops(coords) -> list[SymOp]` keeps its signature.
