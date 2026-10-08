# spglib Symmetry-Detection Front-End — Phase 2

**Date:** 2026-08-07
**Component:** `src/crystalatte/core/crystal.py`, `src/crystalatte/core/space_group.py`, `src/crystalatte/work/generate.py`, `pyproject.toml`
**Status:** Approved design
**Builds on:** Phase 1 (Z′>1 multi-reference enumeration, on `v2`). This layer sits *in front* of Phase 1 and falls back to it.

## Problem / goal

Phase 1 makes Z′>1 crystals (and reduced-symmetry CIFs like the P1 ammonia fixture) produce the *correct* unique n-mers, but cross-reference clusters go through the float point-cloud congruence — slower and not exact-integer. Many such cases are only *apparently* Z′>1 because the CIF understates the symmetry (ammonia is really P2₁3). Detecting the true space group collapses Z′ to 1 and restores the fast, exact-integer path.

**Confirmed feasibility:** spglib (2.7.0, in env `cle`) detects P2₁3 (SG 198, 12 ops) for the ammonia fixture across symprec 1e-3…5e-2. `get_symmetry` returns operations in the **input cell basis** — integer rotations, translations that are exact multiples of 1/12 — that map the ASU onto itself. So no coordinate transform is needed; the ops convert directly to the code's twelfths `SymOp` format.

## Settled decisions

1. **Injection:** use spglib's **input-basis operations directly** (not detect-number-then-reload-DB). Avoids all standard-setting/origin coordinate-transform complexity. The tradeoff — no DB special-position/Wyckoff table for the detected group — is absorbed by the `get_unique_sym_ops` refactor below.
2. **Activation:** **automatic, conservative, with fallback.** `generate()` attempts detection with a tight default `symprec=1e-3`; uses the result only if strictly higher symmetry is found; otherwise silently falls back to the declared symmetry + Phase-1 multi-reference. Escape hatches: `detect_symmetry=False`, tunable `symprec`.

## Design

### 1. Dependency
Add `spglib` to `pyproject.toml` dependencies.

### 2. `SpaceGroup.from_sym_ops(sym_ops, number=None, symbol=None) -> SpaceGroup`
Construct a `SpaceGroup` from an explicit `list[SymOp]`, bypassing the DB/registry: sets `_sym_ops` directly, `_special_positions = {}`, and `_number`/`_symbol` from the optional metadata (for reporting only). Not cached in the registry (detected groups are per-structure). Everything else (`sym_ops` property, `get_site_symmetry`, `get_unique_sym_ops`) works on `_sym_ops`.

### 3. `get_unique_sym_ops` refactor (load-bearing)
Today it dedups ops only when the centroid matches a **DB** `special_positions` entry, else returns all ops. Refactor to **always** dedup ops by their action on the centroid — keep one op per distinct `op.apply(coords) mod 1` image — and use `special_positions` (when present) only to *validate* the deduped count against the known multiplicity. This is:
- **behavior-preserving for the DB path**: general position → all centroid images distinct → all ops kept (= current "return all"); special position → site-symmetry ops share a centroid image → deduped (= current special-position branch);
- **correct and efficient for a detected group** with empty `special_positions`: e.g. NH₃ on P2₁3's 3-fold axis → 12 ops dedup to 4 (the coset reps), so the reference isn't generated 3× and then discarded.

Benzene's unchanged 3-dimer result is the regression guard that DB-path behavior is preserved.

### 4. `Crystal.detect_symmetry(symprec=1e-3) -> Crystal`
- Build the spglib cell from the ASU: `(lattice_vectors.T, asu.positions, [Z(s) for s in asu.atoms])`.
- `sym = spglib.get_symmetry(cell, symprec)`; convert each `(rotation int (3,3), translation float (3))` to a `SymOp` via `SymOp.from_components(rot.astype(int16), round(translation*12).astype(int16))`.
- **Accept only if strictly higher symmetry:** `len(detected) > len(self.space_group.sym_ops)` AND the detected op set contains the declared ops (supergroup sanity). If accepted, return a **new** `Crystal` (same cell/ASU) whose `space_group = SpaceGroup.from_sym_ops(detected, number=spglib int number, symbol=...)`. Otherwise return `self`.
- Wrap in `try/except (ImportError, Exception)` → on any failure return `self` (graceful fallback; never raises out of detection).

### 5. Automatic activation in `generate`
`generate(crystal, N, R, *, detect_symmetry=True, symprec=1e-3, **kwargs)`: at the top,
```python
if detect_symmetry:
    crystal = crystal.detect_symmetry(symprec)
```
then proceed unchanged. All downstream (`get_references`, bounds, `_generate_neighbors`) uses the possibly-higher-symmetry crystal. For ammonia this collapses Z′=4→1 and routes clusters through the exact-integer matcher rather than the float congruence.

## Testing

- **Detection basics:** `crystal.detect_symmetry()` on ammonia yields a `SpaceGroup` with number 198 and 12 ops; `detected_crystal.get_references()` returns **1** (Z′ collapsed).
- **The invariant (core correctness):** detection changes *how*, not *what* —
  `len(generate(ammonia, 2, R, detect_symmetry=True)) == len(generate(ammonia, 2, R, detect_symmetry=False))`
  for both the `use_origin` and default conventions (the Phase-1 counts, e.g. 3 / 4 at R=5). This proves the fast detected path and the Phase-1 multi-reference path agree on the unique-n-mer set.
- **Path check:** with detection on, ammonia clusters are mono-reference (all `ref_indices == 0`), so the integer matcher handles them and the float cloud congruence is not needed (assert via `_mono`, or that `get_references()` returns 1).
- **Benzene regression:** `generate(benzene, 2, 5.0)` still returns **3** with detection on (detection finds benzene's real SG 61; the `get_unique_sym_ops` refactor preserves behavior). Also `get_unique_sym_ops` on the DB benzene at its special position returns the same ops as before the refactor.
- **Fallback:** `detect_symmetry=False` disables detection (ammonia stays Z′=4, multi-reference). A crystal where detection finds nothing higher returns `self` (no change). Force a spglib failure path (e.g. monkeypatch) → falls back without raising.
- **`get_unique_sym_ops` refactor unit test:** general position returns all ops; a special position dedups; DB multiplicity validation still fires when applicable.

## Out of scope

- Standard-setting / DB-number-reload injection (rejected in favor of input-basis ops).
- Loose-symprec pseudo-symmetry exploration (default stays tight at 1e-3).
- Co-crystal enumeration (Phase-1 groundwork only; unchanged here).
