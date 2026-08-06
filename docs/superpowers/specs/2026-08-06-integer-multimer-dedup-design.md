# Exact-Integer Multimer Deduplication with Float Congruence Fallback

**Date:** 2026-08-06
**Component:** `src/crystalatte/work/generate.py` (`_is_bijection` and helpers)
**Status:** Approved design

## Problem

`_is_bijection` currently decides "same multimer" with two branches:

- **(a) exact branch** — integer `int16` equality on augmented sym-op matrices
  (`composed = g_aug @ h_aug`, set-compared to `g_N`). Exact and fast, but only
  catches merges whose relating isometry is undressed and already a placement in
  the cluster.
- **(b) geometric branch** — `_same_multimer`, matching clusters in *Cartesian*
  relative-transform space with the full molecular self-isometry group `S` and
  `np.allclose(atol)`. Complete, but floating-point: subject to roundoff (worse
  at large separations) and slower.

The two branches are separate code paths, and the complete one sacrifices the
exactness of the fast one.

## Key insight

The relative transforms `Δ_ab = T_a⁻¹ T_b` were never intrinsically Cartesian.
In the **lattice (fractional) basis** they are exact integers:

- placements are `(rot, tr/12)` with `rot ∈ GL₃(ℤ)`, `det rot = ±1`, `tr ∈ ℤ³`;
- `rot_a⁻¹` is integer (unimodular inverse = `±adj(rot_a)`);
- so `L_ab = rot_a⁻¹ rot_b ∈ ℤ³ˣ³` and `v_ab = rot_a⁻¹(tr_b − tr_a) ∈ ℤ³`
  (twelfths).

The *only* thing that forced floats was the dressing set: a generic molecular
self-isometry (e.g. benzene's `C6`) is irrational in the lattice basis. But
**every crystallographically-required merge** (same symmetry orbit — the merges
needed for correct multiplicity) uses a dressing in the crystallographic **site
symmetry** `W = Stab_𝒢(reference monomer)`, whose linear parts are integer in
the lattice basis. (Proof: `docs/dedup_correctness.tex`, Thm 1 + the `s_a ∈ W`
argument.)

So the exact-integer relative-transform check with dressing `W`:

- is a **strict superset of branch (a)** (which only ever caught `s=I`,
  crystallographic merges);
- is exact at all distances (integer `array_equal`, no tolerance);
- is complete for all crystallographic (multiplicity-required) merges, including
  the centrosymmetric `s=−I` site-inversion case that branch (a) misses.

The residue that genuinely needs floats is **non-crystallographic congruences**:
clusters that are geometrically congruent (equal energy) but crystallographically
inequivalent. These use a dressing in `S ∖ W` — both proper (molecular-symmetry
congruences, e.g. `C6`-related neighbors) and improper (enantiomers). Per the
project goal — *deduplicate all energetically-equivalent multimers* — we keep
merging **all** congruences, not only enantiomers, via a gated float fallback.

## Design

`_is_bijection(g_N, h_N)` — three stages, cheapest first:

1. **`_fingerprints_match`** gate (unchanged).
2. **`_same_multimer_int`** — exact-integer relative-transform matcher, dressing
   set `W` (integer, always includes `I`). Replaces branch (a) *and* the
   crystallographic part of branch (b).
3. **Float fallback** — the existing `_same_multimer` (full molecular `S`,
   Cartesian, `atol`), run **only when stage 2 fails and `S ⊋ W`**. Catches all
   remaining congruences (proper + enantiomer). When `S == W` it can find nothing
   new, so it is skipped — behavior-identical to today, strictly less work.

### Why the gate preserves current behavior

- Crystallographic merge (`s ∈ W`): caught by stage 2. (Today: branch (a) if
  `s=I`, else branch (b).)
- Non-crystallographic congruence (`s ∈ S∖W`): stage 2 fails, `S⊋W` holds, float
  runs and merges. (Today: branch (b).)
- No merge, `S⊋W`: stage 2 fails, float runs and fails. (Today: (a),(b) fail.)
- No merge, `S==W`: stage 2 fails, float **skipped**. (Today: (a),(b) fail.) —
  same result, less float work.

## Components / changes

- **`SpaceGroup.get_site_symmetry(coords) -> list[SymOp]`** (new): the ops
  `op` with `op.apply(coords) ≡ coords (mod 1)` — the stabilizer `W`. Exact,
  sourced from 𝒢, independent of distorted CIF coordinates. (Kept as a method
  rather than inline in `generate.py` for testability.)
- **`generate.py` module globals**: add
  - `_W_lin: list[NDArray]` — unique integer `rot` parts of `W` (always includes
    `I`), the dressing set for the integer matcher;
  - `_check_congruence: bool` — `len(_self_iso) > len(_W_lin)` (i.e. `S ⊋ W`),
    the fallback gate;
  - keep `_self_iso`, `_A`, `_A_inv` for the float fallback.
  All set in `_bind_globals` so `ProcessPoolExecutor` workers inherit them via
  `_init_bucket_worker`.
- **`_relative_transforms_int(g_N) -> (L, v)`** (new): integer `(L, v)` in the
  lattice basis, `L[a,b]=rot_a⁻¹ rot_b`, `v[a,b]=rot_a⁻¹(tr_b−tr_a)` (twelfths).
  Rot inverse via **exact adjugate** (`round(det·inv)` with an integer-check),
  *not* the transpose used by `SymOpList.invert_list` (invalid for
  non-orthogonal lattice bases). Cached in a new `SymOpList._rel_int_cache`
  field, separate from the Cartesian `_rel_cache`.
- **`_same_multimer_int(g_N, h_N) -> bool`** (new): the anchored search
  (`docs/dedup_correctness.tex`, Thm 2) in integer arithmetic — enumerate
  `(h0, s0∈W)`, force `π` by `v`-match, solve `s_b = (L^H)⁻¹ s0 L^G`, test
  `s_b ∈ W`, verify all pairs — every comparison `np.array_equal`.
- **`_same_multimer`** (float) and **`_relative_transforms`** (Cartesian):
  unchanged, now used only as the fallback.
- **`_is_bijection`**: rewritten to the 3 stages; the `composed`/`aug_cache`
  branch-(a) block is deleted.
- **`SymOpList`**: add `_rel_int_cache` field (lazily populated, `compare=False`,
  mirrors `_rel_cache`).
- **Documentation note** (module docstring + `_is_bijection` docstring): the
  fallback uses the full molecular `S`, so the deduper merges *all*
  energetically-equivalent (congruent) N-mers — proper molecular-symmetry
  congruences as well as mirror-image enantiomers. This is intentional and goes
  beyond an enantiomer-only check; the dedup target is energetic equivalence.

## Testing

- **Unit `_same_multimer_int`** (constructed `SymOpList`s, exact):
  - identity / self-match;
  - lattice translate & count-from-other-monomer (`G=[I,g]` vs `[I,g⁻¹]`);
  - centrosymmetric inversion: `G=[I,p]` vs `[I,i₀p]` merged with `s₀=−I∈W`
    (the Thm-6 case branch (a) misses);
  - a genuine non-match returns `False`.
- **Exactness**: a large-lattice-translate equivalent pair that the integer path
  matches exactly (documents the roundoff-immunity motivation).
- **Fallback gate**: a Sohncke-group achiral case where stage 2 fails and the
  float fallback merges; assert `_check_congruence` is `True` there and `False`
  for a general-position case with `S==W`.
- **Regression**: run `generate()` on existing `tests/x23/*` fixtures (benzene,
  etc.) and confirm the unique-multimer counts are unchanged from current `v2`
  behavior (Option C is behavior-preserving).

## Out of scope

No change to translation-fingerprint bucketing, neighbor generation,
`get_unique_sym_ops`, `_process_bucket`/`_process_multimer`, or the
multiprocessing structure.
