# Z′>1 (Multi-Reference) N-mer Enumeration — Phase 1

**Date:** 2026-08-06
**Component:** `src/crystalatte/work/generate.py`, `src/crystalatte/core/crystal.py`, `src/crystalatte/core/sym_ops.py`
**Status:** Approved design
**Follow-on:** Phase 2 = spglib symmetry-detection front-end (separate spec) that reduces Z′ where possible and falls back to this multi-reference engine.

## Problem

The generation pipeline assumes **Z′ = 1** — exactly one molecule per asymmetric unit, with every other molecule a crystal-symmetry image of it. Concretely:

- `from_cif` puts all ASU atoms into `Crystal`, then `crystal.get_reference()` (`crystal.py:196`) returns the **single** connected molecule closest to the origin; the rest are discarded.
- `_generate_neighbors` builds the partner pool as `{ sym_op(reference.frac_coords) + lattice_translation }` — only the symmetry+translation orbit of that one reference.

When the ASU holds multiple symmetry-independent molecules (Z′>1) — either genuinely, or because a CIF is written in a reduced-symmetry setting (the `tests/x23/ammonia/ammonia.cif` fixture is P1 with 4 explicit NH₃) — the pipeline **silently under-enumerates**: it forms n-mers only among copies of one reference and misses every n-mer involving the other independent molecules. Measured on the P1 ammonia: the reference NH₃'s true nearest neighbors (the other three ASU molecules) sit at **3.94 Å**, but the only partners generated are its own lattice translates at **5.13 Å** — the physical dimers are never enumerated.

## Settled decisions

1. **Co-crystal scope:** species-aware *plumbing* now (every member carries its own base molecule → symbols/coords/Z), but only homomolecular Z′>1 is enumerated and tested this pass. Mixed-species enumeration/fixtures come later on the same representation.
2. **API:** `generate(crystal, N, R, **kwargs)` drops the `monomer` argument and enumerates all references internally, returning all unique `Multimer`s. The one internal caller and tests are updated.
3. **Dedup exactness:** exact-integer `_same_multimer_int` for **mono-reference** clusters (all members share one base reference — the common same-molecule contacts, still crystallographic and exact); the **float** `_same_multimer` for everything else (mixed references, or two mono clusters with different bases).

## Design

### 1. Reference enumeration — `Crystal.get_references()`

New method returning **all symmetry-independent molecules** in the ASU (one representative per molecular orbit). Reuses the existing `get_reference` machinery: expand ASU by sym-ops × lattice translations, BFS into complete molecules (`_bfs`), then **group the complete molecules into symmetry orbits** — two complete molecules are the same reference iff some `sym_op + lattice` maps one onto the other (atom sets coincide within tolerance) — and return one canonical (near-origin) representative `Monomer` per orbit.

- Benzene/Pbca (half-molecule ASU, single orbit) → **1** reference (unchanged behavior).
- P1 ammonia (4 molecules, no relating symmetry) → **4** references.

`get_reference()` is retained as `get_references()[0]` for existing single-reference callers.

**This orbit-grouping is the highest-risk new algorithm** — explicit tests for both Z′=1 (benzene) and multi-molecule (ammonia → 4) cases, with attention to matching tolerance.

### 2. Candidate representation — members carry their base molecule

Each cluster **member is `(base_ref_index, SymOp)`** — "reference molecule `base_ref_index`, placed by `SymOp`". `SymOpList` gains a parallel integer array `ref_indices` (one entry per member), **defaulting to all-zeros** so every existing Z′=1 use is unchanged. Per-reference state replaces the single-monomer module globals in `generate.py` (all indexed by base reference):

- `_monomers: list[Monomer]` (replaces `_monomer`)
- `_Z_by_ref: list[NDArray]` (per-reference atomic numbers; replaces `_Z_per_monomer`)
- `_self_iso_by_ref: list[list[NDArray]]` (per-reference/-species molecular point group)
- `_W_lin_by_ref: list[list[NDArray]]` (per-reference integer site-symmetry dressing)
- `_check_congruence_by_ref: list[bool]`
- `_A`, `_A_inv` (lattice, shared)

`_bind_globals(crystal, monomers)` computes all of these for the full reference list; the worker initializer passes the list through.

### 3. Enumeration — `_generate_neighbors` over the union of orbits

The partner pool becomes the union over all references:
`{ (i, sym_op·translation) : ∀ reference i, ∀ sym_op, ∀ lattice translation }`, pruned by the cutoff, each partner tagged with its base index `i`. n-mer candidates are built by **anchoring each reference `i` in turn** (identity placement of molecule `i`) plus `N−1` partners drawn from the full pool. Cross-anchor and cross-reference duplicates are collapsed by dedup. `translation_fp` bucketing is retained (it is a global-isometry invariant; equivalent clusters still share it, and the Z-tagged fingerprint separates species/geometry within a bucket).

### 4. Deduplication — routing + generalized float path

`_is_bijection(g_N, h_N)`:

```
if not _fingerprints_match(g_N, h_N): return False           # Z-tagged, species-separating
if _mono(g_N) and _mono(h_N) and _base(g_N) == _base(h_N):
    i = _base(g_N)
    if _same_multimer_int(g_N, h_N, W=_W_lin_by_ref[i]): return True
    if not _check_congruence_by_ref[i]: return False
return _same_multimer(g_N, h_N)                               # float, species-aware
```

- **`_distance_fingerprint`**: reconstruct each member's atoms from `_monomers[ref_indices[k]]`, tag distances by `Z_iZ_j` using `_Z_by_ref[...]`. Species are separated for free (different Z products).
- **`_same_multimer_int`**: unchanged core; parameterized by the reference's `W` dressing. Only ever invoked on mono-reference clusters that share a base, so it stays crystallographic and exact.
- **`_same_multimer` (float), generalized:** each member's rigid frame is `(R_k, t_k)` where `R_k = (A·rot_g·A⁻¹) · M_i^R` and `t_k` is the member's Cartesian centroid. `M_i^R` is the rotation mapping a **per-species canonical geometry** `X₀` onto reference `i` (a one-time Kabsch fit per reference; exact since `mol_i` is a rigid copy of its species). The existing relative-transform matching then applies, modding out the **species** self-isometries; the member permutation is **restricted to species-preserving maps** (a no-op for same-species now, correct for co-crystals later).

### 5. Multimer construction + downstream

`_process_multimer` builds each `Monomer` from `_monomers[member.base_ref_index]` transformed by the member's `SymOp`. `Multimer` and `count_chemistry_classes` are already per-monomer and need no change. Multiplicity accounting is unchanged in spirit: dedup runs across all candidates from all anchors, so a physical n-mer reached from several anchors collapses to one representative with the correct multiplicity.

### 6. Public API + callers

`generate(crystal, N, R, **kwargs)` calls `crystal.get_references()`, binds the full reference list, enumerates across all anchors, dedups, and returns all unique `Multimer`s. Update the internal call site and all tests that pass a `monomer` positional argument.

## Out of scope (this pass)

- **Symmetry detection (spglib)** — Phase 2, a separate spec layered in front of this engine. spglib is confirmed available (2.7.0) and detects P2₁3 for the ammonia fixture at all tolerances; deferring only the integration (inject-strategy: detect-number-reload-DB vs. use-input-basis-ops, plus setting/origin handling).
- **Co-crystal enumeration and mixed-species fixtures** — the representation is made ready (species-aware `_by_ref` state, Z-tagged fingerprint, species-restricted permutation), but mixed-species ASUs are not enumerated or tested here.

## Testing

- **`get_references`**: benzene → 1 reference; P1 ammonia → 4 references. Assert count and that references are distinct molecules.
- **Z′=1 regression**: benzene end-to-end unchanged (still **3** unique dimers) — proves the single-reference path is untouched, and that `ref_indices` defaulting to zero preserves behavior.
- **P1 ammonia end-to-end**: now returns the physical nearest-neighbor dimer set (the 3.94 Å contacts), exercising multi-reference enumeration + the float cross-reference dedup. Count captured empirically at implementation time; replaces the current artifact-count `test_generate_ammonia_enantiomer_fallback`.
- **Routing**: assert a mono-reference cluster is decided by `_same_multimer_int` (integer path still taken within one reference) and that cross-reference pairs go to the float path.
- **Species-aware fingerprint**: a unit check that clusters differing only in species do not match (guards the co-crystal groundwork even without a mixed fixture).
