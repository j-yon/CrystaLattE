# Integer Extra Molecular Symmetry in the Orbit Key (Retire the Cartesian Congruence Pass)

**Date:** 2026-10-20
**Component:** `src/crystalatte/work/dedup/` (`context.py`, `congruence.py`, `matchers.py`), `work/neighbors.py`
**Status:** Draft, awaiting review

## Problem

Deduplication runs two passes (`neighbors.generate_neighbors`):

1. **Exact pass**: `orbit_labels`, an exact `int64` orbit key in the lattice
   basis, dressed with the crystallographic site symmetry `W`.
2. **Congruence pass**: `congruence.merge_congruent`, a Cartesian float
   search (`same_multimer` → `anchored_match`, `atol=1e-3` Å) dressed with
   the molecular point group `S` from `PointGroup.detect` (0.05 Å ladder).
   Gated per reference by `len(S) > len(W)`.

Measured on the 26 fixtures (dimers to 20 Å unless noted):

- The gate opens on 9 fixtures. On 6 of them (benzene, naphthalene,
  anthracene, pyrazine, imidazole, 1,4-cyclohexanedione) `S` is only
  approximate (worst atom residual 2–5e-2 Å). The `1e-6` Å fingerprint
  screen then rejects every candidate pair and the pass merges nothing.
  The nearest same-element near-misses are ≥0.18 Å apart, and none match
  even at `atol=0.2`, so nothing is being missed.
- Approximate merges are impossible by construction anyway. A rotation
  error δ moves a partner by δ·|v| (benzene δ ≈ 1e-2 rad, so ≈0.2 Å at
  20 Å), far outside the 1e-3 tolerance.
- The pass changes results only when the extra symmetry is **exact to
  float precision**: triazine (3 merges at 20 Å, 27 at 35 Å, 3 for trimers
  at 12 Å) and linear CO₂ (4). Ammonia's extra ops are exact but produce 0
  merges.

So the float pipeline (`PointGroup.detect` in `bind`, gate, spread window,
fingerprint screen, Cartesian matcher) exists to handle exact extra
symmetry, and exact symmetry can be found and applied in integer arithmetic.

## Key insight

The orbit key decides the matching condition of `docs/dedup_correctness.tex`
(`thm:match`) with dressings drawn from a per-reference group. Relative
transforms `Δ_ab = T_a⁻¹ T_b` are already blind to the global isometry
relating two clusters. Nothing in the key requires the dressing group to be
crystallographic; it requires only:

1. **Integer linear parts** in the working basis (the conventional cell
   vectors the code uses), unimodular so `int_inv` is exact.
2. **Twelfths translations**: the dressing is the affine map
   `x ↦ h x + d` with `d = (I − h) c` (c = fractional centroid), and `12 d`
   must be integral.
3. **A group**, so the coset minima are class invariants.
4. **Genuine self-isometries** of the reference monomer.

An orthogonal op is integer in the working basis iff it preserves the box
spanned by the three cell vectors, i.e. `hᵀ G h = G` with `G = AᵀA`. Call
that finite group (at most 48 ops) `H`. Then the usable extra symmetry is

    D = W ∪ { h ∈ H : h fixes the reference monomer about its centroid (≤ 1e-6 Å)
                       and 12 (I − h) c ∈ ℤ³ }

`H` is **not** the textbook point group of the crystal's Bravais lattice.
For an R-centred cell on hexagonal axes (triazine, R-3c) the R lattice has
-3m (12 ops), but the hexagonal box has 6/mmm (24). All six of triazine's
extra ops lie in 6/mmm and outside -3m, and they are exactly the ops behind
its verified merges. The design therefore uses the symmetry of the working
cell, not the Bravais lattice.

Molecular symmetry outside `H` (ammonia's σv, any C5, any icosahedral op)
is deliberately not used. Such an op can relate two clusters exactly only
when its orientation, set by free atom coordinates, happens to carry
lattice vectors exactly onto lattice vectors. That is a coincidence, not
something the structure forces, and ammonia's exact σv produced 0 merges in
every run. The exception is linear molecules: their continuous symmetry
makes such coincidences routine. They are already warned about as
unsupported (see Behaviour changes).

## Design

All new work happens once per reference, inside `context.bind`. Per-cluster
work is the existing exact orbit key, unchanged.

### 1. Cell symmetry `H`, from lookup plus metric test

- **Lookup (`H_lookup`, no cell parameters).** Take the space group's integer
  linear parts `P` (from `space_group.sym_ops`) and form a generic
  `P`-invariant metric by averaging a fixed, generic positive-definite `G₀`
  over them: `G_P = (1/|P|) Σ_{w∈P} wᵀ G₀ w`. `H_lookup` is the set of
  candidates preserving `G_P` (relative tolerance 1e-9). This yields the
  crystal family's cell symmetry in **whatever setting the CIF uses**
  (unique axis, hexagonal vs rhombohedral axes, non-standard cells) with no
  per-setting tables. It also works for groups promoted by `detect_symmetry`,
  whose `SpaceGroup` has `crystal_system = None`, so a table keyed on crystal
  system would break there.
- **Metric test (`H_metric`).** The candidates preserving the actual cell
  metric `G = AᵀA` (relative tolerance 1e-9). This catches cells whose shape
  is more symmetric than their space group requires (e.g. a monoclinic cell
  with β exactly 90°).
- **Candidates.** Integer 3×3 matrices with entries in {-1, 0, 1} and
  `det = ±1`. This set is complete for the conventional settings that
  `SpaceGroup` data carries. To stay safe on skewed non-standard cells, both
  tests run in a Niggli-reduced basis (`spglib.niggli_reduce`) and the
  results are mapped back through the integer change of basis.
- **Combine.** `H = ⟨H_lookup ∪ H_metric⟩`. If the closure is not a finite
  group of order ≤ 48 (only possible with an inconsistent cell), fall back
  to `H = H_lookup` and warn.
- **Measured:** on all 26 fixtures `H_lookup == H_metric` and both match the
  crystal-family orders (P-1 2, P2₁/c 4, Pbca 8, P-42₁c 16, R-3c and
  P6₃/mmc 24, cubic 48). Kept ops satisfy the test to ≤ 4e-16; rejected
  ones miss by ≥ 4e-3.

`H` depends only on the crystal, so compute it once per `bind` and share it
across references.

### 2. Dressing group `D` per reference

- Start from `W` and `W_tr` exactly as today. `W` goes in unconditionally:
  ice's own site ops fit its reconstructed H atoms only to about 2.7e-4 Å
  and would fail a 1e-6 Å test.
- For each `h ∈ H ∖ W`, accept it if:
  - **Monomer test.** Applying `x ↦ h (x − c) + c` to the reference's
    fractional coordinates puts every atom within `EXACT_ATOL = 1e-6` Å
    (Cartesian) of an atom of the same element, and
  - **Translation test.** `12 (I − h) c` lies within 1e-9 of an integer
    vector. The rounded value becomes the twelfths translation, using the
    same `np.rint` pattern as `context.py:96`.
- Exact-but-not-twelfths ops are skipped, with one `RuntimeWarning` per
  reference naming the count. This errs towards keeping duplicates. No
  fixture triggers it.
- Assert that `D` is closed under composition. Products of exact symmetries
  are exact, so a failure indicates a bug, not data.
- **Measured:** `|D| = |W|` on 24 fixtures; triazine 6→12, CO₂ 6→12.
  Every accepted op fits to ≤ 1e-15 Å, and no rejected op is closer than
  1e-3 Å.

### 3. Orbit key

`orbit_labels` / `_group_keys` / `_dressed_minima` are unchanged. They read
`D` (stacked as `(D, D⁻¹, D_tr)`) where they read the `W` stack today.
Improper ops in `D` give enantiomer merges for free.

### 4. Removed

- `congruence.py` (gate, spread window, `merge_congruent`, `same_multimer`)
  and `matchers.py` (`anchored_match` has no other caller).
- The congruence block in `neighbors.generate_neighbors`.
- From `context`: `self_iso_by_ref`, `self_iso_tr_by_ref`,
  `check_congruence_by_ref`, and the `mon.point_group_seeded(...)` call. This
  also removes the pymatgen solver call from every `generate()` run.
- `PointGroup`, `Monomer.point_group()` and `Monomer.point_group_seeded()`
  stay as public library features with their own tests. Dedup simply stops
  depending on them.

### Data flow

```
bind(crystal, monomers)
  H = closure(H_lookup(space_group.sym_ops) ∪ H_metric(lattice))   # once
  per reference i:
    W_i, W_tr_i  (site symmetry, unchanged)
    D_i = W_i ∪ {h ∈ H \ W_i : fixes monomer ≤1e-6 Å, 12(I−h)c ∈ ℤ³}
    dress_stacked_by_ref[i] = (D_i, D_i⁻¹, D_tr_i)
generate_neighbors
  candidates → orbit_labels (reads dress_stacked_by_ref) → buckets → merge
  (no congruence pass)
```

## Components / changes

| File | Change |
|---|---|
| `work/dedup/dressing.py` (new) | `cell_symmetry(space_group, lattice)` returning H, and `dressing_group(monomer, W, W_tr, H)` returning `(D, D_tr)`. Pure functions with no module state, so they are unit-testable without `bind`. |
| `work/dedup/context.py` | `bind` calls `dressing.py` and stores `dress_stacked_by_ref` (replacing `W_stacked_by_ref`); keep `W_lin_by_ref` / `W_tr_by_ref` meaning true site symmetry; drop `self_iso_by_ref`, `self_iso_tr_by_ref`, `check_congruence_by_ref` and the point-group call. Update the module docstring. |
| `work/dedup/invariants.py` | `_dressed_minima` / `_group_keys` read `dress_stacked_by_ref`; docstrings say "dressing group" rather than site symmetry. Delete the congruence-only helpers `fingerprints_match_many`, `is_mono_reference`, `base_reference`, and the `integer=False` branch of `relative_transforms` (the `integer` parameter goes with it). Keep `relative_transforms` (integer), `cluster_atoms_cart`, `distance_fingerprint`: the tests use them as specification and referee. |
| `work/dedup/congruence.py`, `work/dedup/matchers.py` | Delete. |
| `work/neighbors.py` | Remove the congruence import and block. |
| `work/generate.py` | Linear-molecule warning text: still incomplete, but no longer mention the Cartesian pass. |

## Behaviour changes

- **Unchanged results** on every fixture except CO₂. Measured with the
  congruence pass disabled (unique count, total multiplicity, multiplicity
  multiset):
  - triazine: dimers 20 Å 61/418, dimers 35 Å 266/2082, trimers 12 Å 123/2103
  - ammonia: dimers 20 Å 183/1075
  - benzene: dimers 20 Å 116/366
- **CO₂ improves**: 86 → 81 unique dimers (total multiplicity 832 in both).
  All 751 exact-pass duplicates match their representative's NRE and sorted
  distances to 1e-8. Pairs of surviving representatives with identical
  distance fingerprints drop from 6 to 1; the remaining pair falls under the
  existing linear-molecule warning.
- **No tolerance anywhere per cluster**, so results no longer depend on
  cluster separation.
- **Cost**: `_dressed_minima` scales with |D_a|·|D_b|, roughly 4× on that
  step for triazine and CO₂, unchanged elsewhere. `bind` gets cheaper (no
  point-group detection). The removed pass ran on every gated fixture.

## Error handling

- Exact op whose translation is not in twelfths: skip it, one `RuntimeWarning`
  per reference.
- `⟨H_lookup ∪ H_metric⟩` not a finite group of order ≤ 48: use `H_lookup`
  and issue a `RuntimeWarning`.
- `D` not closed, or `W ⊄ H`: `AssertionError`. Both are theorems, so a
  failure means a bug.

## Testing

New:
- `H` per fixture equals the crystal-family order table above, and
  `H_lookup == H_metric` on all fixtures.
- Triazine on its hexagonal axes: `|H| = 24`, and all 6 extra `D` ops lie
  outside the R-lattice group. This guards against a future switch to
  Bravais-lattice symmetry.
- `|D|` per fixture: triazine 12, CO₂ 12, every other fixture `|W|`; `W ⊆ D ⊆ H`.
- `D` is a group, and each element fixes its monomer to ≤ 1e-6 Å with an
  integral `12 d`.
- Golden regression: the counts and multiplicities under Behaviour changes,
  recorded from the current pipeline before the change (CO₂ recorded as the
  new value). Mark triazine 35 Å as `slow`.
- Over-merge referee (NRE + CHSEV, existing test pattern) extended to
  triazine and CO₂.
- No lost merges: on triazine and CO₂, every merge the old pipeline made
  is also made by the new one (old partition refines new). Recorded once
  against the pre-change code and checked via the referee.

Rewrite or remove (they assert removed context fields):
- `test_dedup.py`:
  - `test_bind_sets_site_symmetry_and_gate`: drop the gate assertions.
  - `test_congruence_gate_opens_iff_point_group_exceeds_site_symmetry`: remove.
  - `test_self_isometry_linear_part_is_not_a_symmetry`,
    `test_bind_dressing_translations_vanish_for_benzene`,
    `test_bound_dressing_groups_satisfy_the_containment_theorem`,
    `test_every_dressing_is_a_symmetry_of_its_monomer`: retarget to `D` /
    `D_tr`.
  - The orbit-key helper that sets `context.W_stacked_by_ref` directly
    (around `test_dedup.py:68`): rename to `dress_stacked_by_ref`.
  - Callers of `relative_transforms(g, integer=True)`: drop the argument.
- `test_multiref.py::test_bind_multi_reference_state`: assert on the
  dressing stack length instead of `check_congruence_by_ref`.
- `conftest.py`: remove the stale `test_full_congruence_check_is_unconditional`
  entry from `_SLOW_TESTS` (no such test exists).
- `test_self_isometries.py` keeps testing `PointGroup` as a library feature.
  The one use of `context.W_lin_by_ref` stays valid.

## Out of scope

- Molecular symmetry outside `H` (non-lattice exact ops, C5 / icosahedral).
- Complete dedup for linear molecules (continuous symmetry).
- Cross-reference (Z' > 1) congruences. These were not searched before
  either.
- Updating `docs/dedup_correctness.tex`. It should gain a short section on
  `D` (dressing group = site symmetry ∪ exact cell-symmetry ops) as a
  follow-up.
- Changing `PointGroup.detect` itself.
