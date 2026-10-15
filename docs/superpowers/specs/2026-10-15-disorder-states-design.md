# Disorder → Ordered States — Design

**Date:** 2026-10-15
**Status:** approved decisions, plan in `docs/superpowers/plans/2026-10-15-disorder-states.md`

## Problem

`from_cif` currently resolves partially occupied sites by policy: `error`
(default), `ignore` (keep every site — builds impossible molecules), or `drop`
(keep only fully occupied sites — builds fragments). On the two disordered
structures we have, both non-error policies are silently wrong:

| CIF | `ignore` | `drop` |
|---|---|---|
| ice (Goto 1990, P6₃/mmc) | H₄O | O |
| ritonavir (`data/cif/2207870.cif`, C2) | C46H64N10O6S4 | C22H24N2O4 fragment |

The literature practice is to run calculations on each **ordered state**
the disorder admits. This feature makes `from_cif` yield one `Crystal` per
ordered state.

## Two kinds of disorder

1. **Explicit groups** (`_atom_site_disorder_assembly` /
   `_atom_site_disorder_group`, positive group numbers; SHELX `PART n`).
   Each assembly is a set of mutually exclusive alternatives. A state is one
   group per assembly. The state keeps the declared space group: the ASU is
   filtered, then expanded as usual.
   *Ritonavir:* assemblies A (6 atoms, 0.688/0.312) × B (30 atoms,
   0.520/0.480) → 4 states, all C37H48N6O5S2, all C2.

2. **Symmetry-generated disorder** (partial sites with no group tag, or
   negative SHELX groups). The alternatives are symmetry images of one
   another, so no ASU choice can produce an ordered structure under the
   declared group. A state is a subset of the partial atoms in the full P1
   unit cell; its true (lower) symmetry is recovered with
   `Crystal.detect_symmetry`.
   *Ice:* 18 ice-rule configurations in the 4-molecule cell → 2 states under
   the parent group: Cc (#9, 12/18) and Cmc2₁ (#36, 6/18), both Z′=2.

Both can occur in one block: explicit choices are applied first, and any
remaining partial sites go through the symmetry path.

## Decisions (user, 2026-10-15)

| # | Decision |
|---|---|
| 1 | **API shape: metadata on `Crystal`.** `from_cif` keeps yielding `(Crystal, Monomer)`; each state is one more yield. `Crystal.state` is a `DisorderState` (or `None` for an ordered CIF), and must survive `detect_symmetry`. |
| 2 | **Scope:** explicit groups **and** symmetry-generated disorder. |
| 3 | **Cell:** enumerate in the CIF's own cell only. No supercells (Bernal–Fowler ice is out of reach; ritonavir mixed-neighbour arrangements are out of scope). |
| 4 | **Weighting:** report per-state results only; no combination logic. Each state carries the weight the CIF implies, as information. |
| 5 | **Default policy is enumeration**, and the user is told: a warning naming the disordered sites and the number of states yielded. `error` / `ignore` / `drop` remain available. |
| 6 | **Assemblies are independent; no cap** on the number of states. |
| 7 | **Disk writing untouched** (`write_inputs` is being overhauled separately). |

## Rules

### Which sites are disordered
Every site with `_atom_site_occupancy < 0.999` (unchanged from today).
`_atom_site_occupancy` is the chemical occupancy by the CIF standard.

*Rejected:* reinterpreting `p × |G| = m` as an old SHELX special-position
site-occupation factor. It misfires on ice: H2 lies on a mirror (12 images
of 24 operations) and is genuinely half occupied, so `0.5 × 24 = 12`
would wrongly mark it fully occupied. The numbers cannot distinguish the
two cases.

### Explicit path
- Group values are normalised (`1.0` → `"1"`); `.`/`?` mean untagged.
  Negative groups are routed to the symmetry path.
- Missing or `.` assembly with a tagged group → one unnamed assembly.
- A tagged site that is fully occupied is kept in every state (ritonavir
  H11 is tagged B/1 with occupancy 1; selecting by tag alone loses it).
- Group weight = mean occupancy of its partial sites. If an assembly's group
  weights sum to less than 0.98, an extra "absent" choice with weight
  `1 − sum` is added (e.g. a half-occupied solvent with one group).
- State weight = product over assemblies.

### Symmetry path
- Expand the ASU into the P1 cell, tracking each cell atom's ASU site.
- Each disordered site must contribute a whole number `p × m` of atoms to
  the cell, otherwise the cell has no ordered state → `ValueError` naming
  the site.
- Two atoms are **bonded** if `d ≤ 1.2 (r_cov,i + r_cov,j)` (same rule as
  `Crystal._bfs`), except H–H pairs, which are never bonded.
  Two atoms **clash** if not bonded and `d < ½ (r_vdW,i + r_vdW,j)`.
  (Ice: the two H sites on one O···O contact are 0.76 Å apart — just
  above the 0.74 Å H–H bond cutoff — so without the H–H rule the ice rules
  would hinge on 0.02 Å.)
- **Skeletons** are the connected components of fully occupied atoms. A
  selection is valid when no chosen atoms clash and, after adding them,
  every connected component contains exactly one skeleton and skeletons in
  the same parent-symmetry orbit end up with the same formula. For water
  this is exactly the Bernal–Fowler ice rules.
- Every partial atom must be connected to some skeleton through the
  all-sites bond graph; otherwise (a whole molecule disordered over
  symmetry images) → `ValueError`, not supported.
- Valid selections related by a parent operation are one state; its weight
  is orbit size / number of valid selections. States are ordered by weight
  (descending), then by canonical selection.
- Each state is built as a P1 `Crystal` whose ASU is the selected cell
  atoms, then passed through `detect_symmetry()` before being yielded.

### Labels
- Explicit: `"A1 B2"` (assembly name + group; unnamed assembly → group
  only; absent → `"A-"`).
- Symmetry: `"config 1"`, `"config 2"`, … in state order.
- Both: `"A1 B2 / config 1"`.

## Out of scope
Supercells, mixed-neighbour states, weighting/combination of results,
disk-writing changes, cocrystal/solvent handling in `_find_molecules`
(a half-occupied solvent's "present" state is still dropped by the
max-size molecule filter — existing limitation).
