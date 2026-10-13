# Test suite

337 tests over the v2 pipeline: CIF in, deduplicated multimers out, QC decks on
disk. The v1 suite is gone — all 31 files were removed in `602943c` after they
had been failing for the length of the branch against an API, a module and a
fixture tree that no longer exist.

## Running

```bash
pytest                    # everything, ~134 s
pytest -m "not slow"      # ~55 s, still touches every module
pytest -m slow            # just the 15 expensive cases
pytest tests/test_dedup.py -k over_merge      # one concern
pytest --durations=15     # find the next thing worth marking slow
```

`slow` is applied in `conftest.py`, not with decorators at each site, because
the cost is usually a property of the *fixture* rather than the test: the same
test is fast on benzene and slow on hexamine, whose point-group detection has to
close 42 candidate operations down the tolerance ladder. Keeping the list in one
place also means the fast run's contents are visible at a glance. Use it for a
quick signal; run the full suite before merging.

## Where things are

| Module | Cases | What it holds you to |
| --- | ---: | --- |
| `test_self_isometries.py` | 108 | Molecular point-group detection: closure under composition, valid group order, inverses present. Plus site-symmetry seeding, which is what makes `W_lin ≤ S` true by construction rather than by luck — including adamantane's `S4` rotoreflections and ice's site symmetry, neither of which the detector can reach on its own at any tolerance. Plus the optional external solver: that it names the idealised group, that consulting it can never *shrink* the detected one, and that its contribution is still a group. |
| `test_dedup.py` | 85 | The deduplication core. Integer-matcher correctness and its dressing translations; reference-compatibility of the permutation; cross-reference merges at `Z′ = 2` and `Z′ = 3`; the routing gate between the integer, Cartesian and cloud matchers; Kabsch conditioning; over-merge and multiplicity conservation across `N = 2…5`; the process pool. |
| `test_generate.py` | 36 | End-to-end regression counts at four orders — 14 dimer, 3 trimer, 4 tetramer, 3 pentamer rows — each also asserting the chemistry oracle agrees. Plus count monotonicity in `R`, and the `N ≥ 2` input gate. |
| `test_cif_parsing.py` | 47 | CIF ingestion across 13 space groups spanning every crystal system, several in more than one setting. Space-group number, Hall symbol, lattice parameters, ASU atom count. Plus field-name flexibility: CIF1 vs CIF2 spellings, Hall-only, H-M-only (with punctuation variants), operation-list-only, and a listed-operations-vs-declared-symbol contradiction. Plus the partial-occupancy guard and its three policies. |
| `test_writers.py` | 25 | QC input generation: writer registry, the full Psi4 deck (fragments, charge/multiplicity, geometry, options, BSSE), and `write_inputs` directory layout and naming. |
| `test_validate.py` | 14 | The chemistry oracle itself — see below. |
| `test_multiref.py` | 12 | Multi-reference enumeration: `get_references` at `Z′ = 1` and `Z′ = 4`, per-member `ref_indices`, per-reference binding, and the physical ammonia dimers under two distance conventions. |
| `test_symmetry_detect.py` | 11 | spglib detection: P1 → P2₁3 promotion, no-change for benzene, graceful degradation on failure, pickling of detected groups, and that detection does not move the generated counts. |

## Two things worth knowing before you edit

**`test_validate.py` is not testing a feature.** `count_chemistry_classes` is
the *referee* — the assertion `count_chemistry_classes(mers)[0] == len(mers)`
appears throughout `test_dedup.py`, and every locked count in
`test_generate.py` rests on the oracle having agreed when it was captured. An
untested referee is worth nothing: replacing that function with "every multimer
is its own class" used to leave the whole suite green but for one xfail. The
tests there pin it from both sides, and the homometric case pins the CHSEV half
specifically, without which the oracle degenerates to a distance comparison and
can no longer see an over-merge that clears the deduper's own fingerprint gate.

**Under-merging is visible; over-merging is silent.** A missed merge splits one
class across several representatives — it costs redundant QC jobs but leaves the
lattice sum exactly right, and `n_classes < len(mers)` shows it. A false merge
deletes a distinct interaction and inflates another's multiplicity, corrupting
the sum by `(m_B/N)(E_A − E_B)`, and the usual equality check *cannot see it*
because both sides fall together. That asymmetry is why
`test_no_over_merge_every_discarded_cluster_matches_a_survivor` exists and why
tolerances in point-group detection should always be biased toward finding fewer
operations.

## Fixtures and shared helpers

Crystals live in `x23/` (23 benchmark molecular crystals) and `other/`
(fomepizole, `Z′ = 3`; hydroxyurea). `conftest.py` resolves them by bare name so
no test needs to know which directory a crystal is in:

```python
from conftest import GENERATE_KW, cif, crystal, load

xtal = crystal("benzene")                  # just the Crystal
xtal, reference = load("benzene")          # both halves of the first block
path = cif("fomepizole")                   # resolves under other/ too
mers = cle.generate(xtal, 2, 8.0, **GENERATE_KW)
```

`GENERATE_KW` is `use_origin=True, verbose=0` — the anchor-relative
distance convention and no progress bars. A test that passes something else is
making a point about the convention, and should say which.

Every fixture is laid out as `<name>/<name>.cif`, and `cif()` enforces that: a
directory whose CIF is named something else fails with a message telling you to
rename it, rather than resolving quietly. Add a fixture by creating
`x23/<name>/<name>.cif` — nothing else needs to know about it.

## Conventions

- **Locked counts carry their provenance.** A bare number in a parametrize table
  is unreviewable. Each says how it was captured and what confirmed it, so a
  future failure can be read as "the dedup changed" rather than "the number
  drifted".
- **A test should be able to fail.** Assert the specific value, not that a
  `bool` came back. Where a test's premise could quietly stop holding — a
  fixture that might get collapsed to `Z′ = 1` by symmetry detection, a
  homometric pair that might get edited — assert the premise too, so the test
  cannot pass while testing nothing.
- **Say what breaks, not what was expected.** Assertion messages name the
  consequence ("this is an OVER-MERGE: a distinct multimer was deleted and
  another's multiplicity inflated"), because that is what the reader needs.

## Two fixtures that need saying out loud

**ice is proton-disordered.** Both hydrogen sites carry occupancy 0.5, so
reading every site gives each oxygen four hydrogens at 0.85–1.00 Å. `from_cif`
refuses that by default now; the tests opt back in via `conftest.DISORDERED`,
because ice is a useful high-symmetry fixture (P6₃/mmc) and the dedup machinery
it exercises is orthogonal to whether its monomer is chemically real. Its dimer
counts are **disorder-averaged, not physical water dimers** — do not read them as
statements about ice.

**succinic_acid's CIF is named `succinic_Acid.cif`** — no longer true, it was
renamed, but if a fixture ever again fails to follow `<name>/<name>.cif`,
`conftest.cif()` will refuse it with a message telling you to rename.

## The two remaining xfails

Both are the same known limitation, and both are `strict=True` so a fix surfaces
as a failure rather than passing quietly:
`test_dedup.py::test_linear_monomer_dedup_is_complete` and the
`carbon_dioxide` row of `test_dimer_count`. A linear molecule's twist about its
own axis moves no atom, but a placement stores a full 3×3 rotation, so that
physically vacuous coordinate is still compared and two placements of the same
molecule fail to match — 39 dimers come back where 36 is correct. Both hold 36
as the target. See the limitation note in `crystalatte.work.dedup`.
