# Compute backends: QCEngine, QCFractal, and input files

## Context and intent

CrystaLattE v2 generates symmetry-unique multimers (`cle.generate`) but its
only route to energies is `work/setup.py::write_inputs`, which writes Psi4
input files and stops. Nothing runs calculations, tracks them, or turns
results back into a lattice energy.

The goal is to make running calculations a first-class, program-agnostic part
of the package by building on the MolSSI QCSchema stack:

- **QCEngine** runs individual QC calculations in any supported program.
- **QCManyBody** expands a fragmented molecule into the sub-calculations a
  many-body/BSSE treatment needs and analyzes their results.
- **QCFractal / QCPortal** let users manage large campaigns on their own
  servers.

Two execution paths are primary — **local (QCEngine)** and **managed
(QCFractal)** — and writing input files is kept as a supported fallback.
Everything is driven from a user's standalone Python script through the
`crystalatte` API; there are no `.cle` input files and no CLI (those are v1
artifacts).

## Goals

1. A single `ComputeSpec` describing *what* to compute (program, method, basis,
   BSSE treatment, keywords), reused unchanged across every backend.
2. A `Backend` ABC with `submit` / `status` / `collect`, so a user script looks
   the same whichever backend it uses. Switching paths is a one-line change.
3. `LocalBackend`: one machine (laptop → single cluster node), resource-aware,
   parallel, and resumable after a crash.
4. `FractalBackend`: builds `ManybodyDataset`s in the user's own
   `PortalClient`, one dataset per multimer order N, then hands execution to
   QCFractal entirely.
5. `InputFileBackend`: replaces `write_inputs`, using the existing
   `io/writers/` plugins.
6. A shared `Results` object and one lattice-energy assembly routine, used by
   both computing backends.
7. Stable multimer identifiers so reruns of `generate` map onto previously
   computed or submitted work.

## Non-goals

- Deploying or managing QCFractal infrastructure (servers, databases, compute
  managers, workers). Users bring a `PortalClient`; `FractalSnowflake` works
  for free because it yields an ordinary client.
- Multi-node execution in `LocalBackend`. Anything beyond one machine goes
  through QCFractal.
- Parsing output files written by externally run input files.
- Gradients, Hessians, or anything other than energies.
- Monomer deformation / relaxation energy.
- Changing `from_cif` / `generate` style: they stay plain functions.

## Anti-goals (would make this a failure even if it works)

- A lattice energy silently computed from incomplete data.
- Local and Fractal paths producing different numbers for the same spec.
- Re-running `generate` and having previously computed work orphaned because
  multimer names shifted.
- Users having to hand-build datasets or hand-match records back to multimers.

## User-facing shape

```python
import crystalatte as cle

crystal = next(cle.from_cif("benzene.cif"))
multimers = cle.generate(crystal, N=2, R=10.0) + cle.generate(crystal, N=3, R=8.0)

spec = cle.ComputeSpec(
    program="psi4", method="mp2", basis="aug-cc-pvdz", bsse="cp",
    keywords={"scf_type": "df"},
)

backend = cle.LocalBackend(scratch="bz_run/", ncores=8, memory="16 GiB")
# backend = cle.FractalBackend(client, name="benzene", tag="bigmem", priority="high")
# backend = cle.InputFileBackend("inputs/")

backend.submit(multimers, spec)
backend.status()
res = backend.collect()          # raises IncompleteResultsError if anything is missing

print(res.lattice_energy)        # total
print(res.by_order)              # {2: ..., 3: ...}
res.table                        # per-multimer data
```

All new public names are exported from `crystalatte/__init__.py`. Names above
are the agreed shape; exact keyword names inside each constructor are the
builder's call.

## Design

### Module layout (builder may adjust)

```
src/crystalatte/compute/
    __init__.py       # ComputeSpec, backends, Results, errors
    spec.py           # ComputeSpec
    ids.py            # stable multimer identifiers
    results.py        # Results, IncompleteResultsError, lattice-energy assembly
    backends/
        base.py       # Backend ABC
        local.py      # LocalBackend
        fractal.py    # FractalBackend
        inputs.py     # InputFileBackend
```

### `ComputeSpec`

A frozen pydantic v2 model (user decision: pydantic for persisted boundary
objects `ComputeSpec` and `GenerationInfo`; dataclasses elsewhere) holding the model
chemistry only — never the execution path or resources:

- `program`, `method`, `basis`, `keywords: dict`, `bsse` (default `"cp"`;
  allowed values those QCManyBody supports: `cp`, `nocp`, `vmfc`).

**Tight defaults, overridable (agreed).** `ComputeSpec` merges
program-specific CrystaLattE defaults under the user's `keywords` (user keys
win). For Psi4, copy v1's hard-coded settings from `master` exactly
(`e_convergence=10`, `d_convergence=10`, `freeze_core`, and the DF types v1
used). Nonadditive 3-body terms are ~1e-5 Eh and get multiplied by
multiplicity, so loose program defaults would visibly shift totals. Defaults
are stored in the spec's final keywords, so the local, Fractal and input-file
paths all see identical keywords, and the Fractal specification records them.
Programs without a curated default set get none, with a one-time warning
recommending tight convergence. `use_defaults=False` turns merging off.

Converters:

- `to_qcmanybody(N)` → arguments for `qcmanybody.ManyBodyCore`
  (`bsse_type`, `levels={1: mc, 2: mc, ..., N: mc}`, `return_total_data=False`,
  `supersystem_ie_only=False`, `embedding_charges={}`) plus a QCEngine
  `AtomicInput` template (`driver="energy"`, model, keywords).
- `to_portal(N)` → `qcportal.manybody.ManybodySpecification(program="qcmanybody",
  levels=..., bsse_correction=[bsse])`. Use the same contiguous form
  `{1: qcspec, ..., N: qcspec}` unless the QCFractal service is verified to
  accept a sparse `{N: qcspec}`.
- Writer config for `InputFileBackend`, replacing today's loose `config` dict.

`ManyBodyCore` requires levels to be contiguous from 1 to N
(`ValueError: Levels must be contiguous from 1 to N`); only the higher-level
`ManyBodyComputer` fills gaps. The highest level sets the maximum body order,
which is why one spec per order is required (see Fractal datasets below).

**Physics check:** with `bsse="cp"`, QCManyBody evaluates every subcluster in
the full multimer's basis. The quantity CrystaLattE needs from each N-mer is
its nonadditive N-body energy, which is QCManyBody's
`cp_corrected_{N}_body_contribution_to_energy` (present in qcmanybody 0.8,
`utils.py`) when `max_nbody == N == nfragments`. For dimers this is the CP interaction energy. This matches v1's
definition (Psi4 nbody driver), so results are comparable to v1.

### Stable multimer identifiers

Every multimer needs an ID that is identical across separate runs of
`generate` on the same CIF and settings. It is used as the local cache key and
the Fractal entry name. Today multimers are only identified by list position,
which is not stable.

**Decision (Claude's call, after review):** the ID is geometry-derived:
`f"{order_name}-{hash[:16]}"`, where `hash` is the QCElemental molecule hash
(`Molecule.get_hash()`) of `multimer.to_molecule()`. The dedup orbit labels
are not usable: `orbit_labels` returns per-run bucket numbers and
`lex_codes` packing is only comparable within one call, and the linear-group
dot-product key differs from the frame key.

Why geometry: it is deterministic for identical code, CIF and settings, it is
self-verifying (an ID can never point at a different geometry, so no
separate check layer is needed), and QCFractal already deduplicates
molecules by the same hash. If a future dedup change picks a different
representative for an orbit, the ID changes and the work is redone: this
fails safe and never reuses a wrong energy. Stability across crystalatte
upgrades is not guaranteed.

Requirements: a valid QCFractal entry name and filename, exposed on `Multimer`
(e.g. `Multimer.id`). The crystal/settings fingerprint lives in the manifest
and dataset metadata (see below), not in the ID.

### `Backend` ABC

```python
class Backend(ABC):
    @abstractmethod
    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None: ...
    @abstractmethod
    def status(self) -> Status: ...
    @abstractmethod
    def collect(self, *, partial: bool = False) -> Results: ...
```

Each backend implements all three. Behavioural differences are documented in
docstrings:

- `LocalBackend.submit` blocks until done; `FractalBackend.submit` returns
  once datasets are created and submitted.
- `InputFileBackend.status` / `.collect` raise `NotImplementedError` with a
  message saying this backend only writes files.

A backend stores what it needs between calls (scratch directory, client,
dataset name), including the multimers and spec from `submit`. `collect` on a
fresh backend object must also work from persisted state alone (local scratch
manifest, or the Fractal datasets), so a user can collect in a separate script
or session. `Status` is a small object (or dict) with per-N counts of
complete / pending / failed and a readable `__str__`.

Backend lookup by name through a registry, like `io/writers/`, is optional.

### `LocalBackend`

Single machine, built on `ManyBodyCore` + `qcengine.compute`. It does **not**
use QCManyBody's all-in-one `ManyBodyComputer`, because running the
sub-calculations itself gives finer-grained parallelism and checkpointing.

Flow per `submit`:

1. For each multimer, build `ManyBodyCore(multimer.to_molecule(), ...)` from
   `spec.to_qcmanybody(N)`.
2. Enumerate sub-calculations with `core.iterate_molecules()` →
   `(mc, label, molecule)`.
3. Run each sub-calculation as a QCEngine `AtomicInput` in a process pool.
   Concurrency = `ncores // ncores_per_task`; each task gets a
   `task_config` with its cores, memory, and scratch.
4. Write each finished `AtomicResult` to the scratch directory as soon as it
   completes, keyed by multimer ID + label. On rerun, skip anything already
   on disk. Failed results are recorded as failures, not raised, so one
   failure doesn't abort the run.
5. When all sub-calculations of a multimer are present, run
   `core.analyze({label: {"energy": e}})` and store the multimer's
   nonadditive N-body energy.

Resources: `ncores`, `ncores_per_task`, `memory` (total or per task, builder's
call), `scratch`. Defaults come from `qcengine.get_config()` so a laptop user
passes nothing beyond `scratch`. The scratch directory also holds a manifest
(spec, generation settings, multimer IDs, multiplicities, order) so `collect`
and `status` work from disk alone. Submitting a different spec into an existing scratch
directory is an error.

### `FractalBackend`

`FractalBackend(client, name, *, tag=None, priority=None)` where `client` is
a user-supplied `qcportal.PortalClient`.

`submit`:

1. Group multimers by N.
2. For each N, get or create a `ManybodyDataset` named
   `f"{name}-{order_name(N)}s"` (e.g. `benzene-dimers`; reuse the existing
   `_order_name` helper). One dataset per N is the agreed design: a
   `ManybodySpecification`'s `levels` fixes the maximum body order, so a
   dataset holds only one order.
3. Add a specification derived from `spec.to_portal(N)` under a deterministic
   spec name.
4. Add one entry per multimer: `name = multimer ID`,
   `initial_molecule = multimer.to_molecule()`,
   `attributes = {"multiplicity": ..., "N": ..., "crystalatte_version": ...}`
   (plus anything assembly needs). Adding an entry that already exists is a
   no-op.
5. `dataset.submit(tag=..., priority=...)` and return.

Re-running `submit` is idempotent: it only adds what is missing.

**Active set (Claude's call, after review):** both computing backends
accumulate computed work (union), but each `submit` records, per N, an
*active set*: the multimer IDs submitted plus the generation settings (R,
symprec, detect_symmetry, CIF/crystal fingerprint, crystalatte version).
`collect` sums only the most recent active set per N. For the local path it
lives in the manifest; for Fractal it goes in the dataset's `extras`/metadata.
This lets cutoff-convergence studies (R=8 → R=10) reuse earlier results while
never summing leftover entries from an earlier, different run. To record R
and the other settings, `generate` stores them on each returned `Multimer`
(or a small `GenerationInfo` attached to it). If the CIF/crystal fingerprint
changes under the same backend name, `submit` raises. Submitting a
different spec under the same name adds a second specification to the same
datasets, and `collect` then needs to know which spec to use. The builder
decides between rejecting that case and taking a `spec` argument in
`collect`, and must document the choice.

Extra methods on `FractalBackend` only (not on the ABC): `resubmit_errors()`
to reset errored records.

`collect` reads the records, pulls each N-mer's nonadditive N-body energy
from the manybody record's results, and joins it to multiplicities from entry
attributes. It works on a fresh `FractalBackend(client, name)`.

### `InputFileBackend`

Replaces `work/setup.py::write_inputs`, which is removed along with its
export from `crystalatte/__init__.py`. Agreed: this is a breaking change,
acceptable because v2 is unreleased.

`InputFileBackend(output_dir)`; `submit(multimers, spec)` writes one file per
multimer through the `io/writers/` registry selected by `spec.program`, in the
existing per-order subdirectory layout, now named by stable multimer ID. Writers
take `ComputeSpec` (or its derived config) instead of a free-form dict.
`tests/test_writers.py` moves to the new entry point.

### `Results` and incomplete data

`Results` holds per-multimer rows (ID, N, multiplicity, nonadditive energy,
weighted contribution, status) and exposes `lattice_energy`, `by_order`, and
`table`. `table` is a plain list of dicts or a dataclass list, so pandas is not
a dependency; a `to_dataframe()` convenience that imports pandas lazily is
fine.

Agreed behaviour:

- `collect()` raises `IncompleteResultsError` if any multimer is failed or
  pending. The message gives counts per N, failed IDs with a short error
  excerpt, and pending IDs. The exception carries `.missing`.
- `collect(partial=True)` returns a `Results` whose per-multimer data and
  `.missing` are filled in, but whose `.lattice_energy` and `.by_order` raise
  `IncompleteResultsError`, so a partial total can never be used by accident.

### Lattice-energy assembly

One function, used by every computing backend:

    E_latt = Σ_N Σ_{multimers i of order N}  w_i · ΔE_i^(N-body)

**Normalization: per formula unit (agreed).** The target is a value
directly comparable to a periodic-DFT cocrystal lattice energy
`E_latt = E(bulk)/Z − E(A) − E(B)`, where Z is the number of formula units
per cell. The MBE equivalent is

    E_latt = (1 / Z_fu) · Σ_N Σ_{N-mer orbits} (copies of the orbit per cell) · ΔE^(N-body)

where `Z_fu` is the number of formula units per cell, defined (Claude's call)
as the gcd of the per-cell molecule counts of each chemical species, with an
optional user override for unusual stoichiometries. For a Z'=1
single-component crystal this reduces to per molecule and matches v1
(`w = replicas / N`). Per formula unit stays well-defined when a molecule sits
on a special position, where "per asymmetric unit" does not.

The builder derives "copies per cell" from what `Multimer.multiplicity`
counts in v2. Reviewer reading of the code (`work/neighbors.py`,
`dedup/equivalence.py`): `multiplicity` counts copies across the anchors of
every reference molecule. Z_fu, per-species counts, and the species
composition are persisted in the manifest / dataset metadata so `collect`
works from stored state alone.

**Normalization correctness test (required, no QC program):** a fake backend
scores each multimer with a toy pairwise potential (e.g. Lennard-Jones on
atoms). The symmetry-reduced per-formula-unit result must equal a brute-force
sum over the same clusters enumerated on the P1 expanded cell, divided by
Z_fu. It must pass for Z'=1, Z'>1, a special-position case, and a cocrystal.

Comparability caveat: periodic-DFT `E(A)`, `E(B)` are relaxed gas-phase
monomers; the MBE sum covers interactions at crystal geometry only. Monomer
deformation (relaxation) energy is out of scope here, as in v1; document it
in `Results`.

Validation (Claude's call): a one-off manual check against the v1 benzene
psithon outputs on `master` (`Tests/Analysis/Benzene/*.out`,
`Tests/Analysis/Test/Results.csv`), recorded in the PR. Not a CI test.

### Packaging

Optional extras in `pyproject.toml`:

- `crystalatte[local]`: `qcengine`, `qcmanybody`
- `crystalatte[fractal]`: `qcportal`

Backends import these lazily when constructed and raise an `ImportError` with
the `pip install "crystalatte[...]"` hint. The base install stays light.
QC programs (Psi4 etc.) are the user's responsibility. Pin minimum versions
known to work (verified while drafting: `qcmanybody` 0.8.0, `qcportal` 0.71,
`qcengine` 0.52.0).

**Core dependency bump (forced):** all three require
`qcelemental>=0.50,<0.70`; the dev env currently has 0.30.1. The core
`qcelemental` pin rises accordingly. This touches every user via
`Monomer.to_molecule()` / `Multimer.to_molecule()` (qcelemental 0.50 moves to
pydantic v2 and splits QCSchema v1/v2 models), so it is its own first step
with the existing suite green before any backend work.

Note: the `qcmanybody` 0.8 tree contains both QCSchema v1 and v2 model
packages. The builder must pick one consistently with the installed
`qcelemental` / `qcengine` and with `Monomer.to_molecule()`.

## Testing

- Unit tests with no QC program: `ComputeSpec` converters, ID stability
  (same CIF → same IDs across two `generate` calls; different multimers →
  different IDs), assembly arithmetic, `IncompleteResultsError` behaviour,
  local scratch resume (fake `qcengine.compute` that records calls).
- Integration test for `LocalBackend` with a real program at a tiny level
  (e.g. HF/STO-3G Psi4 on a small dimer), skipped if the program is missing.
  Compare the CP interaction energy with one computed directly by
  QCManyBody's own runner.
- `FractalBackend` tests against `FractalSnowflake`, marked and skipped if
  `qcfractal` / Postgres are unavailable.
- Existing writer tests ported to `InputFileBackend`.

## Suggested build order

0. Bump `qcelemental`; keep the existing suite green.
1. Stable multimer IDs and generation settings on `Multimer`.
2. `ComputeSpec` (with defaults), `Results`, per-formula-unit assembly with
   the toy-potential normalization test, `IncompleteResultsError`.
3. `Backend` ABC + `InputFileBackend` (migrates `write_inputs`; low risk).
4. `LocalBackend`.
5. `FractalBackend`.

## Open questions

None blocking. Per-formula-unit normalization is the user's choice but they
flagged low confidence; the normalization correctness test above is the
safeguard and should be built before the backends rely on it.

## Left to the builder

Exact constructor keyword names, dataclass vs pydantic, `Status` shape,
scratch file format, spec naming in Fractal datasets, whether to add a backend
registry, and the ID format within the constraints above.
