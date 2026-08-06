# ORCA and Q-Chem QC input writers

## Context

`src/crystalatte/io/writers/` currently has a `QCWriter` abstract base class
and a single concrete implementation, `Psi4Writer`, which writes Psithon
input files for counterpoise-corrected (CP) interaction-energy jobs from a
list of `Monomer` objects. `work/setup.py::write_inputs()` looks up a writer
by name via a string registry (`get_writer()`) and writes one file per
`Multimer`, grouped into `dimers/`, `trimers/`, etc. subdirectories.

This spec adds two more writers, `OrcaWriter` and `QChemWriter`, following
the same plugin pattern and producing input files for the same kind of job
(CP-corrected interaction energy for an N-monomer multimer), using each
program's native mechanism for BSSE correction.

## Base class change

`QCWriter` gains a second abstract classmethod:

```python
@classmethod
@abstractmethod
def extension(cls) -> str: ...
```

Returns the file extension (no leading dot) conventional for that program's
input files. `Psi4Writer.extension()` returns `"inp"` (unchanged behavior).

`work/setup.py::write_inputs()` changes the hardcoded `.inp` suffix:

```python
path = order_dir / f"{_order_name(N)}_{idx:03d}.{writer.extension()}"
```

## Registry

`writers/__init__.py::_REGISTRY` gains two entries:

```python
_REGISTRY = {
    "psi4": Psi4Writer,
    "orca": OrcaWriter,
    "qchem": QChemWriter,
}
```

## Shared memory-parsing helper

Psi4's `set_memory()` accepts a free-form string like `"8 GB"` directly.
ORCA (`%maxcore`, MB per core) and Q-Chem (`MEM_TOTAL`, MB) both require an
integer number of megabytes. Add a small shared helper, e.g.
`src/crystalatte/io/writers/_util.py`:

```python
def memory_to_mb(memory: str) -> int:
    """Parse a memory string like '8 GB', '500 MB', '2GB' into an integer MB value."""
```

Supports `GB`/`MB` (case-insensitive, optional space), used by both new
writers. Raises `ValueError` on unrecognized units.

## OrcaWriter

`src/crystalatte/io/writers/orca.py`

- `name()` → `"orca"`
- `extension()` → `"inp"`
- Config keys: `method`, `basis`, `bsse_type` (default `"cp"`; any other
  value raises `ValueError` — CP is the only supported mode for now),
  `charge`, `multiplicity`, `memory`, plus arbitrary extras (same
  `_TOP_LEVEL_KEYS`-style split as `Psi4Writer`).
- ORCA has no automatic CP keyword, so for an N-monomer multimer the writer
  emits **N + 1 chained jobs** separated by `$new_job`:
  1. Supersystem job: every monomer's real atoms together in one
     `* xyz {charge*N} {multiplicity} ... *` block.
  2. One job per monomer `i` (for `i` in `1..N`): monomer `i`'s atoms as
     real atoms, every other monomer's atoms as ghost atoms (element symbol
     suffixed with `:`, e.g. `H:`, no nuclear charge/electrons — basis
     functions only), same overall `charge`/`multiplicity` as the
     supersystem job.
- Each job section repeats the keyword line `! {method} {basis}` (required,
  since `$new_job` resets state) and, if `memory` is given, a `%maxcore {mb}`
  line before the coordinate block.
- Extras (if any) go into a `%method ... end` block as `{key} {value}`
  lines, repeated per job.
- The file is not expected to compute the final CP-corrected interaction
  energy itself — a one-line comment (`# CP interaction energy = job 1 -
  sum(jobs 2..N+1)`) is written at the top of the file. Extracting energies
  from ORCA output remains the job of the downstream output parser, same
  division of responsibility as the rest of the pipeline.
- Charge/multiplicity simplification: total system charge is
  `charge * N` (sum of identical per-fragment charges); multiplicity reuses
  the single config value un-combined. This mirrors the same simplifying
  assumption already present in `Psi4Writer` (identical charge/mult applied
  to every fragment).

## QChemWriter

`src/crystalatte/io/writers/qchem.py`

- `name()` → `"qchem"`
- `extension()` → `"in"`
- Config keys: same as `Psi4Writer` (`method`, `basis`, `bsse_type` default
  `"cp"`, `charge`, `multiplicity`, `memory`, extras). `bsse_type` other
  than `"cp"` raises `ValueError`.
- Single job, one file, using Q-Chem's built-in automatic CP correction:

```
$molecule
{charge} {multiplicity}
{sym}  {x}  {y}  {z}
...
--
{charge} {multiplicity}
{sym}  {x}  {y}  {z}
...
$end

$rem
JOBTYPE sp
EXCHANGE {method}
BASIS {basis}
BSSE TRUE
MEM_TOTAL {mb}
{EXTRA_KEY} {extra_value}
$end
```

- `$molecule` fragment block matches Psi4's format: one `--`-separated block
  per monomer, each opening with its own `{charge} {multiplicity}` line
  (same identical-per-fragment simplification as Psi4Writer/OrcaWriter).
- `MEM_TOTAL` line omitted if `memory` not given (matches `Psi4Writer`'s
  `if memory:` guard).
- Extra config keys become additional `$rem` lines, one per key, with the
  key automatically uppercased (Q-Chem's `$rem` variables are conventionally
  all-caps) and the value written as-is.

## Testing

Following the existing pattern in `Tests/test_writers.py`:
- Registry: `get_writer("orca")` / `get_writer("qchem")` return the right
  types (extending existing case-insensitivity tests).
- `OrcaWriter`: file creation, `$new_job` count equals `N` (one per monomer)
  for a dimer/trimer fixture, ghost-atom suffix (`:`) appears for the
  non-real monomer in each per-monomer job, `%maxcore` line present when
  memory given, `ValueError` raised for non-`"cp"` `bsse_type`.
- `QChemWriter`: file creation, `$molecule`/`$rem` blocks present, `--`
  fragment separator, `BSSE TRUE` present, `MEM_TOTAL` present/absent
  correctly, extras uppercased in `$rem`, `ValueError` for non-`"cp"`
  `bsse_type`.
- `write_inputs`: dimer file for `"orca"` uses `.inp` extension, for
  `"qchem"` uses `.in` extension (extending existing directory-structure
  tests parametrized by software).

## Out of scope

- Non-CP BSSE modes (uncorrected, VMFC, etc.) for any writer.
- Any QC program other than Psi4/ORCA/Q-Chem.
- Parsing writer output files (already out of scope for this package).
