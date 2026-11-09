# <img align="center" src="https://github.com/j-yon/CrystaLattE/blob/master/media/logo/Logo.png" height=260>

Automated calculation of crystal lattice energies with the many-body expansion.

| Category | Badges |
|-------------|-------------|
| **Status** | [![GitHub Actions](https://img.shields.io/github/actions/workflow/status/j-yon/CrystaLattE/CI.yaml?logo=github)](https://github.com/j-yon/CrystaLattE/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/j-yon/CrystaLattE/graph/badge.svg)](https://codecov.io/gh/j-yon/CrystaLattE) ![python](https://img.shields.io/badge/python-3.13+-blue.svg) |
| **Foundation** | [![License](https://img.shields.io/github/license/j-yon/CrystaLattE.svg)](https://opensource.org/licenses/LGPL-3.0) [![GitHub Top Languages](https://img.shields.io/github/languages/top/j-yon/CrystaLattE)](https://github.com/j-yon/CrystaLattE/) |
| **GitHub Info** | [![GitHub Code Size](https://img.shields.io/github/languages/code-size/j-yon/CrystaLattE)](https://github.com/j-yon/CrystaLattE/) [![GitHub Commits per Month](https://img.shields.io/github/commit-activity/m/j-yon/CrystaLattE)](https://github.com/j-yon/CrystaLattE/) [![GitHub Last Commit](https://img.shields.io/github/last-commit/j-yon/CrystaLattE)](https://github.com/j-yon/CrystaLattE/) |
| **Citation** | [![doi](https://img.shields.io/badge/DOI-10.1063%2F1.5120520-blue)](http://dx.doi.org/10.1063/1.5120520) |

## Overview

CrystaLattE automates the computation of crystal lattice energies with the many-body expansion (MBE). From a CIF file it finds the molecules in the crystal, enumerates every dimer, trimer, … within a cutoff, keeps one representative of each symmetry-equivalent set, and assembles the lattice energy from the many-body energies of those representatives. The calculations are independent of each other, so they run in parallel, locally or on a [QCFractal](https://github.com/MolSSI/QCFractal) server.

Version 2 is a rewrite as a Python library. Compared with v1:

- **Exact symmetry deduplication.** Multimers are compared by their crystallographic symmetry operations instead of by numerical fingerprints (nuclear repulsion energy and Coulomb-matrix eigenvalues). Multimers are not built for comparison, and the result does not depend on floating-point tolerances.
- **Any multimer order.** `N` is not limited to 5.
- **Symmetry detection.** spglib finds the full space group, so symmetry missing from the declared one is still used.
- **Z' > 1 and cocrystals.** Crystals with several independent molecules in the asymmetric unit, including different chemical species, are supported. Lattice energies are reported per formula unit.
- **Disorder.** Partially occupied CIF sites can be enumerated into ordered states, or ignored or dropped.
- **Pluggable compute.** Calculations run through QCEngine/QCManyBody, QCFractal, or as input files you run yourself, and work with any program those support, not only Psi4.

## Installation

CrystaLattE requires Python 3.13 or newer. It is not on PyPI yet; install from source:

```bash
git clone https://github.com/j-yon/CrystaLattE.git
cd CrystaLattE
pip install -e .
```

Optional extras add the compute backends:

| Extra | Installs | Needed for |
|-------|----------|------------|
| `local` | `qcengine`, `qcmanybody` | `LocalBackend`; `InputFileBackend` for programs other than Psi4 |
| `fractal` | `qcportal` | `FractalBackend` |
| `tests` | `pytest` | running the test suite |

```bash
pip install -e ".[local,tests]"
```

The extras do not install quantum chemistry programs. Install those yourself; for example, Psi4 from conda-forge:

```bash
conda create -n cle -c conda-forge python=3.13 psi4
conda activate cle
pip install -e ".[local]"
```

## Quick start

### 1. Read a crystal

```python
import crystalatte as cle

crystal = next(cle.from_cif("benzene.cif"))
print(crystal)
```

`from_cif` is a generator because a CIF file may hold several data blocks; it yields one `Crystal` per block. The space group is read from the CIF's Hall or Hermann–Mauguin symbol and checked against any listed symmetry operations.

### 2. Generate the unique multimers

```python
dimers = cle.generate(crystal, N=2, R=10.0)
trimers = cle.generate(crystal, N=3, R=8.0)

for m in dimers[:5]:
    print(m.id, m.multiplicity, m.weight, m.geometric_mean, m.min_separation)
```

`generate(crystal, N, R)` returns one `Multimer` per symmetry-unique cluster of `N` molecules in which **every pair** of molecules is within `R` Å.

| Argument | Default | Description |
|----------|---------|-------------|
| `N` | (required) | Multimer order, an integer ≥ 2 |
| `R` | (required) | Cutoff in Å, applied to every pair of monomers |
| `detect_symmetry` | `True` | Detect the full space group with spglib before generating. If off, multimers related by symmetry the declared group lacks may appear twice |
| `symprec` | `1e-3` | spglib tolerance in Å |
| `z_fu` | auto | Formula units per cell. Defaults to the gcd over chemical species (by Hill formula) of molecules per cell. Override it for, e.g., a cocrystal of two species that share a formula |
| `use_origin` | `False` | Measure distance from the closest atom to the anchor's centroid, instead of the closest atom pair |
| `use_com` | `False` | Measure distance between centres of mass, instead of the closest atom pair |
| `verbose` | `1` | `0` hides progress bars and summaries |

Each `Multimer` provides:

| Attribute | Meaning |
|-----------|---------|
| `id` | Stable identifier, e.g. `dimer-1a2b…`, from the order and a geometry hash |
| `multiplicity` | Number of symmetry-equivalent copies represented |
| `weight` | Factor that multiplies this multimer's N-body energy in the per-formula-unit lattice energy |
| `geometric_mean`, `min_separation` | Distance measures in Å |
| `monomers` | The member `Monomer`s (symbols, fractional/Cartesian coordinates, symmetry operation) |
| `to_molecule()` | A QCElemental `Molecule` with one fragment per monomer |
| `to_monomer_molecules()` | One QCElemental `Molecule` per monomer |

### 3. Compute the lattice energy

Three backends run the calculations:

| Backend | Purpose | Extra |
|---------|---------|-------|
| `LocalBackend` | Run every calculation on this machine through QCEngine and QCManyBody | `local` |
| `FractalBackend` | Submit to a QCFractal server and collect from it | `fractal` |
| `InputFileBackend` | Only write input files (plus `manifest.json`) to run yourself; returns no results | none for Psi4; `local` for other programs |

```python
import crystalatte as cle

crystal = next(cle.from_cif("benzene.cif"))
multimers = cle.generate(crystal, 2, 10.0) + cle.generate(crystal, 3, 8.0)

spec = cle.ComputeSpec(
    program="psi4", method="mp2", basis="aug-cc-pvdz", bsse="cp",
    keywords={"scf_type": "df"},
)

backend = cle.LocalBackend("bz_run/", ncores=8)  # workdir: manifest + per-task results
# backend = cle.FractalBackend(client, "benzene", tag="bigmem", priority="high")
# backend = cle.InputFileBackend("inputs/")

backend.submit(multimers, spec)
print(backend.status())
results = backend.collect()    # raises IncompleteResultsError if anything is missing

print(results.lattice_energy)  # kJ/mol per formula unit
print(results.by_order)        # {2: ..., 3: ...}
results.table                  # per-multimer rows; results.to_dataframe() needs pandas
```

`ComputeSpec` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `program` | (required) | Any QCEngine program, e.g. `psi4`, `nwchem`, `molpro` |
| `method`, `basis` | (required) | Passed to the program |
| `bsse` | `"cp"` | `cp`, `nocp` or `vmfc` |
| `keywords` | `{}` | Program keywords; applied on top of the defaults |
| `use_defaults` | `True` | Apply tight, v1-style Psi4 keywords (convergence thresholds, density fitting, frozen core) under `keywords` |

`LocalBackend(workdir, *, qc_scratch=None, ncores=None, ncores_per_task=1, memory_per_task=None)`. Unset `ncores` and `memory_per_task` are taken from QCEngine's configuration.

#### Notes

- **Definition.** The lattice energy is per formula unit, in kJ/mol: `E_latt = (1/Z_fu) Σ_N Σ_orbits (copies per cell) ΔE^(N-body)`. It is comparable to a periodic-DFT `E(bulk)/Z − E(A) − E(B)`. The energy of deforming or relaxing the monomers from their gas-phase geometries is not included.
- **Incomplete data.** `collect()` refuses to assemble a lattice energy from missing or failed calculations. `collect(partial=True)` returns the rows that exist, so you can inspect per-multimer status and energies.
- **Resuming.** `LocalBackend` stores each finished sub-calculation in `workdir`. Rerunning with the same workdir skips completed work. Multimer ids depend on the geometry of the chosen representative, so changing the code or settings may cause work to be redone, but a result is never assigned to the wrong multimer.
- **Workdir vs QC scratch.** `workdir` holds CrystaLattE's job state. The QC program's own scratch files are separate: pass `qc_scratch="/fast/disk"`, or leave it unset and QCEngine decides (`$QCENGINE_SCRATCH_DIRECTORY`, a `qcengine.yaml` `scratch_directory`, the program's own variable such as `$PSI_SCRATCH`, then the system temp directory). For `FractalBackend`, scratch is set in the compute manager's configuration on the workers.
- **Scripts and parallelism.** `LocalBackend` runs calculations in worker processes. On macOS and Windows, put the code that calls it under `if __name__ == "__main__":`.
- **Input files.** `InputFileBackend` writes files and a `manifest.json`, then stops. Psi4 gets one file per multimer (`dimers/<id>.in`), since Psi4's own many-body driver does the counterpoise splitting. Every other program uses QCEngine's input builder, which writes single-point inputs only, so each multimer becomes one input per QCManyBody sub-calculation (`dimers/<id>/real-2_basis-1-2/…`). The manifest records the spec, generation settings and crystalatte version. For each multimer it records the order, multiplicity, lattice weight and files. For sub-calculations it also records the QCManyBody label, the real and ghost fragments, and the command QCEngine would run. To assemble a multimer, feed your energies to `qcmanybody.ManyBodyCore(...).analyze({label: {"energy": e}})` and take `<bsse>_corrected_<N>_body_contribution_to_energy`. Multiply by the weight and sum for the lattice energy (×2625.4996 for kJ/mol).
  - Counterpoise sub-calculations need ghost atoms, which QCEngine's builders are only verified to write for NWChem and Molpro (Q-Chem's drops them; GAMESS's refuses them). For other programs use `bsse="nocp"`.
  - Programs QCEngine cannot write inputs for (e.g. RDKit, ML potentials) raise an error pointing to `LocalBackend`. Some builders (CFOUR, MRChem, Turbomole) need the program installed where the files are generated.
  - `memory` takes GiB or a string like `"8 GB"`. If unset for QCEngine-built inputs, the memory detected on the machine writing the files is used.

## Disordered crystals

`from_cif(path, disorder=...)` controls how partially occupied sites are handled:

| Policy | Behaviour |
|--------|-----------|
| `"enumerate"` (default) | Yield one `Crystal` per ordered state, with a `RuntimeWarning`. Each crystal's `state` (`DisorderState`) has a `label`, the `weight` of that state implied by the CIF, and the disorder `groups` chosen. States come from explicit `_atom_site_disorder_assembly`/`_group` tags, or from symmetry when the tags are absent |
| `"error"` | Raise on any partial occupancy |
| `"ignore"` | Keep every site as if fully occupied |
| `"drop"` | Remove partially occupied sites |

Generate and compute each state separately, then combine them with their `crystal.state.weight` as appropriate.

## Example scripts

- `scripts/run_generate.py`: generate multimers from a CIF and print counts, multiplicities and timing.

  ```bash
  python scripts/run_generate.py tests/x23/benzene/benzene.cif -N 3 -R 8 --list
  ```

- `scripts/run_lattice_energy.py`: an end-to-end lattice energy with `FractalBackend`.

## Testing

```bash
pip install -e ".[tests]"
pytest
```

Tests in `tests/compute/` that need `qcengine`, `qcmanybody`, `qcportal`, `qcfractal` or `psi4` are skipped when those packages are missing.

## Migrating from v1

v1's `.cle` input file and `crystalatte.py` command-line script have been removed. Their keywords map onto the API as follows:

| v1 keyword | v2 equivalent |
|------------|---------------|
| `cif_input` | `cle.from_cif(path)` |
| `nmers_up_to`, `r_cut_dimer` … `r_cut_pentamer` | one `cle.generate(crystal, N, R)` call per order |
| `r_cut_com` | `generate(..., use_com=True)` |
| `uniq_filter` (ChSEV, Dreamaligner) | removed; deduplication is by exact symmetry. `cle.count_chemistry_classes(multimers)` reproduces the v1 NRE/ChSEV grouping for comparison |
| `cif_a`, `cif_b`, `cif_c`, `r_cut_monomer`, `cif_output` | removed; the search region is sized automatically |
| `bfs_thresh` | fixed at 1.2 × covalent radii for molecule detection |
| `psi4_method` | `ComputeSpec(program="psi4", method=..., basis=...)` |
| `psi4_bsse` | `ComputeSpec(bsse=...)` |
| `psi4_memory` | `LocalBackend(memory_per_task=...)` / `InputFileBackend(memory=...)` |
| `cle_run_type` | choice of backend |

## Citation

If you use CrystaLattE, please cite:

> C. H. Borca, B. W. Bakr, L. A. Burns, C. D. Sherrill, "CrystaLattE: Automated computation of lattice energies of organic crystals exploiting the many-body expansion to achieve dual-level parallelism," *J. Chem. Phys.* **151**, 144103 (2019). [doi:10.1063/1.5120520](http://dx.doi.org/10.1063/1.5120520)

This will be updated for CrystaLattE v2 once it is published.

## Copyright

Copyright (c) 2026, Jaden C. Yon
