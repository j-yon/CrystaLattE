# Compute Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run CrystaLattE multimers through QCEngine (local) or QCFractal (managed) behind one `Backend` interface, with input-file writing as a fallback, and assemble a per-formula-unit lattice energy from the results.

**Architecture:** `generate` attaches stable IDs, per-formula-unit weights and generation settings to every `Multimer`. A `ComputeSpec` describes the model chemistry once; `LocalBackend` (QCManyBody `ManyBodyCore` + `qcengine.compute`), `FractalBackend` (one `ManybodyDataset` per N in the user's `PortalClient`) and `InputFileBackend` (existing writers) all implement `submit`/`status`/`collect`. One `assemble()` turns per-multimer nonadditive energies into a `Results` object.

**Tech Stack:** Python ≥3.13, numpy, qcelemental ≥0.50,<0.70, qcmanybody ≥0.8.0, qcengine ≥0.52.0, qcportal ≥0.71, pytest.

**Spec:** `docs/superpowers/specs/2026-11-02-compute-backends-design.md`

## Global Constraints

- Core dependency: `qcelemental>=0.50,<0.70` (forced by qcengine 0.52 / qcmanybody 0.8 / qcportal 0.71).
- Optional extras: `crystalatte[local]` = `qcengine>=0.52.0`, `qcmanybody>=0.8.0`; `crystalatte[fractal]` = `qcportal>=0.71`. Backends import them lazily and raise `ImportError` containing `pip install "crystalatte[local]"` / `pip install "crystalatte[fractal]"`.
- No `.cle` files, no CLI; everything is the Python API exported from `crystalatte/__init__.py`.
- `ManyBodyCore` levels must be contiguous: `{1: mc, ..., N: mc}`.
- The per-multimer quantity is QCManyBody's `"{bsse}_corrected_{N}_body_contribution_to_energy"` with `max_nbody == N == nfragments`.
- Psi4 defaults, merged under user keywords (user wins): `{"e_convergence": 10, "d_convergence": 10, "scf_type": "df", "mp2_type": "df", "cc_type": "df", "freeze_core": "true"}` (copied from v1 `master:crystalatte/crystalatte.py:2134`).
- Lattice energy is per formula unit; `Results.lattice_energy` and `Results.by_order` are in kJ/mol (v1 convention); per-multimer energies are stored in hartree.
- Multimer ID format: `f"{order_name(N)}-{Molecule.get_hash()[:16]}"`, e.g. `dimer-3f9a...`.
- Commit messages: `<type>(<scope>): <subject>`, imperative, ≤50 chars.

## Review Focus

1. **Collect from a fresh backend object** (new script, same scratch dir / same `name`) must give the same `Results` as collecting on the object that submitted. Tested in Tasks 7 and 8.
2. **Submitting multimers from a different crystal (or disorder state) under an existing backend name/scratch dir** must raise `ValueError` naming the fingerprint mismatch, never mix data. Tested in Tasks 7 and 8.
3. **Re-running with a larger R** must reuse earlier results and `collect` must sum only the latest active set; re-running with a smaller R must not sum the leftover entries. Tested in Task 7 (and the shared helper in Task 5).
4. **A different `ComputeSpec` submitted into an existing scratch dir / dataset name** must raise `ValueError`, not silently mix model chemistries. Tested in Tasks 7 and 8.
5. **Multimers without generation info** (constructed by hand, e.g. in tests or user code) passed to any `submit` must raise `ValueError` explaining they must come from `cle.generate`. Tested in Task 6.

---

### Task 0: Raise the qcelemental floor

**Files:**
- Modify: `pyproject.toml:11`
- Modify: `devtools/conda-envs/test_env.yaml`

- [ ] **Step 1:** Set `"qcelemental>=0.50,<0.70"` in `pyproject.toml` dependencies and add `qcelemental >=0.50,<0.70` to `test_env.yaml`.
- [ ] **Step 2:** Upgrade the dev env: `pip install "qcelemental>=0.50,<0.70"`; confirm `python -c "import qcelemental; print(qcelemental.__version__)"` prints ≥0.50.
- [ ] **Step 3:** Run `pytest -m "not slow" -q`. Expected: same pass/xfail counts as on `v2` before the bump. If `Monomer.to_molecule()` / `Multimer.to_molecule()` break (pydantic v2 / `qcelemental.models` v1–v2 split), fix them to construct `qcelemental.models.Molecule` (the QCSchema v1 model, which is what qcengine 0.52 `compute` and `ManyBodyCore` accept) and keep their docstrings.
- [ ] **Step 4:** Run full `pytest -q`; expected green bar same as before.
- [ ] **Step 5:** Commit `chore(deps): require qcelemental>=0.50`.

---

### Task 1: Per-anchor multiplicity in dedup

Per-formula-unit weights need, for each unique multimer, how many of its merged candidates were anchored on each reference molecule (`multiplicity` alone loses this when references sit on sites of different symmetry).

**Files:**
- Modify: `src/crystalatte/core/sym_ops.py` (`SymOpList`)
- Modify: `src/crystalatte/work/dedup/equivalence.py:15-23` (`merge`)
- Test: `tests/test_dedup.py`

**Interfaces:**
- Produces: `SymOpList.anchor_counts -> dict[int, int]` (reference index → count). Defaults to `{int(self.ref_indices[0]): self.multiplicity}` when never set; has a setter. Invariant: `sum(anchor_counts.values()) == multiplicity`.

- [ ] **Step 1: Write failing tests**

```python
def test_anchor_counts_default_is_own_anchor():
    g = SymOpList.from_components(rot, tr, 3, np.array([1, 0]))  # build like existing tests do
    assert g.anchor_counts == {1: 3}

def test_merge_sums_anchor_counts():
    # a: anchored on ref 0 (mult 1), b: on ref 1 (mult 2)
    rep, _ = merge([a, b])
    assert rep.anchor_counts == {0: 1, 1: 2}
    assert sum(rep.anchor_counts.values()) == rep.multiplicity == 3

@pytest.mark.parametrize("cif", [<pyrazole>, <salicylic_hydroxybenzamide>, <benzene>])
def test_generate_anchor_counts_sum_to_multiplicity(cif):
    for m in cle.generate(crystal, N=2, R=6.0, verbose=0):
        assert sum(m.sym_ops.anchor_counts.values()) == m.multiplicity
```

- [ ] **Step 2:** Run `pytest tests/test_dedup.py -k anchor_counts -v`; expected FAIL (`AttributeError: anchor_counts`).
- [ ] **Step 3:** Implement the property/setter on `SymOpList` (stored as `_anchor_counts: dict[int, int] | None = None`, excluded from `__eq__`), and in `merge` set `rep.anchor_counts` to the key-wise sum over the group before setting multiplicity. Grep `multiplicity =` / `multiplicity +=` under `src/` and confirm `merge` is the only merge site; if another exists, sum anchor counts there too.
- [ ] **Step 4:** Run `pytest tests/test_dedup.py tests/test_generate.py -q`; expected PASS.
- [ ] **Step 5:** Commit `feat(dedup): track per-anchor multiplicity`.

---

### Task 2: Generation info, stable IDs and weights on `Multimer`

**Files:**
- Create: `src/crystalatte/core/generation.py`
- Modify: `src/crystalatte/core/multimer.py` (`Multimer`, add `order_name`)
- Modify: `src/crystalatte/work/generate.py` (`generate`, `_process_multimer`)
- Test: `tests/test_generation_info.py`

**Interfaces:**
- Produces:
  - `order_name(N: int) -> str` in `core/multimer.py` (moved verbatim from `work/setup.py::_order_name`, incl. `_NAMES`).
  - `@dataclass(frozen=True) GenerationInfo` in `core/generation.py` with fields `N: int`, `R: float`, `symprec: float`, `detect_symmetry: bool`, `options: tuple[tuple[str, object], ...]` (sorted distance-mode kwargs: `use_origin`, `use_com`), `crystal_fingerprint: str`, `site_counts: tuple[int, ...]` (n_r per reference), `species: tuple[str, ...]` (Hill formula per reference), `z_fu: int`; methods `to_dict() -> dict` and `from_dict(d) -> GenerationInfo` (JSON-safe round trip).
  - `crystal_fingerprint(crystal: Crystal) -> str`: sha256 hex of canonical JSON of lattice parameters (rounded to 1e-6), space-group operations (sorted string forms), ASU symbols and fractional coordinates (rounded to 1e-6), and disorder state label (or `None`).
  - `Multimer.generation -> GenerationInfo | None` (new optional init field `_generation`, default `None`).
  - `Multimer.id -> str`: `f"{order_name(len(monomers))}-{self.to_molecule().get_hash()[:16]}"`, cached.
  - `Multimer.weight -> float`: `sum(site_counts[r] * a for r, a in sym_ops.anchor_counts.items()) / (N * z_fu)`; raises `ValueError` if `generation is None`.

Weight derivation (put in the `weight` docstring): energy per cell is `(1/N) Σ_r n_r Σ_{C ∋ r} ΔE_C`; candidates anchored on reference r enumerate exactly the clusters containing r, so `a_r` counts them per orbit; dividing by `Z_fu` (gcd over species of `Σ_{r∈species} n_r`) gives per formula unit. For Z′=1 single-component crystals this is `multiplicity / N` (v1's replicas/N).

- [ ] **Step 1: Write failing tests**

```python
def test_ids_stable_across_runs(benzene):
    a = [m.id for m in cle.generate(benzene, N=2, R=8.0, verbose=0)]
    b = [m.id for m in cle.generate(benzene, N=2, R=8.0, verbose=0)]
    assert sorted(a) == sorted(b) and len(set(a)) == len(a)
    assert all(i.startswith("dimer-") and len(i) == len("dimer-") + 16 for i in a)

def test_weight_is_multiplicity_over_N_for_single_reference(benzene):
    for m in cle.generate(benzene, N=3, R=6.0, verbose=0):
        assert m.weight == pytest.approx(m.multiplicity / 3)

def test_generation_info_round_trips(benzene):
    g = cle.generate(benzene, N=2, R=8.0, verbose=0)[0].generation
    assert GenerationInfo.from_dict(json.loads(json.dumps(g.to_dict()))) == g
    assert (g.N, g.R, g.z_fu) == (2, 8.0, 4)

def test_cocrystal_z_fu_and_species(salicylic_hydroxybenzamide):
    g = cle.generate(salicylic_hydroxybenzamide, N=2, R=6.0, verbose=0)[0].generation
    assert len(set(g.species)) == 2 and g.z_fu == <per-cell count of either species>

def test_fingerprint_differs_between_disorder_states(<two states of a disordered fixture from test_disorder.py>):
    assert crystal_fingerprint(s0) != crystal_fingerprint(s1)

def test_weight_without_generation_raises():
    with pytest.raises(ValueError, match="generate"):
        Multimer(mons, ops, 1).weight
```

Fill the `<...>` values by inspecting the fixtures (count molecules per cell from `expand_cell`); hard-code the numbers in the test.

- [ ] **Step 2:** Run `pytest tests/test_generation_info.py -v`; expected FAIL (import errors).
- [ ] **Step 3:** Implement. In `generate`, after `get_references()`: compute `site_counts` as `len(crystal.space_group.get_unique_sym_ops(m.centroid_frac, atol=kwargs["site_atol"]))` for each reference on the (detected) crystal, `species` with `qcel.molparse.order_molecular_formula` (or equivalent Hill formula) on each reference's symbols, `z_fu = gcd` of per-species sums, fingerprint of the *input* crystal (before detection), then pass one `GenerationInfo` into every `_process_multimer`. Update `work/setup.py` to import `order_name` from `core.multimer` (it is deleted in Task 6).
- [ ] **Step 4:** Run `pytest -m "not slow" -q`; expected all PASS.
- [ ] **Step 5:** Commit `feat(multimer): add stable ids and lattice weights`.

---

### Task 3: Normalization correctness test (toy potential vs P1 brute force)

Required safeguard for the per-formula-unit choice; no QC program involved.

**Files:**
- Test: `tests/test_lattice_weights.py`

**Interfaces:**
- Consumes: `Multimer.weight`, `GenerationInfo.z_fu`, `expand_cell`, `SpaceGroup.from_sym_ops`, `ASU`, `Crystal` (P1 construction as in `tests/test_disorder.py:22,47`).

- [ ] **Step 1: Write the tests**

```python
def toy_energy(m: Multimer) -> float:
    """Permutation- and rigid-motion-invariant stand-in for a nonadditive energy"""
    # sum over inter-monomer atom pairs of 1/d**6 (Cartesian, Å)

def symmetric_total(crystal, N, R):
    return sum(m.weight * toy_energy(m) for m in cle.generate(crystal, N=N, R=R, verbose=0))

def p1_total(crystal, N, R):
    # expand the detected crystal's ASU to the full cell, build a P1 Crystal,
    # generate with detect_symmetry=False (suppress its RuntimeWarning),
    # sum multiplicity/N * toy_energy, divide by the symmetric run's z_fu
    ...

@pytest.mark.slow
@pytest.mark.parametrize("fixture, N, R", [
    ("benzene", 2, 6.0), ("benzene", 3, 4.5),          # Z'=1/2, molecule on inversion centre
    ("pyrazole", 2, 6.0),                               # Z'>1
    ("salicylic_hydroxybenzamide", 2, 6.0),             # cocrystal
    (<fixture whose references have different site_counts, if any exists>, 2, 6.0),
])
def test_per_formula_unit_matches_p1_brute_force(fixture, N, R):
    c = load(fixture)
    assert symmetric_total(c, N, R) == pytest.approx(p1_total(c, N, R), rel=1e-9)
```

For the mixed-site-symmetry row: scan `tests/x23` and `tests/other` for a crystal whose `GenerationInfo.site_counts` are not all equal; if none exists, drop that row and add a unit test of `Multimer.weight` with a hand-built `GenerationInfo(site_counts=(2, 4), species=("A", "B"), z_fu=2, ...)` and `anchor_counts={0: 1, 1: 3}`, N=2, asserting `weight == (2*1 + 4*3) / (2*2)`.

- [ ] **Step 2:** Run `pytest tests/test_lattice_weights.py -v`. Expected: PASS. A failure here is a real normalization bug in Task 1/2 — fix the weight there, do not loosen the tolerance. Tune R down only if P1 generation exceeds ~30 s.
- [ ] **Step 3:** Commit `test(weights): check per-formula-unit vs P1 sum`.

---

### Task 4: `ComputeSpec`

**Files:**
- Create: `src/crystalatte/compute/__init__.py`, `src/crystalatte/compute/spec.py`
- Modify: `src/crystalatte/__init__.py` (export `ComputeSpec`)
- Test: `tests/compute/test_spec.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) ComputeSpec(program: str, method: str, basis: str, keywords: Mapping = {}, bsse: str = "cp", use_defaults: bool = True)` with
  - `final_keywords -> dict` (defaults for `program.lower()` from `DEFAULT_KEYWORDS: dict[str, dict]`, overlaid by `keywords`; `{}` defaults plus a one-time `UserWarning` recommending tight convergence for programs absent from `DEFAULT_KEYWORDS`),
  - `atomic_template() -> dict` (`{"driver": "energy", "model": {"method", "basis"}, "keywords": final_keywords}`),
  - `manybody_kwargs(N: int) -> dict` (`molecule` excluded; `bsse_type=[bsse]`, `levels={k: "crystalatte" for k in 1..N}`, `return_total_data=False`, `supersystem_ie_only=False`, `embedding_charges={}`),
  - `to_portal(N: int)` → `qcportal.manybody.ManybodySpecification(program="qcmanybody", levels={k: QCSpecification(program, driver="energy", method, basis, keywords=final_keywords) for k in 1..N}, bsse_correction=[bsse])` (lazy import),
  - `to_dict()` / `from_dict()` and `fingerprint() -> str` (sha256 of canonical `to_dict()` JSON).
  - `bsse` validated against `{"cp", "nocp", "vmfc"}`; program/method/basis lowercased.

- [ ] **Step 1: Write failing tests**

```python
def test_psi4_defaults_merged_user_wins():
    s = ComputeSpec("psi4", "mp2", "aug-cc-pvdz", keywords={"scf_type": "pk"})
    assert s.final_keywords == {"e_convergence": 10, "d_convergence": 10, "scf_type": "pk",
                                "mp2_type": "df", "cc_type": "df", "freeze_core": "true"}

def test_use_defaults_false():
    assert ComputeSpec("psi4", "hf", "sto-3g", use_defaults=False).final_keywords == {}

def test_unknown_program_warns():
    with pytest.warns(UserWarning, match="convergence"):
        ComputeSpec("nwchem", "hf", "sto-3g").final_keywords

def test_levels_contiguous():
    assert ComputeSpec("psi4", "hf", "sto-3g").manybody_kwargs(3)["levels"] == {1: "crystalatte", 2: "crystalatte", 3: "crystalatte"}

def test_bad_bsse_rejected():
    with pytest.raises(ValueError): ComputeSpec("psi4", "hf", "sto-3g", bsse="ssfc")

def test_round_trip_and_fingerprint():
    s = ComputeSpec("psi4", "mp2", "cc-pvdz")
    assert ComputeSpec.from_dict(s.to_dict()) == s and s.fingerprint() == ComputeSpec.from_dict(s.to_dict()).fingerprint()

def test_to_portal(qcportal = pytest.importorskip("qcportal")):
    p = ComputeSpec("psi4", "hf", "sto-3g").to_portal(2)
    assert sorted(p.levels) == [1, 2] and p.bsse_correction == ["cp"]
```

- [ ] **Step 2:** Run `pytest tests/compute/test_spec.py -v`; expected FAIL.
- [ ] **Step 3:** Implement as specified (store `keywords` as a sorted tuple of items internally so the dataclass stays hashable/frozen).
- [ ] **Step 4:** Run; expected PASS (portal test skipped if qcportal absent).
- [ ] **Step 5:** Commit `feat(compute): add ComputeSpec`.

---

### Task 5: `Results`, `Status`, errors and assembly

**Files:**
- Create: `src/crystalatte/compute/results.py`
- Modify: `src/crystalatte/compute/__init__.py`, `src/crystalatte/__init__.py` (export `Results`, `IncompleteResultsError`)
- Test: `tests/compute/test_results.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Row(id: str, N: int, multiplicity: int, weight: float, energy: float | None, status: Literal["complete", "pending", "failed"], error: str | None = None)` (`energy` in hartree).
  - `IncompleteResultsError(Exception)` with `.missing: dict[int, dict[str, list[str]]]` (N → `{"failed": [...], "pending": [...]}`) and a message listing counts per N, failed IDs with the first 200 chars of each error, pending IDs (truncate each list to 20 with "... and k more").
  - `Results(rows: list[Row])` with `.rows`, `.missing`, `.complete: bool`, `.lattice_energy -> float` (kJ/mol, `Σ weight·energy · qcel.constants.hartree2kJmol`), `.by_order -> dict[int, float]` (kJ/mol), `.table -> list[dict]`, `.to_dataframe()` (lazy pandas). `lattice_energy`/`by_order` raise `IncompleteResultsError` unless `complete`.
  - `assemble(rows: list[Row], *, partial: bool) -> Results`: raises `IncompleteResultsError` if any row is not complete and `not partial`.
  - `@dataclass Status(counts: dict[int, dict[str, int]])` with readable `__str__` (one line per N: `dimers: 40 complete, 2 pending, 1 failed`).
  - `select_active(active: dict[int, list[str]], known: dict[str, Row]) -> list[Row]`: rows for the active IDs only; an active ID with no stored row becomes `pending`. Shared by Local and Fractal `collect`.

- [ ] **Step 1: Write failing tests**

```python
def test_lattice_energy_sums_weighted_kjmol():
    r = assemble([Row("dimer-a", 2, 2, 1.0, -0.001, "complete"),
                  Row("trimer-b", 3, 3, 1.0, 0.0001, "complete")], partial=False)
    assert r.lattice_energy == pytest.approx((-0.001 + 0.0001) * qcel.constants.hartree2kJmol)
    assert r.by_order[2] == pytest.approx(-0.001 * qcel.constants.hartree2kJmol)

def test_collect_refuses_incomplete():
    rows = [Row("dimer-a", 2, 1, 0.5, -0.001, "complete"),
            Row("dimer-b", 2, 1, 0.5, None, "failed", "SCF did not converge"),
            Row("dimer-c", 2, 1, 0.5, None, "pending")]
    with pytest.raises(IncompleteResultsError) as e:
        assemble(rows, partial=False)
    assert e.value.missing == {2: {"failed": ["dimer-b"], "pending": ["dimer-c"]}}
    assert "SCF did not converge" in str(e.value)

def test_partial_exposes_rows_not_total():
    r = assemble(rows, partial=True)
    assert len(r.table) == 3 and not r.complete
    with pytest.raises(IncompleteResultsError): r.lattice_energy
    with pytest.raises(IncompleteResultsError): r.by_order

def test_select_active_ignores_stale_and_marks_unknown_pending():
    known = {"dimer-a": Row(...complete...), "dimer-old": Row(...complete...)}
    rows = select_active({2: ["dimer-a", "dimer-new"]}, known)
    assert [(x.id, x.status) for x in rows] == [("dimer-a", "complete"), ("dimer-new", "pending")]
```

- [ ] **Step 2:** Run `pytest tests/compute/test_results.py -v`; expected FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Commit `feat(compute): add Results and lattice assembly`.

---

### Task 6: `Backend` ABC and `InputFileBackend`; remove `write_inputs`

**Files:**
- Create: `src/crystalatte/compute/backends/__init__.py`, `.../backends/base.py`, `.../backends/inputs.py`
- Modify: `src/crystalatte/io/writers/base.py`, `src/crystalatte/io/writers/psi4.py` (take `ComputeSpec`)
- Delete: `src/crystalatte/work/setup.py`
- Modify: `src/crystalatte/__init__.py` (drop `write_inputs`; export `Backend`, `InputFileBackend`)
- Modify/rename: `tests/test_writers.py` → keep writer tests, port `write_inputs` tests to `InputFileBackend`
- Test: `tests/compute/test_backend_base.py`

**Interfaces:**
- Produces:
  - `class Backend(ABC)`: abstract `submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None`, `status(self) -> Status`, `collect(self, *, partial: bool = False) -> Results`.
  - `Backend._validate(multimers) -> dict[int, list[Multimer]]` (concrete helper on the ABC): raises `ValueError` if the list is empty, if any multimer has `generation is None` (message says multimers must come from `crystalatte.generate`), if multimers of one N carry different `GenerationInfo`, or if different N carry different `crystal_fingerprint`; returns multimers grouped by N.
  - `QCWriter.write(self, path: Path, monomers: list[Monomer], spec: ComputeSpec) -> None` and `QCWriter.extension() -> str` (Psi4: `"in"`, keeps current file names' suffix). Psi4 writer reads `method`, `basis`, `bsse` and `final_keywords` from the spec; `charge`/`multiplicity` stay at 0/1 (unchanged behaviour, out of scope); `memory` moves to an `InputFileBackend(memory=...)` argument passed through.
  - `InputFileBackend(output_dir: str | Path, *, memory: str | None = None)`: `submit` writes `output_dir/{order_name(N)}s/{multimer.id}.{ext}` via `get_writer(spec.program)`; `status()` and `collect()` raise `NotImplementedError("InputFileBackend only writes input files; run them yourself or use LocalBackend/FractalBackend")`.

- [ ] **Step 1: Write failing tests**

```python
def test_submit_rejects_multimers_without_generation(tmp_path, hand_built_dimer):
    with pytest.raises(ValueError, match="crystalatte.generate"):
        InputFileBackend(tmp_path).submit([hand_built_dimer], ComputeSpec("psi4", "hf", "sto-3g"))

def test_submit_rejects_mixed_crystals(tmp_path, benzene, ammonia):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)[:1] + cle.generate(ammonia, N=2, R=5.0, verbose=0)[:1]
    with pytest.raises(ValueError, match="fingerprint"):
        InputFileBackend(tmp_path).submit(ms, ComputeSpec("psi4", "hf", "sto-3g"))

def test_input_backend_writes_by_id(tmp_path, benzene):
    ms = cle.generate(benzene, N=2, R=6.0, verbose=0)
    InputFileBackend(tmp_path).submit(ms, ComputeSpec("psi4", "hf", "sto-3g"))
    assert sorted(p.stem for p in (tmp_path / "dimers").glob("*.in")) == sorted(m.id for m in ms)

def test_input_backend_status_collect_not_implemented(tmp_path):
    b = InputFileBackend(tmp_path)
    with pytest.raises(NotImplementedError): b.status()
    with pytest.raises(NotImplementedError): b.collect()

def test_psi4_input_contains_tight_defaults(tmp_path, benzene): ...
    # file text contains "e_convergence 10" and "freeze_core true"

def test_write_inputs_removed():
    assert not hasattr(cle, "write_inputs")
```

Hand-built multimers for writer unit tests: attach a `GenerationInfo` built in a fixture in `tests/conftest.py` so the remaining writer tests stay as they are.

- [ ] **Step 2:** Run `pytest tests/compute/test_backend_base.py tests/test_writers.py -v`; expected FAIL.
- [ ] **Step 3:** Implement; delete `work/setup.py`; update `scripts/` if any reference `write_inputs` (grep).
- [ ] **Step 4:** Run `pytest -m "not slow" -q`; expected PASS.
- [ ] **Step 5:** Commit `feat(compute): add Backend ABC and InputFileBackend` (body notes `write_inputs` removal as breaking).

---

### Task 7: `LocalBackend`

**Files:**
- Create: `src/crystalatte/compute/backends/local.py`
- Modify: `src/crystalatte/compute/backends/__init__.py`, `src/crystalatte/__init__.py` (export `LocalBackend`), `pyproject.toml` (`[project.optional-dependencies] local = [...]`), `tests/conftest.py` (register `psi4` marker)
- Test: `tests/compute/test_local.py`

**Interfaces:**
- Consumes: `Backend`, `Backend._validate`, `ComputeSpec.manybody_kwargs/atomic_template/fingerprint`, `Row`, `Status`, `assemble`, `select_active`, `Multimer.id/weight/generation`.
- Produces: `LocalBackend(scratch: str | Path, *, ncores: int | None = None, ncores_per_task: int = 1, memory_per_task: float | None = None)` (memory in GiB; defaults from `qcengine.get_config()`: `ncores` = detected cores, `memory_per_task` = detected memory / concurrency). Concurrency = `max(1, ncores // ncores_per_task)`; concurrency 1 runs serially in-process (tests rely on this), otherwise `concurrent.futures.ProcessPoolExecutor`.
  - Module-level `_run_task(atomic_input: dict, program: str, task_config: dict) -> dict` wraps `qcengine.compute(..., raise_error=False)` and returns `{"success": bool, "energy": float | None, "error": str | None}` — the single seam tests monkeypatch.

Scratch layout (all JSON):
- `scratch/manifest.json`: `{"spec": spec.to_dict(), "spec_fingerprint", "crystal_fingerprint", "generation": {N: GenerationInfo.to_dict()}, "active": {N: [ids]}, "multimers": {id: {"N", "multiplicity", "weight", "molecule": to_molecule().dict()}}}`. Written atomically (write tmp + `os.replace`).
- `scratch/tasks/<multimer id>/<sha256(label)[:16]>.json`: `{"label", "success", "energy", "error"}`, written as each task finishes.

Behaviour:
- `submit`: `_validate`; if a manifest exists and its `spec_fingerprint` or `crystal_fingerprint` differs → `ValueError` naming which. Merge multimers into the manifest, set `active[N]` to the submitted IDs (replacing that N's previous active list), save manifest *before* running. For each multimer build `ManyBodyCore(molecule, **spec.manybody_kwargs(N))`, iterate `iterate_molecules()`, skip labels whose task file has `success: true`, run the rest (failed task files are retried on resubmit), then when every label of a multimer succeeded call `core.analyze({label: {"energy": e}})` and store `results["results"][f"{bsse}_corrected_{N}_body_contribution_to_energy"]` in `scratch/tasks/<id>/result.json`. Blocks until done; never raises for task failures.
- `status()` / `collect(partial=False)` read only the scratch dir (work on a fresh object): a multimer is complete if `result.json` exists, failed if any task file has `success: false` (error = first failing task's error), else pending; rows come from `select_active(manifest["active"], ...)`.

- [ ] **Step 1: Write failing tests** (fake `_run_task` returning `energy = toy function of the molecule's geometry`, `ncores=1`)

```python
def test_submit_collect_and_fresh_object(tmp_path, fake_engine, benzene):
    ms = cle.generate(benzene, N=2, R=6.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1); b.submit(ms, SPEC)
    r1 = b.collect(); r2 = LocalBackend(tmp_path, ncores=1).collect()
    assert r1.lattice_energy == r2.lattice_energy and r1.complete

def test_resume_skips_finished_tasks(tmp_path, fake_engine, benzene):
    # first submit with the fake failing on one specific label; second submit
    # runs only that label (count calls) and collect then succeeds

def test_failure_reported_not_raised(tmp_path, failing_engine, benzene):
    b = LocalBackend(tmp_path, ncores=1); b.submit(ms, SPEC)        # no exception
    assert "failed" in str(b.status())
    with pytest.raises(IncompleteResultsError, match="boom"): b.collect()
    assert b.collect(partial=True).missing[2]["failed"]

def test_spec_mismatch_raises(tmp_path, fake_engine, benzene): ...      # second submit with other basis -> ValueError "spec"
def test_crystal_mismatch_raises(tmp_path, fake_engine, benzene, ammonia): ...  # -> ValueError "fingerprint"

def test_larger_R_reuses_and_smaller_R_excludes(tmp_path, fake_engine, benzene):
    small = cle.generate(benzene, N=2, R=5.0, verbose=0); big = cle.generate(benzene, N=2, R=7.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(small, SPEC); calls_small = fake_engine.calls
    b.submit(big, SPEC)    # only new multimers' tasks run
    new_ids = {m.id for m in big} - {m.id for m in small}
    assert fake_engine.calls - calls_small == n_tasks(m for m in big if m.id in new_ids)
    b.submit(small, SPEC)  # no new tasks
    assert {row.id for row in b.collect().rows} == {m.id for m in small}

def test_missing_extra_gives_install_hint(monkeypatch): ...  # hide qcengine -> ImportError 'crystalatte[local]'

@pytest.mark.psi4
def test_cp_dimer_matches_direct_psi4(tmp_path, ammonia):
    # one ammonia dimer, HF/STO-3G, use_defaults=False; compare LocalBackend energy to
    # E(AB) - E(A in AB basis) - E(B in AB basis) computed with qcengine directly
    # (ghost atoms via Molecule.get_fragment(real, ghost)); abs tol 1e-8 Eh
```

`pytest.importorskip("qcmanybody")` at module top; the `psi4` test also `importorskip("psi4")`.

- [ ] **Step 2:** Run `pytest tests/compute/test_local.py -v`; expected FAIL.
- [ ] **Step 3:** Implement; add `local = ["qcengine>=0.52.0", "qcmanybody>=0.8.0"]` extra; register marker `psi4: needs a working Psi4 install` in `tests/conftest.py::pytest_configure`.
- [ ] **Step 4:** Run `pytest tests/compute -v`; expected PASS (psi4 test runs if Psi4 is installed in the env; report whether it ran).
- [ ] **Step 5:** Commit `feat(compute): add LocalBackend`.

---

### Task 8: `FractalBackend`

**Files:**
- Create: `src/crystalatte/compute/backends/fractal.py`
- Modify: `src/crystalatte/compute/backends/__init__.py`, `src/crystalatte/__init__.py` (export `FractalBackend`), `pyproject.toml` (`fractal = ["qcportal>=0.71"]`), `tests/conftest.py` (register `fractal` marker)
- Test: `tests/compute/test_fractal.py`

**Interfaces:**
- Consumes: as Task 7, plus `ComputeSpec.to_portal`.
- Produces: `FractalBackend(client, name: str, *, tag: str | None = None, priority: str | None = None)` and extra method `resubmit_errors() -> int` (number of records reset via `dataset.reset_records(...)` on errored records).

Behaviour:
- Dataset per N: `f"{name}-{order_name(N)}s"`, created with `client.add_dataset("manybody", ...)` or fetched with `client.get_dataset("manybody", ...)`.
- One specification per dataset, named `"crystalatte"`. Decision for the spec's open point: a different `ComputeSpec` under an existing name raises `ValueError` (compare `spec_fingerprint` in extras); users pick a new `name` for a new model chemistry.
- Dataset extras: `{"crystalatte": {"spec": ..., "spec_fingerprint", "crystal_fingerprint", "generation": GenerationInfo.to_dict(), "active": [ids]}}` via `dataset.set_extras` (merge, don't clobber other keys). Crystal fingerprint mismatch → `ValueError`.
- Entries: `name=multimer.id`, `initial_molecule=multimer.to_molecule()`, `attributes={"N", "multiplicity", "weight", "crystal_fingerprint"}`; only names not in `dataset.entry_names` are added.
- `submit`: then `dataset.submit(specification_names="crystalatte", tag=tag, priority=priority)`; returns immediately.
- `status()` / `collect()`: for each dataset `f"{name}-*s"` that exists, read extras `active`, `iterate_records(specification_names="crystalatte")`; status `complete` → energy from `record.properties[f"{bsse}_corrected_{N}_body_contribution_to_energy"]` (verify the exact key on the first Snowflake run and pin it in a constant), `error` → `failed` with the record's error message, anything else → `pending`; weights/multiplicities from entry attributes; rows via `select_active`.

- [ ] **Step 1: Write failing tests** (module-level `pytest.importorskip("qcfractal")`; marker `fractal`; one module-scoped `FractalSnowflake` fixture with Psi4 workers)

```python
@pytest.mark.fractal
def test_submit_creates_one_dataset_per_order(snowflake_client, ammonia): ...
    # N=2 and N=3 lists -> datasets "nh3-dimers", "nh3-trimers"; entry names == multimer ids

@pytest.mark.fractal
def test_resubmit_is_idempotent(snowflake_client, ammonia): ...   # entry count unchanged

@pytest.mark.fractal
def test_collect_matches_local(snowflake_client, tmp_path, ammonia):
    # HF/STO-3G, use_defaults=False, N=2 small R; wait for completion (poll status, timeout 600 s);
    # FractalBackend(client, "nh3").collect().lattice_energy == LocalBackend(...).collect().lattice_energy (abs 1e-8 kJ/mol)
    # and a fresh FractalBackend(client, "nh3").collect() gives the same value

@pytest.mark.fractal
def test_spec_and_crystal_mismatch_raise(snowflake_client, ammonia, benzene): ...

def test_missing_extra_gives_install_hint(monkeypatch): ...      # unmarked; hides qcportal
```

- [ ] **Step 2:** Run `pytest tests/compute/test_fractal.py -v`; expected FAIL (or all-skip without qcfractal — then install `qcfractal` + Postgres into a test env to actually run them; report if that was not possible).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `pytest tests/compute/test_fractal.py -v -m fractal`; expected PASS.
- [ ] **Step 5:** Commit `feat(compute): add FractalBackend`.

---

### Task 9: User-facing example and docs

**Files:**
- Create: `scripts/run_lattice_energy.py` (the spec's "User-facing shape" script, runnable with `--backend local|inputs`)
- Modify: `README.md` (short "Computing lattice energies" section: the three backends, extras, per-formula-unit definition, deformation-energy caveat)

- [ ] **Step 1:** Write both; run `PYTHONPATH=src python scripts/run_lattice_energy.py tests/x23/ammonia/ammonia.cif --backend inputs --out <tmp>`; expected input files written under `<tmp>/dimers/`.
- [ ] **Step 2:** Run full `pytest -q`; expected green (fractal/psi4 tests skipped or passing).
- [ ] **Step 3: One-off v1 validation (manual, not CI).** With Psi4 installed, run `LocalBackend` on benzene at the level used by the v1 outputs on `master` (`git show master:Tests/Analysis/Benzene/`; read method/basis/cutoffs from those `.out` files) for the dimers and trimers present there, and compare per-multimer nonadditive energies and the lattice energy with v1 (contribution = energy × replicas / N). Record the comparison table and any discrepancies in the PR description; do not commit the outputs.
- [ ] **Step 4:** Commit `docs(compute): add lattice energy example`.
