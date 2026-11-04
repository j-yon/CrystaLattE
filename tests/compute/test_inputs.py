import json

import pytest

import crystalatte as cle
from crystalatte.compute import ComputeSpec, InputFileBackend

pytest.importorskip("qcengine")
qcmanybody = pytest.importorskip("qcmanybody")


def _spec(program="nwchem", bsse="cp"):
    return ComputeSpec(program=program, method="hf", basis="sto-3g", bsse=bsse, use_defaults=False)


def _labels(multimer, spec):
    core = qcmanybody.ManyBodyCore(multimer.to_molecule(), **spec.manybody_kwargs(len(multimer.monomers)))
    return {label for _, label, _ in core.iterate_molecules()}


@pytest.fixture(scope="module")
def nh3_dimers(ammonia):
    return cle.generate(ammonia, N=2, R=4.0, verbose=0)


def _manifest(path):
    return json.loads((path / "manifest.json").read_text())


@pytest.mark.parametrize("program, infile", [("nwchem", "nwchem.nw"), ("molpro", "dispatch.mol")])
def test_qcengine_program_writes_one_input_per_subcalculation(tmp_path, nh3_dimers, program, infile):
    spec = _spec(program)
    InputFileBackend(tmp_path, memory=2).submit(nh3_dimers, spec)
    entries = _manifest(tmp_path)["multimers"]
    for m in nh3_dimers:
        tasks = entries[m.id]["tasks"]
        assert {t["label"] for t in tasks} == _labels(m, spec)
        for t in tasks:
            assert (tmp_path / t["dir"] / infile).exists()
            assert t["infiles"] == [f"{t['dir']}/{infile}"]
            assert set(t["real"]) <= set(t["basis"])


def test_nwchem_counterpoise_input_has_ghost_atoms(tmp_path, nh3_dimers):
    InputFileBackend(tmp_path, memory=2).submit(nh3_dimers[:1], _spec("nwchem"))
    tasks = _manifest(tmp_path)["multimers"][nh3_dimers[0].id]["tasks"]
    ghosted = [t for t in tasks if len(t["basis"]) > len(t["real"])]
    assert ghosted
    for t in ghosted:
        assert "bqN" in (tmp_path / t["infiles"][0]).read_text()


def test_unverified_ghost_program_refuses_counterpoise(tmp_path, nh3_dimers):
    with pytest.raises(ValueError, match="ghost"):
        InputFileBackend(tmp_path).submit(nh3_dimers, _spec("qchem", bsse="cp"))
    assert not (tmp_path / "manifest.json").exists()


def test_unverified_ghost_program_allows_nocp(tmp_path, nh3_dimers):
    InputFileBackend(tmp_path, memory=2).submit(nh3_dimers, _spec("qchem", bsse="nocp"))
    for t in _manifest(tmp_path)["multimers"][nh3_dimers[0].id]["tasks"]:
        assert t["real"] == t["basis"]
        assert (tmp_path / t["infiles"][0]).exists()


def test_program_without_input_writer_points_to_local_backend(tmp_path, nh3_dimers):
    with pytest.raises(ValueError, match="LocalBackend"):
        InputFileBackend(tmp_path).submit(nh3_dimers, _spec("rdkit", bsse="nocp"))


def test_manifest_records_run(tmp_path, nh3_dimers):
    spec = _spec("nwchem")
    InputFileBackend(tmp_path, memory=2).submit(nh3_dimers, spec)
    manifest = _manifest(tmp_path)
    assert manifest["spec_fingerprint"] == spec.fingerprint()
    assert ComputeSpec.model_validate(manifest["spec"]) == spec
    assert manifest["crystal_fingerprint"] == nh3_dimers[0].generation.crystal_fingerprint
    assert manifest["crystalatte_version"]
    assert manifest["active"] == {"2": [m.id for m in nh3_dimers]}
    for m in nh3_dimers:
        entry = manifest["multimers"][m.id]
        assert (entry["N"], entry["multiplicity"], entry["weight"]) == (2, m.multiplicity, m.weight)
        assert entry["geometric_mean"] == m.geometric_mean
        assert entry["min_separation"] == m.min_separation
        assert entry["formulas"] == ["H3N", "H3N"]


def test_psi4_manifest_lists_whole_multimer_file(tmp_path, nh3_dimers):
    InputFileBackend(tmp_path).submit(nh3_dimers, _spec("psi4"))
    manifest = _manifest(tmp_path)
    for m in nh3_dimers:
        entry = manifest["multimers"][m.id]
        assert entry["files"] == [f"dimers/{m.id}.in"]
        assert "tasks" not in entry


def test_manifest_labels_feed_qcmanybody_analysis(tmp_path, nh3_dimers):
    spec = _spec("nwchem")
    InputFileBackend(tmp_path, memory=2).submit(nh3_dimers[:1], spec)
    m = nh3_dimers[0]
    tasks = _manifest(tmp_path)["multimers"][m.id]["tasks"]
    core = qcmanybody.ManyBodyCore(m.to_molecule(), **spec.manybody_kwargs(2))
    out = core.analyze({t["label"]: {"energy": -1.0 - 0.1 * len(t["real"])} for t in tasks})
    assert "cp_corrected_2_body_contribution_to_energy" in out["results"]


def test_resubmitting_other_spec_into_same_dir_raises(tmp_path, nh3_dimers):
    InputFileBackend(tmp_path, memory=2).submit(nh3_dimers, _spec("nwchem"))
    with pytest.raises(ValueError, match="spec"):
        InputFileBackend(tmp_path, memory=2).submit(nh3_dimers, _spec("molpro"))
