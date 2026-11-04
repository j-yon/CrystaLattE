import json
import signal
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("qcmanybody")
from qcmanybody import ManyBodyCore

import crystalatte as cle
from crystalatte.compute import ComputeSpec, IncompleteResultsError
from crystalatte.compute.backends import LocalBackend, local

SPEC = ComputeSpec(program="psi4", method="hf", basis="sto-3g")


def _toy_energy(atomic_input: dict) -> float:
    """-sum 1/r over pairs of real atoms, so the interaction energy is nonzero"""
    mol = atomic_input["molecule"]
    xyz = np.reshape(mol["geometry"], (-1, 3))
    xyz = xyz[np.asarray(mol.get("real", [True] * len(xyz)), dtype=bool)]
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return -float(np.sum(1.0 / d[np.triu_indices(len(xyz), 1)]))


class FakeEngine:
    def __init__(self, fail=lambda call: False):
        self.calls = 0
        self.fail = fail
        self.task_configs = []

    def __call__(self, atomic_input, program, task_config):
        self.calls += 1
        self.task_configs.append(task_config)
        if self.fail(self.calls):
            return {"success": False, "energy": None, "error": "boom"}
        return {"success": True, "energy": _toy_energy(atomic_input), "error": None}


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(local, "_run_task", engine)
    return engine


@pytest.fixture
def failing_engine(monkeypatch):
    engine = FakeEngine(fail=lambda call: True)
    monkeypatch.setattr(local, "_run_task", engine)
    return engine


def n_tasks(multimers, spec=SPEC) -> int:
    return sum(
        len(list(ManyBodyCore(m.to_molecule(), **spec.manybody_kwargs(len(m.monomers))).iterate_molecules()))
        for m in multimers
    )


def test_submit_collect_and_fresh_object(tmp_path, fake_engine, benzene):
    ms = cle.generate(benzene, N=2, R=6.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(ms, SPEC)
    r1 = b.collect()
    r2 = LocalBackend(tmp_path, ncores=1).collect()
    assert r1.complete
    assert r1.lattice_energy != 0.0
    assert r1.lattice_energy == r2.lattice_energy
    assert fake_engine.calls == n_tasks(ms)
    assert {row.id for row in r1.rows} == {m.id for m in ms}
    assert {row.id: row.weight for row in r1.rows} == {m.id: m.weight for m in ms}


def test_energy_is_cp_interaction_energy(tmp_path, fake_engine, ammonia):
    m = cle.generate(ammonia, N=2, R=4.0, verbose=0)[0]
    b = LocalBackend(tmp_path, ncores=1)
    b.submit([m], SPEC)
    mol = m.to_molecule()
    ab, a, b_ = (json.loads(x.json()) for x in (mol, mol.get_fragment([0], [1]), mol.get_fragment([1], [0])))
    expected = _toy_energy({"molecule": ab}) - _toy_energy({"molecule": a}) - _toy_energy({"molecule": b_})
    assert b.collect().rows[0].energy == pytest.approx(expected, abs=1e-12)


def test_workdir_layout(tmp_path, fake_engine, ammonia):
    ms = cle.generate(ammonia, N=2, R=4.0, verbose=0)
    LocalBackend(tmp_path, ncores=1).submit(ms, SPEC)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["spec_fingerprint"] == SPEC.fingerprint()
    assert ComputeSpec.model_validate(manifest["spec"]) == SPEC
    assert manifest["crystal_fingerprint"] == ms[0].generation.crystal_fingerprint
    assert manifest["active"] == {"2": [m.id for m in ms]}
    assert set(manifest["multimers"]) == {m.id for m in ms}
    for m in ms:
        task_dir = tmp_path / "tasks" / m.id
        assert (task_dir / "result.json").exists()
        task_files = [p for p in task_dir.glob("*.json") if p.name != "result.json"]
        assert len(task_files) == n_tasks([m])
        assert all(json.loads(p.read_text())["success"] for p in task_files)


def test_resume_skips_finished_tasks(tmp_path, monkeypatch, benzene):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)
    engine = FakeEngine(fail=lambda call: call == 2)
    monkeypatch.setattr(local, "_run_task", engine)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(ms, SPEC)
    assert not b.collect(partial=True).complete
    first = engine.calls
    assert first == n_tasks(ms)

    b.submit(ms, SPEC)
    assert engine.calls - first == 1
    assert b.collect().complete


def test_failure_reported_not_raised(tmp_path, failing_engine, benzene):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(ms, SPEC)
    assert "failed" in str(b.status())
    assert b.status().counts[2]["failed"] == len(ms)
    with pytest.raises(IncompleteResultsError, match="boom"):
        b.collect()
    assert b.collect(partial=True).missing[2]["failed"]


def test_analysis_error_marks_failed_and_retries(tmp_path, fake_engine, ammonia):
    ms = cle.generate(ammonia, N=2, R=4.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)

    def broken(self, *args, **kwargs):
        raise RuntimeError("analysis exploded")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ManyBodyCore, "analyze", broken)
        b.submit(ms, SPEC)
        b.submit(ms, SPEC)
    assert b.status().counts[2] == {"complete": 0, "pending": 0, "failed": len(ms)}
    with pytest.raises(IncompleteResultsError, match="analysis exploded"):
        b.collect()

    calls = fake_engine.calls
    b.submit(ms, SPEC)
    assert fake_engine.calls == calls
    assert b.collect().complete


def test_manifest_records_crystalatte_version(tmp_path, fake_engine, ammonia):
    from importlib.metadata import version

    LocalBackend(tmp_path, ncores=1).submit(cle.generate(ammonia, N=2, R=4.0, verbose=0), SPEC)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["crystalatte_version"] == {"2": version("crystalatte")}


def test_status_counts_complete(tmp_path, fake_engine, benzene):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(ms, SPEC)
    assert b.status().counts == {2: {"complete": len(ms), "pending": 0, "failed": 0}}


def test_collect_without_submission_raises(tmp_path):
    with pytest.raises(ValueError, match="no submission"):
        LocalBackend(tmp_path, ncores=1).collect()


def test_spec_mismatch_raises(tmp_path, fake_engine, benzene):
    ms = cle.generate(benzene, N=2, R=5.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(ms, SPEC)
    other = ComputeSpec(program="psi4", method="hf", basis="cc-pvdz")
    with pytest.raises(ValueError, match="spec"):
        b.submit(ms, other)


def test_crystal_mismatch_raises(tmp_path, fake_engine, benzene, ammonia):
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(cle.generate(benzene, N=2, R=5.0, verbose=0), SPEC)
    with pytest.raises(ValueError, match="crystal fingerprint"):
        b.submit(cle.generate(ammonia, N=2, R=4.0, verbose=0), SPEC)


def test_larger_R_reuses_and_smaller_R_excludes(tmp_path, fake_engine, benzene):
    small = cle.generate(benzene, N=2, R=5.0, verbose=0)
    big = cle.generate(benzene, N=2, R=7.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    b.submit(small, SPEC)
    calls_small = fake_engine.calls

    b.submit(big, SPEC)
    new_ids = {m.id for m in big} - {m.id for m in small}
    assert new_ids and len(new_ids) < len(big)
    assert fake_engine.calls - calls_small == n_tasks(m for m in big if m.id in new_ids)
    assert {row.id for row in b.collect().rows} == {m.id for m in big}

    calls_big = fake_engine.calls
    b.submit(small, SPEC)
    assert fake_engine.calls == calls_big
    assert {row.id for row in b.collect().rows} == {m.id for m in small}


def test_missing_extra_gives_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "qcengine", None)
    with pytest.raises(ImportError, match=r"crystalatte\[local\]"):
        LocalBackend(tmp_path, ncores=1)


def test_concurrency_from_cores():
    assert LocalBackend("unused", ncores=8, ncores_per_task=2, memory_per_task=1.0).concurrency == 4
    assert LocalBackend("unused", ncores=1, ncores_per_task=4, memory_per_task=1.0).concurrency == 1


def test_workdir_is_the_job_state_directory(tmp_path, fake_engine, ammonia):
    b = LocalBackend(workdir=tmp_path, ncores=1)
    b.submit(cle.generate(ammonia, N=2, R=4.0, verbose=0), SPEC)
    assert b.workdir == tmp_path
    assert (tmp_path / "manifest.json").exists()
    with pytest.raises(TypeError):
        LocalBackend(scratch=tmp_path, ncores=1)


def test_qc_scratch_unset_leaves_qcengine_resolution(tmp_path, fake_engine, ammonia):
    LocalBackend(tmp_path, ncores=1).submit(cle.generate(ammonia, N=2, R=4.0, verbose=0), SPEC)
    assert fake_engine.task_configs
    assert all("scratch_directory" not in c for c in fake_engine.task_configs)


def test_qc_scratch_is_passed_to_every_task(tmp_path, fake_engine, ammonia):
    qc = tmp_path / "qc"
    b = LocalBackend(tmp_path / "work", qc_scratch=qc, ncores=1)
    b.submit(cle.generate(ammonia, N=2, R=4.0, verbose=0), SPEC)
    assert fake_engine.task_configs
    assert all(c["scratch_directory"] == str(qc) for c in fake_engine.task_configs)
    assert qc.is_dir()


@pytest.mark.psi4
def test_psi4_scratch_location(tmp_path, monkeypatch, ammonia):
    pytest.importorskip("psi4")
    spec = ComputeSpec(program="psi4", method="hf", basis="sto-3g", use_defaults=False)
    m = cle.generate(ammonia, N=2, R=4.0, verbose=0)[0]
    psi_scratch, explicit = tmp_path / "psi_scratch", tmp_path / "explicit"
    psi_scratch.mkdir()
    monkeypatch.setenv("PSI_SCRATCH", str(psi_scratch))
    monkeypatch.setenv("QCENGINE_SCRATCH_MESSY", "true")  # keep per-task dirs so they can be found

    LocalBackend(tmp_path / "w1", ncores=1, memory_per_task=1.0).submit([m], spec)
    assert list(psi_scratch.glob("*_psi_scratch")), "unset qc_scratch must fall back to $PSI_SCRATCH"

    LocalBackend(tmp_path / "w2", qc_scratch=explicit, ncores=1, memory_per_task=1.0).submit([m], spec)
    assert list(explicit.glob("*_psi_scratch")), "explicit qc_scratch must win over $PSI_SCRATCH"


@pytest.mark.psi4
def test_cp_dimer_matches_direct_psi4(tmp_path, ammonia):
    pytest.importorskip("psi4")
    import qcengine

    spec = ComputeSpec(program="psi4", method="hf", basis="sto-3g", use_defaults=False)
    m = cle.generate(ammonia, N=2, R=4.0, verbose=0)[0]
    # two workers, so this also exercises the process-pool path
    b = LocalBackend(tmp_path, ncores=2, ncores_per_task=1, memory_per_task=1.0)
    b.submit([m], spec)
    energy = b.collect().rows[0].energy

    mol = m.to_molecule()
    task_config = {"ncores": 1, "memory": 1.0}

    def e(molecule):
        out = qcengine.compute(
            {**spec.atomic_template(), "molecule": molecule}, "psi4",
            raise_error=True, task_config=task_config,
        )
        return out.return_result

    direct = e(mol) - e(mol.get_fragment([0], [1])) - e(mol.get_fragment([1], [0]))
    print(f"LocalBackend {energy!r} direct {direct!r}")
    assert energy == pytest.approx(direct, abs=1e-8)


def test_run_task_keeps_version_wall_time_and_threads(monkeypatch):
    import qcengine

    provenance = {"creator": "Psi4", "version": "1.10", "wall_time": 1.5, "nthreads": 2, "username": "someone"}
    result = {"success": True, "return_result": -1.0, "provenance": provenance}
    monkeypatch.setattr(qcengine, "compute", lambda *args, **kwargs: result)
    outcome = local._run_task({}, "psi4", {})
    assert outcome["energy"] == -1.0
    assert outcome["provenance"] == {"version": "1.10", "wall_time": 1.5, "nthreads": 2}


def test_run_task_tolerates_programs_reporting_less_provenance(monkeypatch):
    import qcengine

    result = {"success": True, "return_result": -1.0, "provenance": {"creator": "x", "version": "2"}}
    monkeypatch.setattr(qcengine, "compute", lambda *args, **kwargs: result)
    assert local._run_task({}, "x", {})["provenance"] == {"version": "2", "wall_time": None, "nthreads": None}


def test_run_task_failure_has_no_provenance(monkeypatch):
    import qcengine

    result = {"success": False, "error": {"error_message": "boom"}}
    monkeypatch.setattr(qcengine, "compute", lambda *args, **kwargs: result)
    outcome = local._run_task({}, "psi4", {})
    assert outcome["error"] == "boom"
    assert "provenance" not in outcome


@pytest.mark.psi4
def test_task_files_record_psi4_provenance(tmp_path, ammonia):
    pytest.importorskip("psi4")
    spec = ComputeSpec(program="psi4", method="hf", basis="sto-3g", use_defaults=False)
    m = cle.generate(ammonia, N=2, R=4.0, verbose=0)[0]
    LocalBackend(tmp_path, ncores=2, ncores_per_task=2, memory_per_task=1.0).submit([m], spec)
    tasks = [p for p in (tmp_path / "tasks" / m.id).glob("*.json") if p.name != "result.json"]
    assert tasks
    for path in tasks:
        provenance = json.loads(path.read_text())["provenance"]
        assert provenance["version"]
        assert provenance["wall_time"] > 0
        assert provenance["nthreads"] == 2


HOLD_LOCK = """
import sys
from pathlib import Path
from crystalatte.compute.backends import local
with local._workdir_lock(Path(sys.argv[1])):
    print("locked", flush=True)
    sys.stdin.read()
"""


@pytest.fixture
def other_process_lock(tmp_path):
    """Another live process holding the workdir lock"""
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLD_LOCK, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout.readline().strip() == "locked"
    yield holder
    holder.kill()
    holder.wait()


def test_submit_refused_while_another_process_holds_workdir(tmp_path, fake_engine, ammonia, other_process_lock):
    ms = cle.generate(ammonia, N=2, R=4.0, verbose=0)
    with pytest.raises(RuntimeError, match=rf"pid {other_process_lock.pid}"):
        LocalBackend(tmp_path, ncores=1).submit(ms, SPEC)
    assert fake_engine.calls == 0
    assert not (tmp_path / "manifest.json").exists()


def test_lock_released_when_holder_is_killed(tmp_path, fake_engine, ammonia, other_process_lock):
    other_process_lock.send_signal(signal.SIGKILL)
    other_process_lock.wait()
    ms = cle.generate(ammonia, N=2, R=4.0, verbose=0)
    LocalBackend(tmp_path, ncores=1).submit(ms, SPEC)
    assert fake_engine.calls == n_tasks(ms)


def test_lock_released_after_interrupted_submit(tmp_path, monkeypatch, ammonia):
    ms = cle.generate(ammonia, N=2, R=4.0, verbose=0)

    def interrupted(*args):
        raise KeyboardInterrupt

    with monkeypatch.context() as mp:
        mp.setattr(local, "_run_task", interrupted)
        with pytest.raises(KeyboardInterrupt):
            LocalBackend(tmp_path, ncores=1).submit(ms, SPEC)

    engine = FakeEngine()
    monkeypatch.setattr(local, "_run_task", engine)
    LocalBackend(tmp_path, ncores=1).submit(ms, SPEC)
    assert engine.calls == n_tasks(ms)


def test_status_and_collect_work_during_a_submit(tmp_path, monkeypatch, ammonia):
    ms = cle.generate(ammonia, N=2, R=4.0, verbose=0)
    b = LocalBackend(tmp_path, ncores=1)
    seen = []

    def engine(atomic_input, program, task_config):
        seen.append((b.status().counts[2], b.collect(partial=True).complete))
        with pytest.raises(RuntimeError, match="in use"):
            LocalBackend(tmp_path, ncores=1).submit(ms, SPEC)
        return {"success": True, "energy": _toy_energy(atomic_input), "error": None}

    monkeypatch.setattr(local, "_run_task", engine)
    b.submit(ms, SPEC)
    assert len(seen) == n_tasks(ms)
    assert seen[0][1] is False
    assert b.collect().complete
