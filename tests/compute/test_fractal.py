import importlib.util
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import pydantic
import pytest

import crystalatte as cle
from crystalatte.compute import ComputeSpec
from crystalatte.compute.backends import FractalBackend, LocalBackend
from crystalatte.compute.backends.base import manifest_entry
from crystalatte.core.multimer import Multimer, order_name

# tight SCF so Local and Fractal agree to ~1e-11 Eh per dimer; at Psi4's default convergence
# repeated runs of the same dimer drift by ~1e-9 Eh, i.e. ~1e-6 kJ/mol in the lattice energy
SPEC = ComputeSpec(
    program="psi4",
    method="hf",
    basis="sto-3g",
    keywords={"e_convergence": 1e-10, "d_convergence": 1e-10},
    use_defaults=False,
)
TIMEOUT = 600.0


@pytest.fixture(scope="module")
def psi4_preloaded() -> bool:
    """Whether psi4 was imported in this process before the Snowflake forked its workers"""
    return "psi4" in sys.modules


@pytest.fixture(scope="module")
def snowflake_client(psi4_preloaded):
    """A throwaway QCFractal server with Psi4 workers, shared by the module"""
    pytest.importorskip("qcfractal")
    if importlib.util.find_spec("psi4") is None:
        pytest.skip("psi4 is not installed")
    from qcfractal.snowflake import FractalSnowflake

    with pytest.MonkeyPatch.context() as mp:
        # the env may not be activated: Postgres tools and psi4 live next to this interpreter
        mp.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
        with FractalSnowflake(compute_workers=2) as snowflake:
            yield snowflake.client()


def wait(backend: FractalBackend, timeout: float = TIMEOUT) -> None:
    """Poll until nothing is pending, or fail after `timeout` seconds"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        counts = backend.status().counts
        if all(c["pending"] == 0 for c in counts.values()):
            return
        time.sleep(2.0)
    pytest.fail(f"Fractal calculations still pending after {timeout} s: {backend.status()}")


def dimers(crystal):
    return cle.generate(crystal, N=2, R=4.0, verbose=0)


@pytest.mark.fractal
def test_submit_creates_one_dataset_per_order(snowflake_client, ammonia):
    ms2, ms3 = dimers(ammonia), cle.generate(ammonia, N=3, R=3.0, verbose=0)
    FractalBackend(snowflake_client, "orders", priority="low").submit(ms2 + ms3, SPEC)

    names = {d["dataset_name"] for d in snowflake_client.list_datasets()}
    assert {"orders-dimers", "orders-trimers"} <= names
    for N, ms in ((2, ms2), (3, ms3)):
        ds = snowflake_client.get_dataset("manybody", f"orders-{order_name(N)}s")
        assert set(ds.entry_names) == {m.id for m in ms}
        assert ds.specification_names == ["crystalatte"]
        meta = ds.extras["crystalatte"]
        assert meta["active"] == [m.id for m in ms]
        assert meta["multimers"] == {m.id: manifest_entry(m) for m in ms}
        assert meta["crystalatte_version"] == version("crystalatte")
        assert meta["spec_fingerprint"] == SPEC.fingerprint()
        assert meta["crystal_fingerprint"] == ms[0].generation.crystal_fingerprint
        assert ComputeSpec.model_validate(meta["spec"]) == SPEC
        entry = ds.get_entry(ms[0].id)
        assert entry.attributes == {
            **manifest_entry(ms[0]),
            "crystal_fingerprint": ms[0].generation.crystal_fingerprint,
            "crystalatte_version": version("crystalatte"),
        }


@pytest.mark.fractal
def test_resubmit_is_idempotent(snowflake_client, ammonia):
    ms = dimers(ammonia)
    b = FractalBackend(snowflake_client, "again", priority="low")
    b.submit(ms[:2], SPEC)
    ds = snowflake_client.get_dataset("manybody", "again-dimers")
    ds.set_extras({**ds.extras, "mine": 1})

    b.submit(ms, SPEC)
    b.submit(ms, SPEC)
    ds = snowflake_client.get_dataset("manybody", "again-dimers")
    assert sorted(ds.entry_names) == sorted(m.id for m in ms)
    assert len(list(ds.iterate_records(specification_names="crystalatte"))) == len(ms)
    assert ds.extras["mine"] == 1
    assert ds.extras["crystalatte"]["active"] == [m.id for m in ms]


@pytest.mark.fractal
def test_collect_matches_local(snowflake_client, psi4_preloaded, tmp_path, ammonia):
    if psi4_preloaded:
        pytest.skip(
            "psi4 already imported in this process; forked Snowflake workers inherit its state "
            "- run tests/compute/test_fractal.py on its own"
        )
    ms = dimers(ammonia)
    fractal = FractalBackend(snowflake_client, "nh3")
    fractal.submit(ms, SPEC)

    local = LocalBackend(tmp_path, ncores=1)
    local.submit(ms, SPEC)
    expected = local.collect()

    wait(fractal)
    got = fractal.collect()
    fresh = FractalBackend(snowflake_client, "nh3").collect()

    assert got.complete
    assert got.lattice_energy != 0.0
    assert got.lattice_energy == pytest.approx(expected.lattice_energy, abs=1e-6)
    assert fresh.lattice_energy == got.lattice_energy
    assert {r.id: (r.weight, r.multiplicity) for r in got.rows} == {
        m.id: (m.weight, m.multiplicity) for m in ms
    }
    by_id = {r.id: r.energy for r in expected.rows}
    for row in got.rows:
        assert row.energy == pytest.approx(by_id[row.id], abs=1e-10)
    assert str(fractal.status()) == f"dimers: {len(ms)} complete, 0 pending, 0 failed"


@pytest.mark.fractal
def test_spec_and_crystal_mismatch_raise(snowflake_client, ammonia, benzene):
    ms = dimers(ammonia)
    b = FractalBackend(snowflake_client, "mismatch", priority="low")
    b.submit(ms, SPEC)

    with pytest.raises(ValueError, match="spec fingerprint"):
        b.submit(ms, SPEC.model_copy(update={"bsse": "nocp"}))
    with pytest.raises(ValueError, match="spec fingerprint"):
        FractalBackend(snowflake_client, "mismatch").submit(
            cle.generate(ammonia, N=3, R=3.0, verbose=0), SPEC.model_copy(update={"basis": "6-31g"})
        )
    with pytest.raises(ValueError, match="crystal fingerprint"):
        b.submit(cle.generate(benzene, N=2, R=4.0, verbose=0), SPEC)
    names = {d["dataset_name"] for d in snowflake_client.list_datasets()}
    assert "mismatch-trimers" not in names


@pytest.mark.fractal
def test_resubmit_uses_new_weights(snowflake_client, ammonia):
    ms = dimers(ammonia)
    b = FractalBackend(snowflake_client, "reweight", priority="low")
    b.submit(ms, SPEC)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Multimer, "weight", property(lambda self: 0.125))
        b.submit(ms, SPEC)
    assert {r.weight for r in b.collect(partial=True).rows} == {0.125}
    fresh = FractalBackend(snowflake_client, "reweight").collect(partial=True)
    assert {r.id: r.weight for r in fresh.rows} == {m.id: 0.125 for m in ms}


@pytest.mark.fractal
def test_stored_specification_mismatch_raises(snowflake_client, ammonia):
    ms = dimers(ammonia)
    b = FractalBackend(snowflake_client, "stored", priority="low")
    b.submit(ms, SPEC)
    # extras that agree with another spec, as if written by a CrystaLattE with other defaults
    other = SPEC.model_copy(update={"keywords": {**SPEC.keywords, "maxiter": 33}})
    ds = snowflake_client.get_dataset("manybody", "stored-dimers")
    meta = ds.extras["crystalatte"]
    ds.set_extras(
        {
            **ds.extras,
            "crystalatte": {**meta, "spec": other.model_dump(mode="json"), "spec_fingerprint": other.fingerprint()},
        }
    )
    with pytest.raises(ValueError, match="specification 'crystalatte'"):
        b.submit(ms, other)


@pytest.mark.fractal
def test_corrupted_extras_raise_validation_error(snowflake_client, ammonia):
    b = FractalBackend(snowflake_client, "corrupt", priority="low")
    b.submit(dimers(ammonia), SPEC)
    ds = snowflake_client.get_dataset("manybody", "corrupt-dimers")
    meta = ds.extras["crystalatte"]
    ds.set_extras({**ds.extras, "crystalatte": {**meta, "spec": {**meta["spec"], "bsse": "bogus"}}})
    with pytest.raises(pydantic.ValidationError):
        b.status()


@pytest.mark.fractal
def test_failures_reported_and_reset(snowflake_client, ammonia):
    ms = dimers(ammonia)[:1]
    b = FractalBackend(snowflake_client, "broken")
    b.submit(ms, ComputeSpec(program="psi4", method="no-such-method", basis="sto-3g", use_defaults=False))

    deadline = time.monotonic() + TIMEOUT
    while b.status().counts[2]["failed"] == 0 and time.monotonic() < deadline:
        time.sleep(2.0)
    rows = b.collect(partial=True).rows
    assert [r.status for r in rows] == ["failed"]
    assert rows[0].error
    with pytest.raises(cle.IncompleteResultsError):
        b.collect()

    assert b.resubmit_errors() == 1


@pytest.mark.fractal
def test_cancelled_record_is_failed(snowflake_client, ammonia):
    ms = dimers(ammonia)[:1]
    b = FractalBackend(snowflake_client, "cancelled", priority="low")
    # a spec no other test uses, so the server cannot reuse an already-finished record
    b.submit(ms, SPEC.model_copy(update={"keywords": {**SPEC.keywords, "maxiter": 77}}))
    snowflake_client.get_dataset("manybody", "cancelled-dimers").cancel_records(
        specification_names="crystalatte"
    )

    rows = b.collect(partial=True).rows
    assert [r.status for r in rows] == ["failed"]
    assert "cancelled" in rows[0].error
    assert b.status().counts[2] == {"complete": 0, "pending": 0, "failed": 1}
    with pytest.raises(cle.IncompleteResultsError) as excinfo:
        b.collect()
    assert excinfo.value.missing == {2: {"failed": [ms[0].id], "pending": []}}
    assert b.resubmit_errors() == 0


@pytest.mark.fractal
def test_collect_before_submit_raises(snowflake_client):
    with pytest.raises(ValueError, match="submit"):
        FractalBackend(snowflake_client, "never").collect()


class _FakeDataset:
    """Stands in for a ManybodyDataset whose one record completed without the expected energy"""

    def __init__(self, m):
        from qcportal.record_models import RecordStatusEnum

        self.name = "fake-dimers"
        self.extras = {
            "crystalatte": {
                "spec": SPEC.model_dump(mode="json"),
                "spec_fingerprint": SPEC.fingerprint(),
                "crystal_fingerprint": m.generation.crystal_fingerprint,
                "generation": m.generation.model_dump(mode="json"),
                "active": [m.id],
                "multimers": {m.id: {"multiplicity": m.multiplicity, "weight": m.weight}},
            }
        }
        self._entry = SimpleNamespace(name=m.id, attributes={"multiplicity": m.multiplicity, "weight": m.weight})
        self._record = SimpleNamespace(status=RecordStatusEnum.complete, properties={"results": {}})

    def iterate_entries(self):
        yield self._entry

    def iterate_records(self, specification_names=None):
        yield self._entry.name, specification_names, self._record


def test_complete_record_without_energy_is_failed(ammonia):
    pytest.importorskip("qcportal")
    m = dimers(ammonia)[0]
    ds = _FakeDataset(m)
    client = SimpleNamespace(
        list_datasets=lambda: [{"dataset_name": ds.name, "dataset_type": "manybody"}],
        get_dataset=lambda kind, name: ds,
    )
    b = FractalBackend(client, "fake")
    rows = b.collect(partial=True).rows
    assert [r.status for r in rows] == ["failed"]
    assert "cp_corrected_2_body_contribution_to_energy" in rows[0].error
    assert b.status().counts[2] == {"complete": 0, "pending": 0, "failed": 1}
    with pytest.raises(cle.IncompleteResultsError, match="missing"):
        b.collect()


def _fake_client(ds):
    return SimpleNamespace(
        list_datasets=lambda: [{"dataset_name": ds.name, "dataset_type": "manybody"}],
        get_dataset=lambda kind, name: ds,
    )


def test_failed_row_keeps_multimer_metadata(ammonia):
    pytest.importorskip("qcportal")
    m = dimers(ammonia)[0]
    ds = _FakeDataset(m)
    ds.extras["crystalatte"]["multimers"][m.id] = manifest_entry(m)
    row = FractalBackend(_fake_client(ds), "fake").collect(partial=True).rows[0]
    assert row.status == "failed"
    assert row.geometric_mean == m.geometric_mean
    assert row.min_separation == m.min_separation
    assert row.formulas == ("H3N", "H3N")


def test_dataset_without_metadata_still_collects(ammonia):
    pytest.importorskip("qcportal")
    row = FractalBackend(_fake_client(_FakeDataset(dimers(ammonia)[0])), "fake").collect(partial=True).rows[0]
    assert row.min_separation is None


@pytest.mark.fractal
def test_rows_carry_multimer_metadata(snowflake_client, ammonia):
    ms = dimers(ammonia)
    b = FractalBackend(snowflake_client, "rowmeta", priority="low")
    b.submit(ms, SPEC)
    by_id = {r.id: r for r in b.collect(partial=True).rows}
    for m in ms:
        assert by_id[m.id].geometric_mean == m.geometric_mean
        assert by_id[m.id].min_separation == m.min_separation
        assert by_id[m.id].formulas == ("H3N", "H3N")


def test_missing_extra_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "qcportal", None)
    with pytest.raises(ImportError, match=r"crystalatte\[fractal\]"):
        FractalBackend(object(), "nh3")
