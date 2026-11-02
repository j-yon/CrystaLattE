import warnings

import pydantic
import pytest

from crystalatte.compute import ComputeSpec
from crystalatte.compute import spec as spec_module

PSI4 = {"program": "psi4", "method": "hf", "basis": "sto-3g"}


def test_psi4_defaults_merged_user_wins():
    s = ComputeSpec(program="psi4", method="mp2", basis="aug-cc-pvdz", keywords={"scf_type": "pk"})
    assert s.final_keywords == {
        "e_convergence": 10,
        "d_convergence": 10,
        "scf_type": "pk",
        "mp2_type": "df",
        "cc_type": "df",
        "freeze_core": "true",
    }


def test_use_defaults_false():
    assert ComputeSpec(**PSI4, use_defaults=False).final_keywords == {}


def test_unknown_program_warns_once(monkeypatch):
    monkeypatch.setattr(spec_module, "_WARNED_PROGRAMS", set())
    s = ComputeSpec(program="nwchem", method="hf", basis="sto-3g")
    with pytest.warns(UserWarning, match="convergence"):
        assert s.final_keywords == {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.final_keywords


def test_levels_contiguous():
    assert ComputeSpec(**PSI4).manybody_kwargs(3)["levels"] == {
        1: "crystalatte",
        2: "crystalatte",
        3: "crystalatte",
    }


def test_manybody_kwargs_fields():
    kw = ComputeSpec(**PSI4, bsse="NoCP").manybody_kwargs(2)
    assert "molecule" not in kw
    assert kw["bsse_type"] == ["nocp"]
    assert kw["return_total_data"] is False
    assert kw["supersystem_ie_only"] is False
    assert kw["embedding_charges"] == {}


def test_atomic_template():
    t = ComputeSpec(**PSI4).atomic_template()
    assert t["driver"] == "energy"
    assert t["model"] == {"method": "hf", "basis": "sto-3g"}
    assert t["keywords"] == ComputeSpec(**PSI4).final_keywords


def test_lowercases():
    s = ComputeSpec(program="Psi4", method="MP2", basis="Cc-pVDZ")
    assert (s.program, s.method, s.basis) == ("psi4", "mp2", "cc-pvdz")


def test_bad_bsse_rejected():
    with pytest.raises(pydantic.ValidationError):
        ComputeSpec(**PSI4, bsse="ssfc")


def test_round_trip_and_fingerprint():
    s = ComputeSpec(program="psi4", method="mp2", basis="cc-pvdz")
    t = ComputeSpec.model_validate(s.model_dump(mode="json"))
    assert t == s and t.fingerprint() == s.fingerprint()
    a = ComputeSpec(program="psi4", method="mp2", basis="cc-pvdz", keywords={"a": 1, "b": 2})
    b = ComputeSpec(program="psi4", method="mp2", basis="cc-pvdz", keywords={"b": 2, "a": 1})
    assert a.fingerprint() == b.fingerprint()


def test_unknown_field_rejected():
    with pytest.raises(pydantic.ValidationError):
        ComputeSpec(**PSI4, basiss="oops")


def test_frozen():
    s = ComputeSpec(**PSI4)
    with pytest.raises(pydantic.ValidationError):
        s.basis = "cc-pvdz"


def test_to_portal():
    pytest.importorskip("qcportal")
    p = ComputeSpec(**PSI4).to_portal(2)
    assert sorted(p.levels) == [1, 2] and p.bsse_correction == ["cp"]
