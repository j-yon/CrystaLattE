"""SymOp xyz strings: formatting from components must invert parsing.

Ops built from components (spglib detection, lattice-translated partners,
lazily materialized cluster members) carry no source string, so ``SymOp.op``
formats one on demand. These tests hold that string to round-tripping.
"""

import numpy as np
import pytest

import crystalatte as cle
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

from conftest import DATA, GENERATE_KW, crystal

FIXTURES = sorted(p.name for sub in ("x23", "other") for p in (DATA / sub).iterdir() if p.is_dir())


def _table_ops() -> list[str]:
    SpaceGroup.load_data()
    return sorted({s for d in SpaceGroup._data.values() for s in d["sym_ops"]})


@pytest.mark.parametrize("shift", [(0, 0, 0), (1, -2, 0), (-3, 1, 2)])
def test_format_round_trips_every_table_op(shift):
    shift12 = np.array(shift, dtype=np.int16) * 12
    for s in _table_ops():
        moved = SymOp(s).translate(shift12)
        assert SymOp(moved.op) == moved, s


def test_translate_reformats_instead_of_keeping_the_stale_string():
    op = SymOp("-x,y+1/2,-z")
    assert op.translate(np.array([12, 0, -24], dtype=np.int16)).op == "-x+1,y+1/2,-z-2"
    # a zero shift is still the same op, so its original spelling survives
    assert op.translate(np.zeros(3, dtype=np.int16)).op == "-x,y+1/2,-z"


def test_from_components_formats_on_demand():
    rot = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]], dtype=np.int16)
    op = SymOp.from_components(rot, np.array([0, 0, 8], dtype=np.int16))
    assert op.op == "-y,x-y,z+2/3"


def test_format_rejects_coefficients_outside_unit():
    rot = np.diag([2, 1, 1]).astype(np.int16)
    with pytest.raises(ValueError, match="±1"):
        SymOp.from_components(rot, np.zeros(3, dtype=np.int16)).op


@pytest.mark.parametrize("name", FIXTURES)
def test_generated_monomer_ops_round_trip(name):
    # covers spglib-detected ops, which are built from components in the crystal's own basis
    for m in cle.generate(crystal(name), N=2, R=6.0, **GENERATE_KW):
        for mon in m.monomers:
            assert SymOp(mon.symop.op) == mon.symop
