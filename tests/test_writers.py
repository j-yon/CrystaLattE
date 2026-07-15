"""Tests for QC input file writers (src/crystalatte/io/writers/) and the
write_inputs entry point (src/crystalatte/work/setup.py).

Run with:
    conda run -n cle python -m pytest Tests/test_writers.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from crystalatte.core.multimer import Monomer, Multimer
from crystalatte.core.sym_ops import SymOp, SymOpList


# ---------------------------------------------------------------------------
# Test helpers / fixtures
# ---------------------------------------------------------------------------

def _make_monomer(symbols: list[str], cart_coords: list) -> Monomer:
    cart = np.array(cart_coords, dtype=float)
    centroid = cart.mean(axis=0)
    return Monomer(
        _symbols=symbols,
        _frac_coords=np.zeros_like(cart),
        _cart_coords=cart,
        _centroid_frac=np.zeros(3),
        _centroid_cart=centroid,
    )


def _make_multimer(*monomers: Monomer) -> Multimer:
    ops = SymOpList([SymOp.identity()] * len(monomers), 1)
    return Multimer(list(monomers), ops)


@pytest.fixture
def h2():
    return _make_monomer(["H", "H"], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])


@pytest.fixture
def h2b():
    return _make_monomer(["H", "H"], [[4.0, 0.0, 0.0], [4.74, 0.0, 0.0]])


@pytest.fixture
def h2c():
    return _make_monomer(["H", "H"], [[8.0, 0.0, 0.0], [8.74, 0.0, 0.0]])


@pytest.fixture
def dimer(h2, h2b):
    return _make_multimer(h2, h2b)


@pytest.fixture
def trimer(h2, h2b, h2c):
    return _make_multimer(h2, h2b, h2c)


@pytest.fixture
def psi4_dimer_text(tmp_path, h2, h2b):
    """Write a Psi4 dimer input and return the file contents."""
    from crystalatte.io.writers.psi4 import Psi4Writer
    path = tmp_path / "dimer.inp"
    Psi4Writer().write(path, [h2, h2b], {"method": "mp2", "basis": "cc-pVDZ"})
    return path.read_text()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_get_writer_psi4_returns_psi4writer():
    from crystalatte.io.writers import get_writer
    from crystalatte.io.writers.psi4 import Psi4Writer
    assert isinstance(get_writer("psi4"), Psi4Writer)


def test_get_writer_is_case_insensitive():
    from crystalatte.io.writers import get_writer
    from crystalatte.io.writers.psi4 import Psi4Writer
    assert isinstance(get_writer("PSI4"), Psi4Writer)
    assert isinstance(get_writer("Psi4"), Psi4Writer)


def test_get_writer_unknown_raises_value_error():
    from crystalatte.io.writers import get_writer
    with pytest.raises(ValueError, match="Unknown software"):
        get_writer("gaussian")


# ---------------------------------------------------------------------------
# Psi4Writer: file creation
# ---------------------------------------------------------------------------

def test_psi4_writer_creates_file(tmp_path, h2, h2b):
    from crystalatte.io.writers.psi4 import Psi4Writer
    path = tmp_path / "out.inp"
    Psi4Writer().write(path, [h2, h2b], {"method": "hf", "basis": "sto-3g"})
    assert path.exists()


# ---------------------------------------------------------------------------
# Psi4Writer: molecule block
# ---------------------------------------------------------------------------

def test_psi4_molecule_block_separates_fragments_with_dashes(psi4_dimer_text):
    """Two-monomer input must contain the '--' fragment separator."""
    assert "--" in psi4_dimer_text


def test_psi4_each_fragment_opens_with_charge_mult(psi4_dimer_text):
    """Default charge=0, mult=1 → '0 1' appears once per fragment (2 total)."""
    assert psi4_dimer_text.count("0 1") == 2


def test_psi4_atom_symbols_appear_in_output(psi4_dimer_text):
    assert "H" in psi4_dimer_text


def test_psi4_atom_coords_appear_in_output(psi4_dimer_text):
    """Coordinate value 0.74 (H-H bond length) must be in the molecule block."""
    assert "0.74" in psi4_dimer_text or "0.740" in psi4_dimer_text


def test_psi4_custom_charge_and_mult(tmp_path, h2, h2b):
    from crystalatte.io.writers.psi4 import Psi4Writer
    path = tmp_path / "out.inp"
    Psi4Writer().write(path, [h2, h2b], {"method": "hf", "basis": "sto-3g", "charge": 1, "multiplicity": 2})
    text = path.read_text()
    assert text.count("1 2") == 2


# ---------------------------------------------------------------------------
# Psi4Writer: Python API structure
# ---------------------------------------------------------------------------

def test_psi4_has_import_statement(psi4_dimer_text):
    assert "import psi4" in psi4_dimer_text


def test_psi4_uses_geometry_function(psi4_dimer_text):
    assert "psi4.geometry(" in psi4_dimer_text


def test_psi4_uses_set_options(psi4_dimer_text):
    assert "psi4.set_options(" in psi4_dimer_text


def test_psi4_set_options_contains_basis(psi4_dimer_text):
    assert "'basis': 'cc-pVDZ'" in psi4_dimer_text


def test_psi4_energy_call_uses_psi4_prefix(psi4_dimer_text):
    assert "psi4.energy('mp2'" in psi4_dimer_text


def test_psi4_energy_call_contains_bsse_type(psi4_dimer_text):
    assert "bsse_type='cp'" in psi4_dimer_text


def test_psi4_default_bsse_type_is_cp(tmp_path, h2, h2b):
    from crystalatte.io.writers.psi4 import Psi4Writer
    path = tmp_path / "out.inp"
    Psi4Writer().write(path, [h2, h2b], {"method": "hf", "basis": "sto-3g"})
    assert "bsse_type='cp'" in path.read_text()


def test_psi4_memory_uses_set_memory(tmp_path, h2, h2b):
    from crystalatte.io.writers.psi4 import Psi4Writer
    path = tmp_path / "out.inp"
    Psi4Writer().write(path, [h2, h2b], {"method": "hf", "basis": "sto-3g", "memory": "8 GB"})
    assert "psi4.set_memory('8 GB')" in path.read_text()


def test_psi4_extra_config_key_goes_in_set_options(tmp_path, h2, h2b):
    from crystalatte.io.writers.psi4 import Psi4Writer
    path = tmp_path / "out.inp"
    Psi4Writer().write(path, [h2, h2b], {"method": "hf", "basis": "sto-3g", "freeze_core": True})
    assert "'freeze_core': True" in path.read_text()


# ---------------------------------------------------------------------------
# write_inputs: directory structure and file naming
# ---------------------------------------------------------------------------

def test_write_inputs_creates_dimers_directory(tmp_path, dimer):
    from crystalatte.work.setup import write_inputs
    write_inputs([dimer], tmp_path, "psi4", {"method": "hf", "basis": "sto-3g"})
    assert (tmp_path / "dimers").is_dir()


def test_write_inputs_creates_one_file_per_multimer(tmp_path, dimer):
    from crystalatte.work.setup import write_inputs
    write_inputs([dimer, dimer], tmp_path, "psi4", {"method": "hf", "basis": "sto-3g"})
    files = list((tmp_path / "dimers").glob("*.inp"))
    assert len(files) == 2


def test_write_inputs_dimer_file_name(tmp_path, dimer):
    from crystalatte.work.setup import write_inputs
    write_inputs([dimer], tmp_path, "psi4", {"method": "hf", "basis": "sto-3g"})
    assert (tmp_path / "dimers" / "dimer_001.inp").exists()


def test_write_inputs_trimer_uses_trimers_directory(tmp_path, trimer):
    from crystalatte.work.setup import write_inputs
    write_inputs([trimer], tmp_path, "psi4", {"method": "hf", "basis": "sto-3g"})
    assert (tmp_path / "trimers" / "trimer_001.inp").exists()


def test_write_inputs_mixed_orders_create_separate_dirs(tmp_path, dimer, trimer):
    from crystalatte.work.setup import write_inputs
    write_inputs([dimer, trimer], tmp_path, "psi4", {"method": "hf", "basis": "sto-3g"})
    assert (tmp_path / "dimers").is_dir()
    assert (tmp_path / "trimers").is_dir()


def test_write_inputs_str_path_accepted(tmp_path, dimer):
    from crystalatte.work.setup import write_inputs
    write_inputs([dimer], str(tmp_path), "psi4", {"method": "hf", "basis": "sto-3g"})
    assert (tmp_path / "dimers" / "dimer_001.inp").exists()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_write_inputs_importable_from_package():
    import crystalatte
    assert hasattr(crystalatte, "write_inputs")
