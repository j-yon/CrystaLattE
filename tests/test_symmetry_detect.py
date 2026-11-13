import pickle
import numpy as np
import pytest
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

from conftest import load


def test_from_sym_ops_holds_explicit_ops():
    ops = [SymOp.identity(), SymOp("-x,-y,-z")]
    sg = SpaceGroup.from_sym_ops(ops, number=2)
    assert len(sg.sym_ops) == 2
    assert sg.number == 2
    # not cached in the registry (detected groups are per-structure)
    assert SpaceGroup(2) is not sg


def test_str_of_group_built_from_ops_without_number():
    sg = SpaceGroup.from_sym_ops([SymOp.identity(), SymOp("-x,-y,-z")])
    assert "Space Group" in str(sg)


def test_str_of_detected_group_names_its_number():
    sg = SpaceGroup.from_sym_ops([SymOp.identity(), SymOp("-x,-y,-z")], number=2)
    assert "2" in str(sg)


def test_from_sym_ops_pickles_with_its_ops():
    # a pickled detected group (e.g. a Crystal sent to another process) must keep
    # their explicit input-basis ops, NOT reload the DB standard setting.
    ops = [SymOp.identity(), SymOp("-x,-y+1/2,-z")]
    sg = SpaceGroup.from_sym_ops(ops, number=14)
    round_tripped = pickle.loads(pickle.dumps(sg))
    got = {op.augment.tobytes() for op in round_tripped.sym_ops}
    want = {op.augment.tobytes() for op in ops}
    assert got == want


def test_get_unique_sym_ops_detected_group_dedups_by_centroid():
    # A group with a 2-fold that fixes the origin: at the origin the 2-fold
    # shares the identity's centroid image, so unique ops = 1 (deduped).
    ops = [SymOp.identity(), SymOp("-x,-y,z")]  # 2-fold about z through origin
    sg = SpaceGroup.from_sym_ops(ops)
    at_origin = sg.get_unique_sym_ops(np.array([0.0, 0.0, 0.0]))
    assert len(at_origin) == 1
    # at a general point both ops give distinct centroid images -> both kept
    at_general = sg.get_unique_sym_ops(np.array([0.11, 0.23, 0.37]))
    assert len(at_general) == 2


def test_get_unique_sym_ops_db_behavior_preserved():
    # DB benzene (SG 61) at the origin special position: same op count as before.
    sg = SpaceGroup(61)
    ops_origin = sg.get_unique_sym_ops(np.array([0.0, 0.0, 0.0]))
    ops_general = sg.get_unique_sym_ops(np.array([0.13, 0.21, 0.37]))
    assert len(ops_general) == len(sg.sym_ops)          # general position: all ops
    assert 1 <= len(ops_origin) <= len(sg.sym_ops)       # special position: deduped


def test_detect_symmetry_ammonia_p1_to_p213():
    crystal = load("ammonia")
    assert len(crystal.space_group.sym_ops) == 1          # declared P1
    detected = crystal.detect_symmetry()
    assert len(detected.space_group.sym_ops) == 12        # P2_13
    assert detected.space_group.number == 198
    # symmetry now relates the 4 NH3 -> a single reference
    assert len(detected.get_references()) == 1


def test_detect_symmetry_benzene_no_change():
    crystal = load("benzene")
    detected = crystal.detect_symmetry()
    # benzene's declared SG 61 is already correct -> not strictly higher -> self
    assert detected is crystal


def test_detect_symmetry_graceful_on_failure(monkeypatch):
    import spglib
    crystal = load("ammonia")
    monkeypatch.setattr(spglib, "get_symmetry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert crystal.detect_symmetry() is crystal            # falls back, no raise


def test_detect_symmetry_benzene_full_cell_expansion():
    # Benzene is non-P1 (SG 61, Z=4); detection must expand ASU to full cell
    # before passing to spglib. Verify it works (returns self, no raise/warning).
    import warnings
    crystal = load("benzene")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detected = crystal.detect_symmetry()
        # No symmetry-detection warnings (silenced spglib + handled errors)
        sym_warnings = [x for x in w if "symmetry detection" in str(x.message)]
        assert len(sym_warnings) == 0
    assert detected is crystal  # benzene already at full symmetry


from crystalatte.work import generate


def test_generate_detection_preserves_multiplicity_ammonia():
    # Without detection ammonia stays P1 with Z'=4, and dimers related by the missed symmetry are
    # not merged. The duplicates split the multiplicity, so the multiplicity per reference (what
    # the lattice sum weights) must not change.
    crystal = load("ammonia")
    z_on, z_off = len(crystal.detect_symmetry().get_references()), len(crystal.get_references())
    for kw in (dict(use_origin=True), dict()):  # use_origin and default conventions
        on = generate.generate(crystal, 2, 5.0, detect_symmetry=True, **kw)
        with pytest.warns(RuntimeWarning, match="symmetry detection is off"):
            off = generate.generate(crystal, 2, 5.0, detect_symmetry=False, **kw)
        per_ref_on = sum(m.multiplicity for m in on) / z_on
        per_ref_off = sum(m.multiplicity for m in off) / z_off
        assert per_ref_on == per_ref_off, f"detection changed the multiplicity for kwargs={kw}"


def test_generate_detection_uses_single_reference_path_ammonia():
    # With detection, ammonia collapses to Z'=1 (one reference), so the integer
    # matcher (not the float cloud congruence) handles it.
    crystal = load("ammonia")
    detected = crystal.detect_symmetry()
    assert len(detected.get_references()) == 1


def test_generate_benzene_regression_with_detection():
    crystal = load("benzene")
    assert len(generate.generate(crystal, 2, 5.0, use_origin=True)) == 3


def _p1_cif_with_shifted_origin(crystal, shift, path):
    """Write ``crystal`` as a P1 CIF with every atom moved by ``shift`` (fractional)."""
    from crystalatte.core.crystal import expand_cell

    positions, symbols, _ = expand_cell(
        crystal._asu.positions, crystal._asu.atoms, crystal.space_group.sym_ops
    )
    (a, b, c), (alpha, beta, gamma) = crystal.lattice_parameters
    lines = [
        "data_shifted",
        f"_cell_length_a {a:.6f}",
        f"_cell_length_b {b:.6f}",
        f"_cell_length_c {c:.6f}",
        f"_cell_angle_alpha {alpha:.6f}",
        f"_cell_angle_beta {beta:.6f}",
        f"_cell_angle_gamma {gamma:.6f}",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "x,y,z",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    for i, (sym, pos) in enumerate(zip(symbols, (positions + shift) % 1.0)):
        lines.append(f"{sym}{i + 1} {sym} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_detection_survives_an_origin_off_the_twelfths_grid(tmp_path):
    # succinic acid sits on an inversion centre (P2_1/c, Z' = 1/2). Written in P1 with an
    # arbitrary origin, the detected inversion's translation falls between twelfths; rounding
    # it used to overlay each molecule with a copy offset by about 0.1 A
    import crystalatte as cle

    original = load("succinic_acid")
    path = _p1_cif_with_shifted_origin(
        original, np.array([0.0137, 0.0291, 0.0423]), tmp_path / "shifted.cif"
    )
    shifted = next(cle.from_cif(path))

    detected = shifted.detect_symmetry()
    assert len(detected.space_group.sym_ops) == 4

    n_atoms = len(original.get_references()[0].symbols)
    dimers = cle.generate(shifted, N=2, R=6.0, verbose=0)
    assert {len(m.symbols) for d in dimers for m in d.monomers} == {n_atoms}
    assert len(dimers) == len(cle.generate(original, N=2, R=6.0, verbose=0))


def test_detection_keeps_declared_symmetry_when_origin_cannot_be_moved(tmp_path, monkeypatch):
    import spglib

    import crystalatte as cle

    original = load("succinic_acid")
    path = _p1_cif_with_shifted_origin(
        original, np.array([0.0137, 0.0291, 0.0423]), tmp_path / "shifted.cif"
    )
    shifted = next(cle.from_cif(path))

    def boom(*args, **kwargs):
        raise RuntimeError("no dataset")

    monkeypatch.setattr(spglib, "get_symmetry_dataset", boom)
    with pytest.warns(RuntimeWarning, match="twelfths grid"):
        assert shifted.detect_symmetry() is shifted


def _loosely_shifted_succinic_acid(tmp_path):
    import crystalatte as cle

    original = load("succinic_acid")
    path = _p1_cif_with_shifted_origin(
        original, np.array([0.001, 0.0015, 0.002]), tmp_path / "shifted.cif"
    )
    return original, next(cle.from_cif(path))


def test_loose_symprec_still_moves_an_off_grid_origin(tmp_path):
    # the rounding error here (~0.02 A) is below symprec = 0.05; the origin must move anyway
    # rather than the translation being rounded
    _, shifted = _loosely_shifted_succinic_acid(tmp_path)
    detected = shifted.detect_symmetry(symprec=0.05)
    assert len(detected.space_group.sym_ops) == 4
    assert not np.allclose(detected._asu.positions, shifted._asu.positions)


@pytest.mark.xfail(
    strict=True,
    reason="detected symmetry is applied to the declared (P1) ASU without reducing it; atoms "
    "that deviate from the detected symmetry by more than expand_cell's 1e-3 fractional merge "
    "tolerance but less than symprec expand into near-duplicate images",
)
def test_loose_symprec_keeps_monomers_whole(tmp_path):
    import crystalatte as cle

    original, shifted = _loosely_shifted_succinic_acid(tmp_path)
    n_atoms = len(original.get_references()[0].symbols)
    dimers = cle.generate(shifted, N=2, R=6.0, verbose=0, symprec=0.05)
    assert {len(m.symbols) for d in dimers for m in d.monomers} == {n_atoms}
