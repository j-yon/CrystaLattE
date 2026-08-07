import pickle
import numpy as np
import crystalatte as cle
from pathlib import Path
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp

DATA = Path(__file__).parent


def test_from_sym_ops_holds_explicit_ops():
    ops = [SymOp.identity(), SymOp("-x,-y,-z")]
    sg = SpaceGroup.from_sym_ops(ops, number=2)
    assert len(sg.sym_ops) == 2
    assert sg.number == 2
    # not cached in the registry (detected groups are per-structure)
    assert SpaceGroup(2) is not sg


def test_from_sym_ops_pickles_with_its_ops():
    # detected groups travel to ProcessPool workers via pickle; they must keep
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
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    assert len(crystal.space_group.sym_ops) == 1          # declared P1
    detected = crystal.detect_symmetry()
    assert len(detected.space_group.sym_ops) == 12        # P2_13
    assert detected.space_group.number == 198
    # symmetry now relates the 4 NH3 -> a single reference
    assert len(detected.get_references()) == 1


def test_detect_symmetry_benzene_no_change():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    detected = crystal.detect_symmetry()
    # benzene's declared SG 61 is already correct -> not strictly higher -> self
    assert detected is crystal


def test_detect_symmetry_graceful_on_failure(monkeypatch):
    import spglib
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    monkeypatch.setattr(spglib, "get_symmetry", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert crystal.detect_symmetry() is crystal            # falls back, no raise


def test_detect_symmetry_benzene_full_cell_expansion():
    # Benzene is non-P1 (SG 61, Z=4); detection must expand ASU to full cell
    # before passing to spglib. Verify it works (returns self, no raise/warning).
    import warnings
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        detected = crystal.detect_symmetry()
        # No symmetry-detection warnings (silenced spglib + handled errors)
        sym_warnings = [x for x in w if "symmetry detection" in str(x.message)]
        assert len(sym_warnings) == 0
    assert detected is crystal  # benzene already at full symmetry


from crystalatte.work import generate


def test_generate_detection_invariant_ammonia():
    # Detection changes HOW (fast integer path), not WHAT (the unique n-mer set).
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    for kw in (dict(use_origin=True), dict()):  # use_origin and default conventions
        n_on = len(generate.generate(crystal, 2, 5.0, n_jobs=1, detect_symmetry=True, **kw))
        n_off = len(generate.generate(crystal, 2, 5.0, n_jobs=1, detect_symmetry=False, **kw))
        assert n_on == n_off, f"detection changed the count for kwargs={kw}: {n_on} != {n_off}"


def test_generate_detection_uses_single_reference_path_ammonia():
    # With detection, ammonia collapses to Z'=1 (one reference), so the integer
    # matcher (not the float cloud congruence) handles it.
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    detected = crystal.detect_symmetry()
    assert len(detected.get_references()) == 1


def test_generate_benzene_regression_with_detection():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    assert len(generate.generate(crystal, 2, 5.0, use_origin=True, n_jobs=1)) == 3
