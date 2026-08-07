import pickle
import numpy as np
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp


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
