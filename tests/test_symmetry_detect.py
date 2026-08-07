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
