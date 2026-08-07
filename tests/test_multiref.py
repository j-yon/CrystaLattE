import numpy as np
import crystalatte as cle
from pathlib import Path

DATA = Path(__file__).parent


def test_get_references_z1_benzene():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    refs = crystal.get_references()
    assert len(refs) == 1
    assert len(refs[0].symbols) == 12  # full benzene


def test_get_references_z4_ammonia_p1():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    assert len(refs) == 4                      # 4 independent NH3 in P1
    assert all(len(r.symbols) == 4 for r in refs)  # each NH3 = N + 3H
    # distinct molecules => distinct centroids
    cens = np.array([r.centroid_frac % 1.0 for r in refs])
    assert len({tuple(np.round(c, 3)) for c in cens}) == 4


def test_get_reference_is_first_of_references():
    crystal, _ = next(cle.from_cif(str(DATA / "x23/benzene/benzene.cif")))
    assert np.allclose(crystal.get_reference().centroid_frac,
                       crystal.get_references()[0].centroid_frac)


def test_symoplist_ref_indices_default_zeros():
    from crystalatte.core.sym_ops import SymOp, SymOpList
    s = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    assert list(s.ref_indices) == [0, 0]


def test_symoplist_ref_indices_explicit():
    from crystalatte.core.sym_ops import SymOp, SymOpList
    s = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    s.ref_indices = np.array([0, 1])
    assert list(s.ref_indices) == [0, 1]
    fc = SymOpList.from_components(
        s.rot_cache, s.tr_cache, 1, ref_indices=np.array([1, 0])
    )
    assert list(fc.ref_indices) == [1, 0]


def test_bind_globals_multi_reference_state():
    from crystalatte.work import generate
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    assert len(generate._monomers) == 4
    assert len(generate._Z_by_ref) == 4
    assert len(generate._W_lin_by_ref) == 4
    assert len(generate._check_congruence_by_ref) == 4
    # NH3 Z tags = [7,1,1,1] per reference
    assert list(generate._Z_by_ref[0]) == [7, 1, 1, 1]


def test_fingerprint_uses_per_member_reference():
    # A 2-member cluster whose members come from different NH3 references must
    # reconstruct atoms from each base reference; fingerprint is well-defined.
    from crystalatte.core.sym_ops import SymOp, SymOpList
    from crystalatte.work import generate
    crystal, _ = next(cle.from_cif(str(DATA / "x23/ammonia/ammonia.cif")))
    refs = crystal.get_references()
    generate._bind_globals(crystal, refs)
    g = SymOpList([SymOp.identity(), SymOp.identity()], 1)
    g.ref_indices = np.array([0, 1])   # member 0 <- ref 0, member 1 <- ref 1
    zz, d = generate._distance_fingerprint(g)
    assert len(zz) == len(d) > 0
