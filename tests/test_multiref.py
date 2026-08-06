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
