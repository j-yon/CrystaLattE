from itertools import product, combinations_with_replacement
import numpy as np

import crystalatte as cle


for crystal, reference in cle.from_cif("./data/cif/Benzene.cif"):
    multimers = cle.generate(crystal, reference, N=2, R=15.0)
    print(multimers)
