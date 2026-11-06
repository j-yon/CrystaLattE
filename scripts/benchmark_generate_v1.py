"""Time one v1 run from CIF input to unique N-mers of a single order and print it as JSON

Calls v1's own pipeline without the lower orders or the energy stage, so it does the same job as
``benchmark_generate.py``. Imports are not timed. Put the v1 package on ``PYTHONPATH``::

    PYTHONPATH=/path/to/v1 python scripts/benchmark_generate_v1.py /abs/path/benzene.cif 2 10
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import time

from crystalatte.crystalatte import build_nmer, cif_main, supercell2monomers

NMER_TYPES = {2: "dimers", 3: "trimers", 4: "tetramers", 5: "pentamers"}
NO_COM_CUTOFF = 1e6


def main() -> None:
    cif, N, R = os.path.abspath(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])

    with tempfile.TemporaryDirectory() as scratch:  # cif_main writes the supercell to a file
        os.chdir(scratch)
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            r_cut_monomer = cif_main(cif, "sc.xyz", 0, 0, 0, 0.0, R, True)
            nmers = supercell2monomers("sc.xyz", r_cut_monomer, 1.2, 0)
            build_nmer(nmers, len(nmers), NMER_TYPES[N], R, NO_COM_CUTOFF, "ChSEV", 0)
            elapsed = time.perf_counter() - t0
        os.chdir("/")

    found = [nmer for nmer in nmers.values() if len(nmer["monomers"]) == N]
    print(
        json.dumps(
            {
                "unique": len(found),
                "multiplicity": sum(nmer["replicas"] for nmer in found),
                "seconds": elapsed,
            }
        )
    )


if __name__ == "__main__":
    main()
