"""Time one run from CIF input to returned multimers and print it as JSON

Run in a fresh process per measurement; ``benchmark_driver.py`` handles this. Imports are not timed.

    PYTHONPATH=src python scripts/benchmark_generate.py tests/x23/benzene/benzene.cif 2 10
"""

import json
import sys
import time

import crystalatte as cle


def main() -> None:
    cif, N, R = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])

    t0 = time.perf_counter()
    crystal = next(cle.from_cif(cif))
    multimers = cle.generate(crystal, N, R, verbose=0)
    elapsed = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "unique": len(multimers),
                "multiplicity": sum(m.multiplicity for m in multimers),
                "seconds": elapsed,
            }
        )
    )


if __name__ == "__main__":
    main()
