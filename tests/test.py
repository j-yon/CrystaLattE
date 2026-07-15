"""Smoke test + chemistry-validation harness.

For each (N, R) the deduper is run and its output is validated against the
original CrystaLattE NRE+CHSEV oracle at the same 1e-8 tolerance used by the
old code. Reports the multimer count and the number of distinct chemistry
classes; the two should agree (no duplicate left in the output).
"""

from time import time
import crystalatte as cle


if __name__ == "__main__":
    kwargs = {"use_origin": True, "n_jobs": 8, "verbose": 0}

    for crystal, reference in cle.from_cif("./data/cif/Benzene.cif"):
        for N, R in zip([2, 3, 4], [100.0, 30.0, 20.0]):
            print(f"--- N={N}, R={R} ---")
            start = time()
            multimers = cle.generate(crystal, reference, N=N, R=R, **kwargs)
            print(
                f"  generate: {len(multimers)} unique multimers in "
                f"{time() - start:.2f} s"
            )

            start = time()
            n_classes, _ = cle.count_chemistry_classes(multimers, atol=1e-8)
            print(
                f"  validation (NRE+CHSEV @ 1e-8): {n_classes} chemistry classes "
                f"in {time() - start:.2f} s"
            )
            if n_classes != len(multimers):
                print(
                    f"  WARNING: dedup count {len(multimers)} != chemistry-class "
                    f"count {n_classes} — duplicates remain in the output."
                )
            break  # only run N=2 by default; remove this `break` to run N=3, N=4.
        break
