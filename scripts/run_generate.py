"""Read a CIF and generate its unique multimers, the way a user of the package would

Uses only the public API (``crystalatte.from_cif`` and ``crystalatte.generate``).

Run from the repository root, e.g.::

    PYTHONPATH=src python scripts/run_generate.py tests/x23/benzene/benzene.cif -N 2 -R 10
    PYTHONPATH=src python scripts/run_generate.py my.cif -N 3 -R 8 --mode origin --list -q
"""

import argparse
import time
from collections import Counter

import crystalatte as cle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cif", help="path to the CIF file")
    parser.add_argument("-N", type=int, default=2, help="multimer order (default 2)")
    parser.add_argument("-R", type=float, default=10.0, help="cutoff in Angstroms (default 10)")
    parser.add_argument(
        "--mode",
        choices=("contact", "origin", "com"),
        default="contact",
        help="distance convention: closest atom pair (default), closest atom to the anchor "
             "centroid, or centre of mass",
    )
    parser.add_argument(
        "--disorder",
        choices=("error", "ignore", "drop"),
        default="error",
        help="policy for partially occupied sites (default error)",
    )
    parser.add_argument("--symprec", type=float, default=1e-3, help="spglib tolerance (default 1e-3)")
    parser.add_argument("--no-detect", action="store_true", help="keep the declared space group")
    parser.add_argument("--list", action="store_true", help="print every multimer")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide progress bars")
    args = parser.parse_args()

    # a CIF may hold several data blocks; take the first
    block = next(cle.from_cif(args.cif, disorder=args.disorder))
    crystal = block[0] if isinstance(block, tuple) else block  # older versions yield (Crystal, Monomer)
    print(crystal)

    kwargs = {
        "use_origin": args.mode == "origin",
        "use_com": args.mode == "com",
        "verbose": 0 if args.quiet else 1,
    }
    t0 = time.perf_counter()
    multimers = cle.generate(
        crystal,
        args.N,
        args.R,
        detect_symmetry=not args.no_detect,
        symprec=args.symprec,
        **kwargs,
    )
    elapsed = time.perf_counter() - t0

    histogram = dict(sorted(Counter(m.multiplicity for m in multimers).items()))
    print(f"N={args.N}  R={args.R} A  mode={args.mode}")
    print(f"unique multimers:   {len(multimers)}")
    print(f"total multiplicity: {sum(m.multiplicity for m in multimers)}")
    print(f"multiplicities:     {histogram}")
    print(f"time:               {elapsed:.2f} s")

    if args.list:
        print(f"{'#':>5s}  {'multiplicity':>12s}  {'geometric mean (A)':>18s}")
        for i, m in enumerate(sorted(multimers, key=lambda m: m.geometric_mean), 1):
            print(f"{i:5d}  {m.multiplicity:12d}  {m.geometric_mean:18.4f}")


if __name__ == "__main__":
    main()
