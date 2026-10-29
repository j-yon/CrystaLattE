"""Generate unique dimers from a CIF with CrystaLattE v2 (no QC input files)."""
import argparse
import time

import crystalatte as cle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif", help="path to the CIF file")
    parser.add_argument("-r", "--r-cut", type=float, default=10.0,
                        help="dimer cutoff in Angstrom (default: 10.0)")
    parser.add_argument("--com", action="store_true",
                        help="cut on centroid distance instead of closest atom-atom contact")
    parser.add_argument("--disorder", default="error", choices=["error", "ignore", "drop"],
                        help="policy for partially occupied sites (default: error)")
    parser.add_argument("--xyz", action="store_true",
                        help="also print each dimer's geometry in XYZ format")
    args = parser.parse_args()

    # from_cif yields one Crystal per CIF data block
    for block, crystal in enumerate(cle.from_cif(args.cif, disorder=args.disorder)):
        start = time.time()
        dimers = cle.generate(crystal, N=3, R=args.r_cut, use_com=args.com)
        dimers.sort(key=lambda d: d.geometric_mean)

        cut = "centroid distance" if args.com else "closest contact"
        print(f"Block {block}: {len(dimers)} unique multimers ({cut} <= {args.r_cut} A)")
        print(f"Time taken: {time.time() - start:.2f} seconds")
        # print(f"{'#':>4}  {'R_centroid (A)':>14}  {'mult':>5}  {'atoms':>6}")
        # for i, dimer in enumerate(dimers, 1):
        #     n_atoms = sum(len(m.symbols) for m in dimer.monomers)
        #     print(f"{i:>4}  {dimer.geometric_mean:>14.4f}  {dimer.multiplicity:>5}  {n_atoms:>6}")

        # if args.xyz:
        #     for i, dimer in enumerate(dimers, 1):
        #         print(dimer.to_molecule().to_string("xyz"))


if __name__ == "__main__":
    main()


"""MANUAL VALIDATION AGAINST X23
v1, v2

1,4-cyclohexanedione: 
    30 A dimers: 596, 596
    60 A dimers: 3926, 3926
acetic acid:
    30 A dimers: 937, 937
    60 A dimers: 6595, 6595
    15 A trimers: 6253, 6252
    20 A trimers: 30546, 30546
adamantane:
    30 A dimers: 93, 128 *
    60 A dimers: 756, 756
ammonia:
    30 A dimers: 627, 627
    60 A dimers: -, 4744
anthracene:
    60 A dimers: 1973, 1973
benzene:
    30 A dimers: 420, 420
    60 A dimers: 2841, 2841
    15 A trimers: 1977, 1977
    10 A tetramers: 1332, 1332
carbon dioxide:
    30 A dimers: 251, 284 *
    60 A dimers: 1775, 2114 *
    15 A trimers: 1948, 1985 *
cyanamide:
    30 A dimers: 1465, 1465
    60 A dimers: 10733, 10734 *
cytosine:
    30 A dimers: 692, 692
    60 A dimers: 4624, 4624
ethyl carbamate:
    30 A dimers: 980, 980
    60 A dimers: 6594, 6594
formamide:
    30 A dimers: 1535, 1535
    60 A dimers: 11145, 11146 *
    20 A trimers: 48046, 48064 *
hexamine:
    30 A dimers: 40, 40
    60 A dimers: 206, 206
ice:
    30 A dimers: 1484, error (disordered) *
    60 A dimers: 11153, error (disordered) *
imidazole:
    30 A dimers: 1071, 1071
    60 A dimers: 7495, 7495
    15 A trimers: 5572, 5571 *
    20 A trimers: 26039, 26039
naphthalene:
    30 A dimers: 317, 388 *
    60 A dimers: 2504, 2504
oxalic acid alpha:
    30 A dimers: 597, 597
    60 A dimers: 4167, 4167
oxalic acid beta:
    30 A dimers: 597, 597
    60 A dimers: 4987, 4988 *
pyrazine:
    30 A dimers: 319, 319
    60 A dimers: 2070, 2070
pyrazole:
    30 A dimers: 1271, 1692 *
    60 A dimers: 8940, 11920 *
succinic acid:
    30 A dimers: 525, 525
    60 A dimers: 3641, 3641
triazine:
    30 A dimers: 199, 199
    60 A dimers: 1246, 1246
    15 A trimers: 750, 750
trioxane:
    30 A dimers: 257, 257
    60 A dimers: 1740, 1740
uracil:
    30 A dimers: 859, 859
    60 A dimers: 5800, 5800
urea:
    30 A dimers: 311, 311
    60 A dimers: 1988, 1988
"""
