"""Compute a lattice energy with the many-body expansion, the way a user of the package would

Uses only the public API. ``--backend local`` runs every calculation on this machine
(needs ``crystalatte[local]`` and a QC program such as Psi4); ``--backend inputs`` only writes
the QC input files for you to run elsewhere.

Run from the repository root, e.g.::

    PYTHONPATH=src python scripts/run_lattice_energy.py tests/x23/benzene/benzene.cif \\
        --cutoff 2=10 --cutoff 3=8 --method mp2 --basis aug-cc-pvdz --out bz_run --ncores 8
    PYTHONPATH=src python scripts/run_lattice_energy.py tests/x23/ammonia/ammonia.cif \\
        --backend inputs --out ammonia_inputs
"""

import argparse

import crystalatte as cle


def parse_cutoff(text: str) -> tuple[int, float]:
    order, _, radius = text.partition("=")
    return int(order), float(radius)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cif", help="path to the CIF file")
    parser.add_argument(
        "--cutoff",
        type=parse_cutoff,
        action="append",
        metavar="N=R",
        help="multimer order and cutoff in Angstroms, repeatable (default 2=10 and 3=8)",
    )
    parser.add_argument("--backend", choices=("local", "inputs"), default="local")
    parser.add_argument("--out", required=True, help="job-state workdir (local) or input directory (inputs)")
    parser.add_argument(
        "--qc-scratch",
        default=None,
        help="scratch for the QC program (local); default leaves it to QCEngine, e.g. $PSI_SCRATCH",
    )
    parser.add_argument("--program", default="psi4")
    parser.add_argument("--method", default="mp2")
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument("--bsse", choices=("cp", "nocp", "vmfc"), default="cp")
    parser.add_argument("--ncores", type=int, default=None, help="total cores for the local backend")
    parser.add_argument("--disorder", choices=("error", "ignore", "drop"), default="error")
    parser.add_argument("-q", "--quiet", action="store_true", help="hide progress bars")
    args = parser.parse_args()

    crystal = next(cle.from_cif(args.cif, disorder=args.disorder))
    print(crystal)

    multimers = []
    for order, radius in args.cutoff or [(2, 10.0), (3, 8.0)]:
        found = cle.generate(crystal, order, radius, verbose=0 if args.quiet else 1)
        print(f"N={order}  R={radius} A: {len(found)} unique multimers")
        multimers += found

    spec = cle.ComputeSpec(
        program=args.program, method=args.method, basis=args.basis, bsse=args.bsse
    )

    if args.backend == "inputs":
        backend = cle.InputFileBackend(args.out)
        backend.submit(multimers, spec)
        print(f"wrote input files under {args.out}")
        return

    backend = cle.LocalBackend(args.out, qc_scratch=args.qc_scratch, ncores=args.ncores)
    backend.submit(multimers, spec)
    print(backend.status())
    results = backend.collect()  # raises IncompleteResultsError if anything is missing

    print(f"lattice energy: {results.lattice_energy:.4f} kJ/mol per formula unit")
    for order, energy in sorted(results.by_order.items()):
        print(f"  {order}-body: {energy:.4f} kJ/mol")


if __name__ == "__main__":
    main()
