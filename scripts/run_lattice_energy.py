
import time

from qcportal import PortalClient

import crystalatte as cle


def main() -> None:
    crystal = next(cle.from_cif("/path/to/benzene.cif", disorder="enumerate"))

    multimers = []
    for N, R in [(2, 30.0), (3, 15.0), (4, 10.0)]:
        multimers += cle.generate(crystal, N=N, R=R, verbose=1)

    spec = cle.ComputeSpec(
        program="psi4",
        method="DLPNO-CCSD(T)",
        basis="jun-cc-pvdz",
        bsse="cp",
    )

    client = PortalClient("http://localhost:7777", username=None, password=None)
    backend = cle.FractalBackend(client, "benzene-mbe")
    backend.submit(multimers, spec)

    while any(counts["pending"] for counts in backend.status().counts.values()):
        print(backend.status())
        time.sleep(60)

    results = backend.collect()

    print(f"lattice energy: {results.lattice_energy:.4f} kJ/mol per formula unit")
    for order, energy in sorted(results.by_order.items()):
        print(f"\t{order}-body: {energy:.4f} kJ/mol")


if __name__ == "__main__":
    main()

