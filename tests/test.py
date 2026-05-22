import crystalatte as cle
from time import time
import itertools


if __name__ == "__main__":
    kwargs = {"use_com": False, "n_jobs": 8, "verbose": 0}

    for crystal, reference in cle.from_cif("./data/cif/Benzene.cif"):
        for N, R in zip([2, 3, 4], [15.0, 15.0, 10.0]):
            # returns a list of "Multimers"
            print(f"Generating multimers for {R}...")
            start = time()
            multimers = cle.generate(crystal, reference, N=N, R=R, **kwargs)
            end = time()
            print(f"Returned {len(multimers)} multimers in {end - start:.2f} seconds\n")

            # output the multimers as .xyz files
            nres = []
            for i, multimer in enumerate(multimers):
                mol = multimer.to_molecule()
                nres.append(mol.nuclear_repulsion_energy())
                mol.to_file(f"./data/test/benzene/{N}mer_{i}.xyz")

            # check for identical multimers
            EPSILON = 1e-9
            matches = [
                (i, j)
                for (i, j) in itertools.combinations(range(len(nres)), 2)
                if abs(nres[i] - nres[j]) < EPSILON
            ]
            print(len(matches))
            for match in matches:
                print(
                    f"Multimers {match[0]} and {match[1]} have identical nuclear repulsion energies: {nres[match[0]]:.6f} Hartree"
                )
                print(f"Multimer {match[0]}: {multimers[match[0]]}")
                print(f"Multimer {match[1]}: {multimers[match[1]]}")

            break
