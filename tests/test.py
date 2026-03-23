import crystalatte as cle
from time import time


if __name__ == "__main__":
    for crystal, reference in cle.from_cif("./data/cif/Ritonavir.cif"):
        print("Generating multimers...")
        start = time()
        multimers = cle.generate(crystal, reference, N=4, R=30.0)
        end = time()
        print(f"Returned {len(multimers)} multimers in {end - start:.2f} seconds")
        # for i, multimer in enumerate(multimers):
        #     mol = multimer.to_molecule()
        #     mol.to_file(f"./data/test/ritonavir/{i}.xyz")
