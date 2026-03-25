import crystalatte as cle
from time import time


if __name__ == "__main__":
    kwargs = {"use_com": False, "n_jobs": 16}
    for crystal, reference in cle.from_cif("./data/cif/Ritonavir.cif"):
        print("Generating multimers...")
        start = time()
        multimers = cle.generate(crystal, reference, N=2, R=10.0, **kwargs)
        end = time()
        print(f"Returned {len(multimers)} multimers in {end - start:.2f} seconds")
        # for i, multimer in enumerate(multimers):
        #     mol = multimer.to_molecule()
        #     mol.to_file(f"./data/test/ritonavir/{i}.xyz")
