import crystalatte as cle

for crystal, reference in cle.from_cif("./data/cif/Ritonavir.cif"):
    multimers = cle.generate(crystal, reference, N=2, R=30.0)
    print(f"Returned {len(multimers)} multimers.")
    for i, multimer in enumerate(multimers):
        if i == 1:
            mol = multimer.to_qcel_molecule()
            exit()
        # mol.to_file(f"./data/test/ritonavir/{i}.xyz")
