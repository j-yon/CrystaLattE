from ..core.crystal import Crystal
from ..core.multimer import Monomer, Multimer


def generate(crystal: Crystal, monomer: Monomer, N: int, R: float) -> list[Multimer]:
    """
    Generate unique multimers of the given monomer in the crystal.

    Parameters
    ----------
    crystal : Crystal
        The crystal structure containing the monomer.
    monomer : Monomer
        The reference monomer to generate multimers from.
    N : int
        The number of monomers in the multimer (e.g. N=2 for dimers).
    R : float
        The maximum center-to-center distance in Angstroms for monomers to be considered part of the same multimer.

    Returns
    -------
    List[Multimer]
        A list of unique multimers generated from the reference monomer.
    """

    pass
