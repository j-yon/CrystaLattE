import numpy as np
from numpy.typing import NDArray


def representatives(labels: NDArray, tnorm: NDArray) -> tuple[NDArray, NDArray]:
    """Fold every orbit label into one representative cluster

    The labels are exact, so every cluster sharing one is the same multimer. Its representative is the first
    cluster with the smallest summed translation norm, and representatives come out in the order their labels
    first appear.

    :param labels: ``(N,)`` int64 orbit labels numbered ``0..U-1``
    :param tnorm: ``(N,)`` summed translation norm of each cluster's members
    :returns: ``(rep, multiplicity)``, two ``(U,)`` int64 arrays: each representative's cluster index and the
        number of clusters it stands for
    """
    N = len(labels)
    U = int(labels.max()) + 1
    best = np.full(U, np.inf)
    np.minimum.at(best, labels, tnorm)
    hit = np.flatnonzero(tnorm == best[labels])
    rep = np.full(U, N, dtype=np.int64)
    np.minimum.at(rep, labels[hit], hit)
    first = np.full(U, N, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(N, dtype=np.int64))
    order = np.argsort(first)
    return rep[order], np.bincount(labels, minlength=U)[order]
