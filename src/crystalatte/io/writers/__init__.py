from .base import QCWriter
from .psi4 import Psi4Writer

_REGISTRY: dict[str, type[QCWriter]] = {
    "psi4": Psi4Writer,
}


def get_writer(name: str) -> QCWriter:
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown software '{name}'. Available: {list(_REGISTRY)}")
