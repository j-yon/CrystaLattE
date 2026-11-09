from .base import QCWriter
from .psi4 import Psi4Writer

_REGISTRY: dict[str, type[QCWriter]] = {
    "psi4": Psi4Writer,
}


def has_writer(name: str) -> bool:
    """Whether ``name`` has a dedicated whole-multimer writer (others go through QCEngine)"""
    return name.lower() in _REGISTRY


def get_writer(name: str) -> QCWriter:
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise ValueError(f"Unknown software '{name}'. Available: {list(_REGISTRY)}")
