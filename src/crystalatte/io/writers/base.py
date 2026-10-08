from abc import ABC, abstractmethod
from pathlib import Path

from ...core.multimer import Monomer


class QCWriter(ABC):
    """Abstract base class for writers of quantum chemistry input files"""

    @abstractmethod
    def write(self, path: Path, monomers: list[Monomer], config: dict) -> None: ...

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...
