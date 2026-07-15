from abc import ABC, abstractmethod
from pathlib import Path

from ...core.multimer import Monomer


class QCWriter(ABC):
    @abstractmethod
    def write(self, path: Path, monomers: list[Monomer], config: dict) -> None: ...

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...
