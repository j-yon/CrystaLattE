from abc import ABC, abstractmethod
from pathlib import Path

from ...core.multimer import Monomer
from ...compute.spec import ComputeSpec


class QCWriter(ABC):
    """Abstract base class for writers of quantum chemistry input files"""

    @abstractmethod
    def write(
        self,
        path: Path,
        monomers: list[Monomer],
        spec: ComputeSpec,
        *,
        memory: str | None = None,
    ) -> None: ...

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def extension(cls) -> str: ...
