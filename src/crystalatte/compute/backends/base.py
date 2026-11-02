from abc import ABC, abstractmethod
from collections import defaultdict

from ...core.multimer import Multimer
from ..results import Results, Status
from ..spec import ComputeSpec


class Backend(ABC):
    """Common interface for everything that can run a set of multimers"""

    @abstractmethod
    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None: ...

    @abstractmethod
    def status(self) -> Status: ...

    @abstractmethod
    def collect(self, *, partial: bool = False) -> Results: ...

    @staticmethod
    def _validate(multimers: list[Multimer]) -> dict[int, list[Multimer]]:
        """Check that multimers are submittable and group them by order

        :param multimers: multimers returned by :func:`crystalatte.generate`
        :returns: multimers grouped by number of monomers
        :raises ValueError: if empty, hand-built, duplicated, or drawn from mixed runs or crystals
        """
        if not multimers:
            raise ValueError("no multimers to submit")
        if any(m.generation is None for m in multimers):
            raise ValueError(
                "multimers must come from crystalatte.generate; "
                "hand-built multimers carry no generation info"
            )

        seen: set[str] = set()
        for m in multimers:
            if m.id in seen:
                raise ValueError(f"duplicate multimer id {m.id}")
            seen.add(m.id)

        by_order: dict[int, list[Multimer]] = defaultdict(list)
        for m in multimers:
            by_order[len(m.monomers)].append(m)

        if len({m.generation.crystal_fingerprint for m in multimers}) > 1:
            raise ValueError("multimers come from different crystals (crystal fingerprint differs)")
        for N, group in by_order.items():
            if any(m.generation != group[0].generation for m in group):
                raise ValueError(
                    f"{N}-mers come from different generate() runs (generation info differs)"
                )
        return dict(sorted(by_order.items()))
