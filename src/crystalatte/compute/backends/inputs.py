from pathlib import Path

from ...core.multimer import Multimer, order_name
from ...io.writers import get_writer
from ..results import Results, Status
from ..spec import ComputeSpec
from .base import Backend


class InputFileBackend(Backend):
    """Write one input file per multimer, named by multimer id, for the user to run"""

    def __init__(self, output_dir: str | Path, *, memory: str | None = None):
        self.output_dir = Path(output_dir)
        self.memory = memory

    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None:
        by_order = self._validate(multimers)
        writer = get_writer(spec.program)
        for N, group in by_order.items():
            order_dir = self.output_dir / f"{order_name(N)}s"
            order_dir.mkdir(parents=True, exist_ok=True)
            for m in group:
                path = order_dir / f"{m.id}.{writer.extension()}"
                writer.write(path, m.monomers, spec, memory=self.memory)

    def status(self) -> Status:
        raise NotImplementedError(_MSG)

    def collect(self, *, partial: bool = False) -> Results:
        raise NotImplementedError(_MSG)


_MSG = "InputFileBackend only writes input files; run them yourself or use LocalBackend/FractalBackend"
