import json
import re
from pathlib import Path

from ...core.multimer import Multimer, order_name
from ...io.writers import get_writer, has_writer
from ..results import Results, Status
from ..spec import ComputeSpec
from .base import Backend, crystalatte_version, manifest_entry

_GIB = {"b": 1 / 1024**3, "kb": 1e3 / 1024**3, "mb": 1e6 / 1024**3, "gb": 1e9 / 1024**3, "tb": 1e12 / 1024**3,
        "kib": 1 / 1024**2, "mib": 1 / 1024, "gib": 1.0, "tib": 1024.0}


def _to_gib(memory: str | float) -> float:
    if isinstance(memory, (int, float)):
        return float(memory)
    match = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]+)\s*", memory)
    if not match or match.group(2).lower() not in _GIB:
        raise ValueError(f"cannot read memory {memory!r}; use GiB as a number or a string like '8 GB'")
    return float(match.group(1)) * _GIB[match.group(2).lower()]


class InputFileBackend(Backend):
    """Write native input files for the user to run, plus a ``manifest.json`` describing them

    Psi4 gets one file per multimer (its own many-body driver handles counterpoise). Any other
    program goes through QCEngine's input builder: one input per QCManyBody sub-calculation,
    in ``<order>s/<multimer id>/real-..._basis-.../``. The manifest records the spec, the
    generation settings and, per multimer, its order, multiplicity, lattice weight and files;
    for sub-calculations also the QCManyBody label, real and basis (ghost) fragments and the
    command QCEngine would run. Nothing is read back: run the files yourself and assemble the
    results, e.g. with ``qcmanybody.ManyBodyCore.analyze`` keyed by the recorded labels.

    :param output_dir: directory for the inputs and the manifest
    :param memory: memory per calculation, as GiB or a string like ``"8 GB"``; for QCEngine-built
        inputs, ``None`` uses the memory QCEngine detects on the machine writing the files
    """

    def __init__(self, output_dir: str | Path, *, memory: str | float | None = None):
        self.output_dir = Path(output_dir)
        self.memory = memory

    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None:
        by_order = self._validate(multimers)
        manifest = self._start_manifest(multimers, spec)

        if has_writer(spec.program):
            entries = self._write_whole_multimers(by_order, spec)
        else:
            entries = self._write_subcalculations(by_order, spec)

        for N, group in by_order.items():
            manifest["generation"][str(N)] = group[0].generation.model_dump(mode="json")
            manifest["active"][str(N)] = [m.id for m in group]
            for m in group:
                manifest["multimers"][m.id] = {**manifest_entry(m), **entries[m.id]}
        manifest["crystalatte_version"] = crystalatte_version()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def _start_manifest(self, multimers: list[Multimer], spec: ComputeSpec) -> dict:
        crystal_fp = multimers[0].generation.crystal_fingerprint
        path = self.output_dir / "manifest.json"
        if not path.exists():
            return {
                "program": spec.program,
                "spec": spec.model_dump(mode="json"),
                "spec_fingerprint": spec.fingerprint(),
                "crystal_fingerprint": crystal_fp,
                "generation": {},
                "active": {},
                "multimers": {},
            }
        manifest = json.loads(path.read_text())
        if manifest["spec_fingerprint"] != spec.fingerprint():
            raise ValueError(
                f"spec differs from the one used in {self.output_dir} (spec fingerprint "
                f"{manifest['spec_fingerprint'][:12]} vs {spec.fingerprint()[:12]}); use a new output_dir"
            )
        if manifest["crystal_fingerprint"] != crystal_fp:
            raise ValueError(
                f"crystal fingerprint differs from the one used in {self.output_dir} "
                f"({manifest['crystal_fingerprint'][:12]} vs {crystal_fp[:12]}); use a new output_dir"
            )
        return manifest

    def _write_whole_multimers(self, by_order: dict[int, list[Multimer]], spec: ComputeSpec) -> dict[str, dict]:
        writer = get_writer(spec.program)
        memory = f"{self.memory} GiB" if isinstance(self.memory, (int, float)) else self.memory
        entries = {}
        for N, group in by_order.items():
            order_dir = self.output_dir / f"{order_name(N)}s"
            order_dir.mkdir(parents=True, exist_ok=True)
            for m in group:
                path = order_dir / f"{m.id}.{writer.extension()}"
                writer.write(path, m.monomers, spec, memory=memory)
                entries[m.id] = {"files": [self._relative(path)]}
        return entries

    def _write_subcalculations(self, by_order: dict[int, list[Multimer]], spec: ComputeSpec) -> dict[str, dict]:
        try:
            from ...io.writers.qcengine import QCEngineWriter
        except ImportError as exc:  # pragma: no cover - exercised only without the extras
            raise ImportError(
                f"writing {spec.program} inputs needs QCEngine and QCManyBody: pip install 'crystalatte[local]'"
            ) from exc
        writer = QCEngineWriter(spec.program)
        planned = {m.id: writer.plan(m, spec) for group in by_order.values() for m in group}
        for tasks in planned.values():
            writer.check(tasks)  # validate everything before writing anything

        memory_gib = None if self.memory is None else _to_gib(self.memory)
        entries = {}
        for N, group in by_order.items():
            for m in group:
                directory = self.output_dir / f"{order_name(N)}s" / m.id
                tasks = writer.write(directory, planned[m.id], spec, memory_gib=memory_gib)
                for task in tasks:
                    task["dir"] = self._relative(task["dir"])
                    task["infiles"] = [self._relative(p) for p in task["infiles"]]
                entries[m.id] = {"tasks": tasks}
        return entries

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.output_dir).as_posix()

    def status(self) -> Status:
        raise NotImplementedError(_MSG)

    def collect(self, *, partial: bool = False) -> Results:
        raise NotImplementedError(_MSG)


_MSG = "InputFileBackend only writes input files; run them yourself or use LocalBackend/FractalBackend"
