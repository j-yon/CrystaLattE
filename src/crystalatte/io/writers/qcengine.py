import json
import warnings
from pathlib import Path

from ...compute.spec import ComputeSpec
from ...core.multimer import Multimer

# programs whose QCEngine-built inputs were checked to keep ghost atoms' basis functions
# (NWChem: bq atoms; Molpro: dummy centres). Q-Chem's drops them, so it is not listed.
GHOST_VERIFIED = frozenset({"nwchem", "molpro"})


class QCEngineWriter:
    """Write native inputs for any program whose QCEngine harness implements ``build_input``

    QCEngine builds single-point inputs only, so a multimer becomes one input per QCManyBody
    sub-calculation, each in its own directory (harnesses use fixed file names).
    """

    def __init__(self, program: str):
        import qcengine
        from qcengine.programs.base import get_program
        from qcengine.programs.model import ProgramHarness

        self.program = program.lower()
        try:
            self._harness = get_program(self.program, check=False)
        except Exception as exc:
            raise ValueError(f"QCEngine does not know program {program!r}") from exc
        if type(self._harness).build_input is ProgramHarness.build_input:
            raise ValueError(
                f"QCEngine cannot write {self.program} input files; run it through LocalBackend instead"
            )
        self._qcengine = qcengine

    def plan(self, multimer: Multimer, spec: ComputeSpec) -> list[tuple[str, list[int], list[int], object]]:
        """Return ``(label, real, basis, molecule)`` for every sub-calculation of ``multimer``"""
        from qcmanybody import ManyBodyCore

        core = ManyBodyCore(multimer.to_molecule(), **spec.manybody_kwargs(len(multimer.monomers)))
        tasks = []
        for _, label, molecule in core.iterate_molecules():
            _, real, basis = json.loads(label)
            tasks.append((label, real, basis, molecule))
        return tasks

    def check(self, tasks: list[tuple[str, list[int], list[int], object]]) -> None:
        """Refuse ghost-atom sub-calculations for programs whose inputs aren't verified to keep them"""
        if self.program in GHOST_VERIFIED:
            return
        if any(len(basis) > len(real) for _, real, basis, _ in tasks):
            raise ValueError(
                f"{self.program} inputs built by QCEngine are not verified to keep ghost atoms, which "
                f"counterpoise sub-calculations need (verified: {sorted(GHOST_VERIFIED)}, or psi4); "
                f"use bsse='nocp', another program, or LocalBackend"
            )

    def write(
        self,
        directory: Path,
        tasks: list[tuple[str, list[int], list[int], object]],
        spec: ComputeSpec,
        *,
        memory_gib: float | None = None,
    ) -> list[dict]:
        """Write each sub-calculation under ``directory`` and return its manifest entries"""
        task_config = {"ncores": 1}
        if memory_gib is not None:
            task_config["memory"] = memory_gib
        config = self._qcengine.get_config(task_config=task_config)

        entries = []
        for label, real, basis, molecule in tasks:
            atomic_input = {**spec.atomic_template(), "molecule": json.loads(molecule.json())}
            try:
                with warnings.catch_warnings():
                    # harness internals (e.g. Molpro's) still call deprecated pydantic APIs
                    warnings.simplefilter("ignore", DeprecationWarning)
                    record = self._harness.build_input(self._harness.build_input_model(atomic_input), config)
            except Exception as exc:
                raise ValueError(
                    f"QCEngine could not build a {self.program} input ({type(exc).__name__}: {exc}); "
                    f"some harnesses need the program installed where inputs are generated, and programs "
                    f"without input files run through LocalBackend instead"
                ) from exc

            task_dir = directory / ("real-" + "-".join(map(str, real)) + "_basis-" + "-".join(map(str, basis)))
            task_dir.mkdir(parents=True, exist_ok=True)
            for name, text in record["infiles"].items():
                (task_dir / name).write_text(text)
            command = record.get("commands") or record.get("command")
            entries.append(
                {
                    "label": label,
                    "real": real,
                    "basis": basis,
                    "dir": task_dir,
                    "infiles": [task_dir / name for name in record["infiles"]],
                    "command": [c for c in command if c is not None] if command else None,
                }
            )
        return entries
