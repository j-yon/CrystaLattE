"""Single-machine backend: QCManyBody task enumeration, QCEngine execution, scratch-dir checkpoints."""

import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ...core.multimer import Multimer
from ..results import Results, Row, Status, assemble, select_active
from ..spec import ComputeSpec
from .base import Backend, crystalatte_version

_INSTALL_HINT = "LocalBackend needs qcengine and qcmanybody; install them with: pip install 'crystalatte[local]'"
_RESULT = "result.json"


def _require_extras() -> None:
    try:
        import qcengine  # noqa: F401
        import qcmanybody  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _run_task(atomic_input: dict, program: str, task_config: dict) -> dict:
    """Run one single-point with QCEngine; never raises for a failed calculation"""
    import qcengine

    out = qcengine.compute(atomic_input, program, raise_error=False, task_config=task_config, return_dict=True)
    if out.get("success"):
        return {"success": True, "energy": float(out["return_result"]), "error": None}
    error = out.get("error") or {}
    message = error.get("error_message") if isinstance(error, dict) else str(error)
    return {"success": False, "energy": None, "error": message or "unknown error"}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _task_name(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()[:16] + ".json"


class _Job:
    """Many-body bookkeeping for one multimer during a submit"""

    def __init__(self, multimer: Multimer, spec: ComputeSpec, task_dir: Path):
        from qcmanybody import ManyBodyCore

        self.id = multimer.id
        self.N = len(multimer.monomers)
        self.task_dir = task_dir
        self.core = ManyBodyCore(multimer.to_molecule(), **spec.manybody_kwargs(self.N))
        self.key = f"{spec.bsse}_corrected_{self.N}_body_contribution_to_energy"
        self.energies: dict[str, float] = {}
        self.todo: list[tuple[str, dict]] = []
        for _, label, mol in self.core.iterate_molecules():
            path = task_dir / _task_name(label)
            if path.exists() and (done := _read_json(path))["success"]:
                self.energies[label] = done["energy"]
            else:
                self.todo.append((label, {**spec.atomic_template(), "molecule": json.loads(mol.json())}))
        self.remaining = len(self.todo)
        self.failed = False

    def record(self, label: str, outcome: dict) -> None:
        _write_json(self.task_dir / _task_name(label), {"label": label, **outcome})
        self.remaining -= 1
        if outcome["success"]:
            self.energies[label] = outcome["energy"]
        else:
            self.failed = True
        if self.remaining == 0 and not self.failed:
            self.finish()

    def finish(self) -> None:
        """Assemble the many-body energy; an analysis error is recorded, never raised, and retried on resubmit"""
        try:
            results = self.core.analyze({label: {"energy": e} for label, e in self.energies.items()})
            outcome = {"success": True, "energy": float(results["results"][self.key])}
        except Exception as exc:
            outcome = {"success": False, "error": f"many-body analysis failed: {type(exc).__name__}: {exc}"}
        _write_json(self.task_dir / _RESULT, outcome)


class LocalBackend(Backend):
    """Run many-body calculations on this machine, checkpointing every task to ``scratch``

    :param scratch: directory holding the manifest and per-task results; resubmitting into
        it skips finished tasks and retries failed ones
    :param ncores: cores to use in total; defaults to the cores QCEngine detects
    :param ncores_per_task: cores given to each single-point calculation
    :param memory_per_task: memory in GiB for each calculation; defaults to the memory QCEngine
        detects divided by the number of concurrent tasks
    """

    def __init__(
        self,
        scratch: str | Path,
        *,
        ncores: int | None = None,
        ncores_per_task: int = 1,
        memory_per_task: float | None = None,
    ):
        _require_extras()
        self.scratch = Path(scratch)
        if ncores is None or memory_per_task is None:
            import qcengine

            config = qcengine.get_config()
            ncores = config.ncores if ncores is None else ncores
        self.ncores = ncores
        self.ncores_per_task = ncores_per_task
        self.concurrency = max(1, ncores // ncores_per_task)
        if memory_per_task is None:
            memory_per_task = config.memory / self.concurrency
        self.memory_per_task = memory_per_task

    @property
    def _manifest_path(self) -> Path:
        return self.scratch / "manifest.json"

    def _task_dir(self, multimer_id: str) -> Path:
        return self.scratch / "tasks" / multimer_id

    def _load_manifest(self) -> dict | None:
        path = self._manifest_path
        return _read_json(path) if path.exists() else None

    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None:
        by_order = self._validate(multimers)
        crystal_fp = multimers[0].generation.crystal_fingerprint
        manifest = self._load_manifest()
        if manifest is None:
            manifest = {
                "spec": spec.model_dump(mode="json"),
                "spec_fingerprint": spec.fingerprint(),
                "crystal_fingerprint": crystal_fp,
                "generation": {},
                "active": {},
                "multimers": {},
            }
        elif manifest["spec_fingerprint"] != spec.fingerprint():
            raise ValueError(
                f"spec differs from the one used in {self.scratch} (spec fingerprint "
                f"{manifest['spec_fingerprint'][:12]} vs {spec.fingerprint()[:12]}); use a new scratch directory"
            )
        elif manifest["crystal_fingerprint"] != crystal_fp:
            raise ValueError(
                f"crystal fingerprint differs from the one used in {self.scratch} "
                f"({manifest['crystal_fingerprint'][:12]} vs {crystal_fp[:12]}); use a new scratch directory"
            )

        for N, group in by_order.items():
            manifest["generation"][str(N)] = group[0].generation.model_dump(mode="json")
            manifest.setdefault("crystalatte_version", {})[str(N)] = crystalatte_version()
            manifest["active"][str(N)] = [m.id for m in group]
            for m in group:
                manifest["multimers"][m.id] = {
                    "N": N,
                    "multiplicity": m.multiplicity,
                    "weight": m.weight,
                    "molecule": json.loads(m.to_molecule().json()),
                }
        self.scratch.mkdir(parents=True, exist_ok=True)
        _write_json(self._manifest_path, manifest)

        jobs = []
        for group in by_order.values():
            for m in group:
                task_dir = self._task_dir(m.id)
                result = task_dir / _RESULT
                if result.exists() and _read_json(result)["success"]:
                    continue
                task_dir.mkdir(parents=True, exist_ok=True)
                job = _Job(m, spec, task_dir)
                if job.remaining == 0:
                    job.finish()
                else:
                    jobs.append(job)

        tasks = [(job, label, inp) for job in jobs for label, inp in job.todo]
        for job, label, outcome in self._run(tasks, spec.program):
            job.record(label, outcome)

    def _run(self, tasks: list[tuple[_Job, str, dict]], program: str) -> Iterator[tuple[_Job, str, dict]]:
        task_config = {"ncores": self.ncores_per_task, "memory": self.memory_per_task}
        if self.concurrency == 1:
            for job, label, inp in tasks:
                yield job, label, _safe_call(_run_task, inp, program, task_config)
            return
        pool = ProcessPoolExecutor(max_workers=self.concurrency)
        try:
            futures = {pool.submit(_run_task, inp, program, task_config): (job, label) for job, label, inp in tasks}
            for future in as_completed(futures):
                job, label = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = _failure(exc)
                yield job, label, outcome
        finally:
            # on interrupt, drop queued tasks instead of running them all before exiting
            pool.shutdown(wait=True, cancel_futures=True)

    def _rows(self) -> list[Row]:
        manifest = self._load_manifest()
        if manifest is None:
            raise ValueError(f"no submission found in {self.scratch}; call submit() first")
        known: dict[str, Row] = {}
        for id_, info in manifest["multimers"].items():
            known[id_] = self._row(id_, info)
        active = {int(N): ids for N, ids in manifest["active"].items()}
        return select_active(dict(sorted(active.items())), known)

    def _row(self, id_: str, info: dict) -> Row:
        task_dir = self._task_dir(id_)
        common = {"id": id_, "N": info["N"], "multiplicity": info["multiplicity"], "weight": info["weight"]}
        result = task_dir / _RESULT
        if result.exists():
            done = _read_json(result)
            if done["success"]:
                return Row(**common, energy=done["energy"], status="complete")
            return Row(**common, energy=None, status="failed", error=done["error"])
        if task_dir.is_dir():
            for path in sorted(task_dir.glob("*.json")):
                task = _read_json(path)
                if not task["success"]:
                    return Row(**common, energy=None, status="failed", error=task["error"])
        return Row(**common, energy=None, status="pending")

    def status(self) -> Status:
        counts: dict[int, dict[str, int]] = {}
        for row in self._rows():
            c = counts.setdefault(row.N, {"complete": 0, "pending": 0, "failed": 0})
            c[row.status] += 1
        return Status(counts)

    def collect(self, *, partial: bool = False) -> Results:
        return assemble(self._rows(), partial=partial)


def _failure(exc: BaseException) -> dict:
    return {"success": False, "energy": None, "error": f"{type(exc).__name__}: {exc}"}


def _safe_call(fn, *args) -> dict:
    try:
        return fn(*args)
    except Exception as exc:
        return _failure(exc)
