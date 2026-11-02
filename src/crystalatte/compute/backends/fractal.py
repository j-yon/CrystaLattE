"""Managed backend: one QCFractal ManybodyDataset per multimer order, run by the server's workers."""

import json
import re

from ...core.generation import GenerationInfo
from ...core.multimer import Multimer, order_name
from ..results import Results, Row, Status, assemble, select_active
from ..spec import ComputeSpec
from .base import Backend

_INSTALL_HINT = "FractalBackend needs qcportal; install it with: pip install 'crystalatte[fractal]'"
_DATASET_TYPE = "manybody"
_SPEC_NAME = "crystalatte"
_EXTRAS_KEY = "crystalatte"
# a complete ManybodyRecord keeps QCManyBody's results under properties["results"];
# verified against QCFractal 0.71 + QCManyBody 0.8 on a FractalSnowflake run
_RESULTS_KEY = "results"
_ENERGY_KEY = "{bsse}_corrected_{N}_body_contribution_to_energy"


def _require_extras() -> None:
    try:
        import qcportal  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc


def _order_of(suffix: str) -> int | None:
    """Invert :func:`order_name` for a dataset suffix like ``dimer`` or ``7mer``"""
    match = re.fullmatch(r"(\d+)mer", suffix)
    N = int(match.group(1)) if match else next((n for n in range(2, 7) if order_name(n) == suffix), None)
    # reject non-canonical spellings such as "2mer" so each order maps to exactly one dataset
    return N if N is not None and order_name(N) == suffix else None


class FractalBackend(Backend):
    """Run many-body calculations on a QCFractal server, one ``ManybodyDataset`` per order

    Datasets are named ``f"{name}-{order_name(N)}s"`` and carry what is needed to assemble the
    lattice energy in their extras, so a fresh ``FractalBackend(client, name)`` can collect.

    :param client: a connected :class:`qcportal.PortalClient`
    :param name: prefix of the dataset names; one name holds one model chemistry and one crystal
    :param tag: compute tag for submitted records; defaults to the dataset's default tag
    :param priority: compute priority for submitted records; defaults to the dataset's default
    """

    def __init__(self, client, name: str, *, tag: str | None = None, priority: str | None = None):
        _require_extras()
        self.client = client
        self.name = name
        self.tag = tag
        self.priority = priority

    def _dataset_name(self, N: int) -> str:
        return f"{self.name}-{order_name(N)}s"

    def _datasets(self) -> dict[int, object]:
        """Every dataset on the server that belongs to this name, by order"""
        prefix = f"{self.name}-".lower()
        found: dict[int, object] = {}
        for info in self.client.list_datasets():
            ds_name = info["dataset_name"]
            if info["dataset_type"] != _DATASET_TYPE or not ds_name.lower().startswith(prefix):
                continue
            rest = ds_name[len(prefix):]
            N = _order_of(rest[:-1]) if rest.endswith("s") else None
            if N is not None:
                found[N] = self.client.get_dataset(_DATASET_TYPE, ds_name)
        return dict(sorted(found.items()))

    def _meta(self, ds) -> dict:
        """CrystaLattE's extras of a dataset, validated

        :raises ValueError: if the dataset was not made by CrystaLattE
        :raises pydantic.ValidationError: if the stored spec or generation info is corrupted
        """
        meta = (ds.extras or {}).get(_EXTRAS_KEY)
        if meta is None:
            raise ValueError(f"dataset {ds.name!r} exists but was not created by CrystaLattE; pick another name")
        ComputeSpec.model_validate(meta["spec"])
        GenerationInfo.model_validate(meta["generation"])
        return meta

    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None:
        by_order = self._validate(multimers)
        crystal_fp = multimers[0].generation.crystal_fingerprint
        existing = self._datasets()
        for ds in existing.values():
            meta = self._meta(ds)
            if meta["spec_fingerprint"] != spec.fingerprint():
                raise ValueError(
                    f"spec differs from the one used in dataset {ds.name!r} (spec fingerprint "
                    f"{meta['spec_fingerprint'][:12]} vs {spec.fingerprint()[:12]}); use a new name"
                )
            if meta["crystal_fingerprint"] != crystal_fp:
                raise ValueError(
                    f"crystal fingerprint differs from the one used in dataset {ds.name!r} "
                    f"({meta['crystal_fingerprint'][:12]} vs {crystal_fp[:12]}); use a new name"
                )

        from qcportal.manybody import ManybodyDatasetNewEntry

        for N, group in by_order.items():
            ds = existing.get(N) or self.client.add_dataset(_DATASET_TYPE, self._dataset_name(N))
            if _SPEC_NAME not in ds.specification_names:
                ds.add_specification(_SPEC_NAME, spec.to_portal(N))
            have = set(ds.entry_names)
            new = [
                ManybodyDatasetNewEntry(
                    name=m.id,
                    initial_molecule=json.loads(m.to_molecule().json()),
                    attributes={
                        "N": N,
                        "multiplicity": m.multiplicity,
                        "weight": m.weight,
                        "crystal_fingerprint": crystal_fp,
                    },
                )
                for m in group
                if m.id not in have
            ]
            if new:
                ds.add_entries(new)
            meta = {
                "spec": spec.model_dump(mode="json"),
                "spec_fingerprint": spec.fingerprint(),
                "crystal_fingerprint": crystal_fp,
                "generation": group[0].generation.model_dump(mode="json"),
                "active": [m.id for m in group],
            }
            ds.set_extras({**(ds.extras or {}), _EXTRAS_KEY: meta})
            ds.submit(specification_names=_SPEC_NAME, compute_tag=self.tag, compute_priority=self.priority)

    def _rows(self) -> list[Row]:
        datasets = self._datasets()
        if not datasets:
            raise ValueError(f"no submission named {self.name!r} found on the server; call submit() first")
        active: dict[int, list[str]] = {}
        known: dict[str, Row] = {}
        for N, ds in datasets.items():
            meta = self._meta(ds)
            spec = ComputeSpec.model_validate(meta["spec"])
            active[N] = meta["active"]
            key = _ENERGY_KEY.format(bsse=spec.bsse, N=N)
            for entry in ds.iterate_entries():
                attrs = entry.attributes
                known[entry.name] = Row(
                    entry.name, N, attrs["multiplicity"], attrs["weight"], None, "pending"
                )
            for entry_name, _, record in ds.iterate_records(specification_names=_SPEC_NAME):
                row = known[entry_name]
                if record.status == "complete":
                    energy = float(record.properties[_RESULTS_KEY][key])
                    known[entry_name] = Row(row.id, N, row.multiplicity, row.weight, energy, "complete")
                elif record.status == "error":
                    error = record.error or {}
                    message = error.get("error_message") if isinstance(error, dict) else str(error)
                    known[entry_name] = Row(
                        row.id, N, row.multiplicity, row.weight, None, "failed", message or "unknown error"
                    )
        return select_active(active, known)

    def status(self) -> Status:
        counts: dict[int, dict[str, int]] = {}
        for row in self._rows():
            c = counts.setdefault(row.N, {"complete": 0, "pending": 0, "failed": 0})
            c[row.status] += 1
        return Status(counts)

    def collect(self, *, partial: bool = False) -> Results:
        return assemble(self._rows(), partial=partial)

    def resubmit_errors(self) -> int:
        """Reset errored records so the server runs them again

        :returns: number of records reset
        """
        from qcportal.record_models import RecordStatusEnum

        n = 0
        for ds in self._datasets().values():
            errored = [
                entry_name
                for entry_name, _, _ in ds.iterate_records(
                    specification_names=_SPEC_NAME, status=RecordStatusEnum.error
                )
            ]
            if errored:
                ds.reset_records(entry_names=errored, specification_names=_SPEC_NAME)
                n += len(errored)
        return n
