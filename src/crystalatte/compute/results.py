"""Backend-independent results layer: per-multimer rows and lattice-energy assembly."""

from dataclasses import asdict, dataclass
from typing import Literal

import qcelemental as qcel

from ..core.multimer import order_name

_LIST_LIMIT = 20
_ERROR_LIMIT = 200


@dataclass(frozen=True)
class Row:
    """One multimer's result

    `energy` is the nonadditive energy in hartree. `geometric_mean` (centroid distances) and
    `min_separation` (closest atoms) are in Angstroms, and `formulas` holds each monomer's Hill
    formula. All three are ``None`` for results stored before they were recorded.
    """

    id: str
    N: int
    multiplicity: int
    weight: float
    energy: float | None
    status: Literal["complete", "pending", "failed"]
    error: str | None = None
    geometric_mean: float | None = None
    min_separation: float | None = None
    formulas: tuple[str, ...] | None = None


def multimer_fields(info: dict) -> dict:
    """The :class:`Row` fields describing a multimer, from its stored manifest entry"""
    formulas = info.get("formulas")
    return {
        "geometric_mean": info.get("geometric_mean"),
        "min_separation": info.get("min_separation"),
        "formulas": None if formulas is None else tuple(formulas),
    }


def _missing(rows: list[Row]) -> dict[int, dict[str, list[str]]]:
    missing: dict[int, dict[str, list[str]]] = {}
    for row in rows:
        if row.status == "complete":
            continue
        entry = missing.setdefault(row.N, {"failed": [], "pending": []})
        entry["failed" if row.status == "failed" else "pending"].append(row.id)
    return dict(sorted(missing.items()))


def _truncated(ids: list[str]) -> str:
    shown = ", ".join(ids[:_LIST_LIMIT])
    if len(ids) > _LIST_LIMIT:
        shown += f", ... and {len(ids) - _LIST_LIMIT} more"
    return shown


class IncompleteResultsError(Exception):
    """Raised when a lattice energy is requested from incomplete results."""

    def __init__(self, missing: dict[int, dict[str, list[str]]], errors: dict[str, str] | None = None):
        self.missing = missing
        self.errors = errors or {}
        super().__init__(self._message())

    def _message(self) -> str:
        lines = ["Cannot assemble lattice energy: results are incomplete."]
        for N, entry in self.missing.items():
            failed, pending = entry["failed"], entry["pending"]
            lines.append(f"{order_name(N)}s: {len(failed)} failed, {len(pending)} pending")
            if failed:
                shown = [f"{i} ({self.errors.get(i, '')[:_ERROR_LIMIT]})" for i in failed[:_LIST_LIMIT]]
                tail = f", ... and {len(failed) - _LIST_LIMIT} more" if len(failed) > _LIST_LIMIT else ""
                lines.append(f"  failed: {', '.join(shown)}{tail}")
            if pending:
                lines.append(f"  pending: {_truncated(pending)}")
        return "\n".join(lines)


def _error_for(rows: list[Row]) -> IncompleteResultsError:
    errors = {r.id: r.error or "" for r in rows if r.status == "failed"}
    return IncompleteResultsError(_missing(rows), errors)


class Results:
    """Per-multimer rows and the lattice energy assembled from them

    ``lattice_energy`` and ``by_order`` are in kJ/mol per formula unit:
    ``sum(weight * energy)`` over the rows, where each weight already divides by ``N`` and
    ``Z_fu`` (see :attr:`~crystalatte.core.multimer.Multimer.weight`). Each row's ``energy`` is
    the multimer's nonadditive N-body interaction energy in hartree. The energy of deforming or
    relaxing the monomers from their gas-phase geometries is not included.
    """

    def __init__(self, rows: list[Row]):
        self.rows = list(rows)

    @property
    def missing(self) -> dict[int, dict[str, list[str]]]:
        return _missing(self.rows)

    @property
    def complete(self) -> bool:
        return all(r.status == "complete" for r in self.rows)

    def _require_complete(self) -> None:
        if not self.complete:
            raise _error_for(self.rows)

    @property
    def by_order(self) -> dict[int, float]:
        self._require_complete()
        out: dict[int, float] = {}
        for r in self.rows:
            out[r.N] = out.get(r.N, 0.0) + r.weight * r.energy * qcel.constants.hartree2kJmol
        return dict(sorted(out.items()))

    @property
    def lattice_energy(self) -> float:
        self._require_complete()
        return sum(r.weight * r.energy for r in self.rows) * qcel.constants.hartree2kJmol

    @property
    def table(self) -> list[dict]:
        return [asdict(r) for r in self.rows]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.table)


def assemble(rows: list[Row], *, partial: bool) -> Results:
    if not partial and not all(r.status == "complete" for r in rows):
        raise _error_for(rows)
    return Results(rows)


@dataclass
class Status:
    counts: dict[int, dict[str, int]]

    def __str__(self) -> str:
        return "\n".join(
            f"{order_name(N)}s: {c.get('complete', 0)} complete, "
            f"{c.get('pending', 0)} pending, {c.get('failed', 0)} failed"
            for N, c in sorted(self.counts.items())
        )


def select_active(active: dict[int, list[str]], known: dict[str, Row]) -> list[Row]:
    """Rows for the active IDs only; an active ID without a stored row is pending."""
    rows = []
    for N, ids in active.items():
        for id_ in ids:
            row = known.get(id_)
            rows.append(row if row is not None else Row(id_, N, 1, 1.0, None, "pending"))
    return rows
