from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

if TYPE_CHECKING:
    from .crystal import Crystal


class GenerationInfo(BaseModel):
    """Provenance of a :func:`~crystalatte.generate` run, shared by every multimer it returns

    :param N: Number of monomers per multimer
    :param R: Cutoff in Angstroms
    :param symprec: ``spglib`` symmetry detection tolerance in Angstroms
    :param detect_symmetry: Whether symmetry detection was applied
    :param options: Distance-mode flags (``use_origin``, ``use_com``)
    :param crystal_fingerprint: :func:`crystal_fingerprint` of the input crystal
    :param site_counts: Molecules per unit cell generated from each reference
    :param species: Hill formula of each reference
    :param z_fu: Formula units per unit cell
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    N: int
    R: float
    symprec: float
    detect_symmetry: bool
    options: dict[str, bool]
    crystal_fingerprint: str
    site_counts: tuple[int, ...]
    species: tuple[str, ...]
    z_fu: int

    @field_validator("N")
    @classmethod
    def _check_N(cls, value: int) -> int:
        if value < 2:
            raise ValueError(f"N must be >= 2; got {value}")
        return value

    @field_validator("z_fu")
    @classmethod
    def _check_z_fu(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"z_fu must be >= 1; got {value}")
        return value

    @model_validator(mode="after")
    def _check_lengths(self) -> GenerationInfo:
        if len(self.site_counts) != len(self.species):
            raise ValueError(
                f"site_counts and species must align; got {len(self.site_counts)} "
                f"and {len(self.species)} entries"
            )
        return self


def _round(x: float) -> float:
    """Round to 1e-6, folding -0.0 into 0.0 so the JSON is canonical"""
    return round(float(x), 6) + 0.0


def crystal_fingerprint(crystal: Crystal) -> str:
    """Hash the parts of a crystal that determine its multimers

    :param crystal: :class:`~crystalatte.core.crystal.Crystal` to fingerprint
    :returns: sha256 hex digest of the lattice, space-group operations, ASU and disorder state
    """
    vectors, angles = crystal.lattice_parameters
    asu = crystal._asu
    payload = {
        "lattice": [_round(x) for x in (*vectors, *angles)],
        "sym_ops": sorted(op.op for op in crystal.space_group.sym_ops),
        "symbols": list(asu.atoms),
        "positions": [[_round(x) for x in row] for row in asu.positions],
        "state": crystal.state.label if crystal.state is not None else None,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()
