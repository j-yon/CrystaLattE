"""Model-chemistry description shared by every compute backend."""

import hashlib
import json
import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import to_jsonable_python

LEVEL_LABEL = "crystalatte"

DEFAULT_KEYWORDS: dict[str, dict] = {
    "psi4": {
        "e_convergence": 10,
        "d_convergence": 10,
        "scf_type": "df",
        "mp2_type": "df",
        "cc_type": "df",
        "freeze_core": "true",
    },
}

_WARNED_PROGRAMS: set[str] = set()


class ComputeSpec(BaseModel):
    """Program, method, basis, BSSE treatment and keywords for one model chemistry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    program: str
    method: str
    basis: str
    keywords: dict[str, Any] = Field(default_factory=dict)
    bsse: Literal["cp", "nocp", "vmfc"] = "cp"
    use_defaults: bool = True

    @field_validator("program", "method", "basis", mode="after")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower()

    @field_validator("bsse", mode="before")
    @classmethod
    def _lower_bsse(cls, v: Any) -> Any:
        return v.lower() if isinstance(v, str) else v

    @property
    def final_keywords(self) -> dict:
        """Program defaults overlaid by user keywords (user wins)."""
        if not self.use_defaults:
            return dict(self.keywords)
        defaults = DEFAULT_KEYWORDS.get(self.program)
        if defaults is None:
            if self.program not in _WARNED_PROGRAMS:
                _WARNED_PROGRAMS.add(self.program)
                warnings.warn(
                    f"no default keywords for program '{self.program}'; "
                    "set tight convergence thresholds in keywords",
                    UserWarning,
                    stacklevel=2,
                )
            defaults = {}
        return {**defaults, **self.keywords}

    def atomic_template(self) -> dict:
        return {
            "driver": "energy",
            "model": {"method": self.method, "basis": self.basis},
            "keywords": self.final_keywords,
        }

    def manybody_kwargs(self, N: int) -> dict:
        return {
            "bsse_type": [self.bsse],
            "levels": {k: LEVEL_LABEL for k in range(1, N + 1)},
            "return_total_data": False,
            "supersystem_ie_only": False,
            "embedding_charges": {},
        }

    def to_portal(self, N: int):
        from qcportal.manybody import ManybodySpecification
        from qcportal.singlepoint import QCSpecification

        qc = QCSpecification(
            program=self.program,
            driver="energy",
            method=self.method,
            basis=self.basis,
            keywords=self.final_keywords,
        )
        return ManybodySpecification(
            program="qcmanybody",
            levels={k: qc for k in range(1, N + 1)},
            bsse_correction=[self.bsse],
        )

    def fingerprint(self) -> str:
        """Hash of what is sent to the program, so a change in the defaults changes it too"""
        sent = {
            "program": self.program,
            "method": self.method,
            "basis": self.basis,
            "bsse": self.bsse,
            "final_keywords": self.final_keywords,
        }
        payload = json.dumps(to_jsonable_python(sent), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
