from pathlib import Path

from ...core.multimer import Monomer
from ...compute.spec import ComputeSpec
from .base import QCWriter


class Psi4Writer(QCWriter):
    """Writer for Psi4 input files"""

    @classmethod
    def name(cls) -> str:
        return "psi4"

    @classmethod
    def extension(cls) -> str:
        return "in"

    def write(
        self,
        path: Path,
        monomers: list[Monomer],
        spec: ComputeSpec,
        *,
        memory: str | None = None,
    ) -> None:
        method = spec.method
        basis = spec.basis
        bsse_type = spec.bsse
        charge = 0
        multiplicity = 1
        set_extras = spec.final_keywords

        # begin constructing the input file
        lines: list[str] = ["import psi4", ""]

        if memory:
            lines.append(f"psi4.set_memory('{memory}')")
            lines.append("")

        lines.append('mol = psi4.geometry("""')
        for i, monomer in enumerate(monomers):
            if i > 0:
                lines.append("--")
            lines.append(f"  {charge} {multiplicity}")
            for sym, (x, y, z) in zip(monomer.symbols, monomer.cart_coords):
                lines.append(f"  {sym}  {x:.6f}  {y:.6f}  {z:.6f}")
        lines.append('""")')
        lines.append("")

        options: dict = {"basis": basis, **set_extras}
        options_str = ", ".join(f"'{k}': {v!r}" for k, v in options.items())
        lines.append(f"psi4.set_options({{{options_str}}})")
        lines.append("")

        lines.append(
            f"e = psi4.energy('{method}', bsse_type='{bsse_type}', molecule=mol)"
        )
        lines.append("")

        # write the lines to the specified path
        path.write_text("\n".join(lines))
