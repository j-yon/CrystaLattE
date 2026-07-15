from pathlib import Path

from .base import QCWriter
from ...core.multimer import Monomer

_TOP_LEVEL_KEYS = {"method", "bsse_type", "charge", "multiplicity", "memory", "basis"}


class Psi4Writer(QCWriter):
    @classmethod
    def name(cls) -> str:
        return "psi4"

    def write(self, path: Path, monomers: list[Monomer], config: dict) -> None:
        method = config.get("method", "hf")
        basis = config["basis"]
        bsse_type = config.get("bsse_type", "cp")
        charge = config.get("charge", 0)
        multiplicity = config.get("multiplicity", 1)
        memory = config.get("memory")
        set_extras = {k: v for k, v in config.items() if k not in _TOP_LEVEL_KEYS}

        lines: list[str] = ["import psi4", ""]

        if memory:
            lines.append(f"psi4.set_memory('{memory}')")
            lines.append("")

        lines.append("mol = psi4.geometry(\"\"\"")
        for i, monomer in enumerate(monomers):
            if i > 0:
                lines.append("--")
            lines.append(f"  {charge} {multiplicity}")
            for sym, (x, y, z) in zip(monomer.symbols, monomer.cart_coords):
                lines.append(f"  {sym}  {x:.6f}  {y:.6f}  {z:.6f}")
        lines.append("\"\"\")")
        lines.append("")

        options: dict = {"basis": basis, **set_extras}
        options_str = ", ".join(f"'{k}': {repr(v)}" for k, v in options.items())
        lines.append(f"psi4.set_options({{{options_str}}})")
        lines.append("")

        lines.append(f"e = psi4.energy('{method}', bsse_type='{bsse_type}', molecule=mol)")
        lines.append("")

        path.write_text("\n".join(lines))
