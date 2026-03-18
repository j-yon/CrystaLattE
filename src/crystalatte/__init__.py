"""
Crys2LattE: Automated crystal structure generation and analysis for many-body applications.
TODO: fill this out
"""

from .io.cif import from_cif
from .work.generate import generate

__all__ = ["from_cif", "generate"]
