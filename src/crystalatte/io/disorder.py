"""Enumerate the ordered states a disordered CIF block admits

Two kinds of disorder are handled:

* explicit groups (``_atom_site_disorder_assembly`` / ``_atom_site_disorder_group``): one group
  is chosen per assembly, and the declared space group still holds
* symmetry-generated disorder (untagged or negative-group partial sites): a subset of the partial
  atoms of the full unit cell is chosen, and the result has no symmetry until
  :meth:`~crystalatte.core.crystal.Crystal.detect_symmetry` recovers it
"""

import itertools
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# occupancy at or above this counts as fully occupied
FULL_OCCUPANCY = 0.999

# an assembly whose group weights sum below this also admits "no group present"
_ABSENT_BELOW = 0.98


def normalize_tag(value) -> str | None:
    """Normalise a disorder assembly or group value

    :param value: Raw value from the cleaned block (numbers arrive as floats)
    :returns: ``None`` for an untagged site, integral numbers without a decimal point, else the text
    """
    if value is None:
        return None
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    text = str(value).strip()
    return None if text in ("", ".", "?") else text


class ExplicitChoice(NamedTuple):
    """One group chosen per explicit disorder assembly

    :param keep: Mask of the ASU sites present in this choice
    :param settled: Mask of the ASU sites whose occupancy this choice resolved
    :param groups: ``(assembly, group)`` pairs; ``group`` is ``"-"`` when the assembly is absent
    :param label: e.g. ``"A1 B2"``; empty when the block has no explicit groups
    :param weight: Product of the chosen groups' occupancies
    """

    keep: NDArray[np.bool_]
    settled: NDArray[np.bool_]
    groups: tuple[tuple[str, str], ...]
    label: str
    weight: float


def _group_order(group: str) -> tuple:
    return (0, int(group), "") if group.isdigit() else (1, 0, group)


def explicit_choices(
    occupancies, assemblies: list[str | None], groups: list[str | None]
) -> list[ExplicitChoice]:
    """Enumerate one group per explicit disorder assembly, taking assemblies as independent

    A site takes part when it has a positive group and is partially occupied. A tagged but
    fully occupied site is kept in every choice; negative groups and untagged partial sites
    are left untouched for the symmetry stage.

    :param occupancies: Per-site occupancies
    :param assemblies: Per-site normalised assembly names (``None`` when untagged or absent)
    :param groups: Per-site normalised group names (``None`` when untagged)
    :returns: Every combination of choices, in assembly then group order
    """
    occupancies = np.asarray(occupancies, dtype=float)
    n = len(occupancies)
    selectable = np.array(
        [
            g is not None and not g.startswith("-") and o < FULL_OCCUPANCY
            for g, o in zip(groups, occupancies)
        ],
        dtype=bool,
    )
    if not selectable.any():
        return [ExplicitChoice(np.ones(n, dtype=bool), np.zeros(n, dtype=bool), (), "", 1.0)]

    # options for each assembly: its groups and their weights
    assembly_of = [a or "" for a in assemblies]
    members = np.flatnonzero(selectable)
    options: dict[str, list[tuple[str, float]]] = {}
    for name in sorted({assembly_of[i] for i in members}):
        mine = [i for i in members if assembly_of[i] == name]
        opts = [
            (g, float(np.mean([occupancies[i] for i in mine if groups[i] == g])))
            for g in sorted({groups[i] for i in mine}, key=_group_order)
        ]
        total = sum(w for _, w in opts)
        if total < _ABSENT_BELOW:
            opts.append(("-", 1.0 - total))
        options[name] = opts

    choices = []
    for combo in itertools.product(*options.values()):
        picked = dict(zip(options, (g for g, _ in combo)))
        keep = np.array(
            [not selectable[i] or picked[assembly_of[i]] == groups[i] for i in range(n)],
            dtype=bool,
        )
        pairs = tuple(picked.items())
        label = " ".join(f"{a}{g}" for a, g in pairs)
        weight = float(np.prod([w for _, w in combo]))
        choices.append(ExplicitChoice(keep, selectable.copy(), pairs, label, weight))
    return choices
