import re
import warnings
from collections.abc import Generator

import numpy as np
from CifFile import ReadCif
from numpy.typing import NDArray

from ..core.crystal import Crystal
from ..core.multimer import ASU, Monomer
from ..core.space_group import SpaceGroup
from ..core.sym_ops import SymOp

# accepted spellings for each item
_NUMBER_FIELDS = ("_symmetry_int_tables_number", "_space_group_it_number")
_HALL_FIELDS = ("_symmetry_space_group_name_hall", "_space_group_name_hall")
_HM_FIELDS = (
    "_symmetry_space_group_name_h-m",
    "_space_group_name_h-m_alt",
    "_space_group_name_h-m",
    "_space_group_name_h-m_full",
)
_SYMOP_FIELDS = ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz")
_CELL_LENGTH_FIELDS = ("_cell_length_a", "_cell_length_b", "_cell_length_c")
_CELL_ANGLE_FIELDS = ("_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma")

# lazily built map from a normalised Hermann-Mauguin symbol to its Hall symbol
_HM_INDEX: dict[str, str] = {}

# occupancy at or above this counts as fully occupied
_FULL_OCCUPANCY = 0.999

# what `from_cif` may be told to do about partially occupied sites
DISORDER_POLICIES = ("error", "ignore", "drop")


def _clean_block(block):
    """Strip standard-uncertainty parentheses and convert numeric fields to floats

    :param block: Raw ``pycifrw`` data block
    :returns: Plain dict with numbers converted and uncertainties removed
    """
    cleaned_block = {}
    # pycifrw partitions CIF into a dict of lists/strings/floats
    for key, value in block.items():
        # found a list, corresponds to a looped field
        if isinstance(value, list):
            cleaned_block[key] = []
            for item in value:
                if isinstance(item, str):
                    # clean uncertainties and convert to float if possible
                    cleaned_item = re.sub(r"\([^)]*\)", "", item)
                    try:
                        cleaned_block[key].append(float(cleaned_item))
                    except ValueError:
                        cleaned_block[key].append(cleaned_item)
                else:
                    cleaned_block[key].append(item)

        # found a string, corresponds to a single-valued field
        elif isinstance(value, str):
            # clean uncertainties and convert to float if possible
            cleaned_value = re.sub(r"\([^)]*\)", "", value)
            try:
                cleaned_block[key] = float(cleaned_value)
            except ValueError:
                cleaned_block[key] = cleaned_value

        # if not a string or list, leave it as is (most likely float)
        else:
            cleaned_block[key] = value

    return cleaned_block


def _first(block: dict, fields: tuple[str, ...]):
    """Return the first of ``fields`` present in ``block``, or ``None``

    :param block: Cleaned data block
    :param fields: Accepted spellings, most-preferred first
    :returns: Value of the first field present, else None
    """
    for name in fields:
        if name in block and block[name] not in (None, "", "?", "."):
            return block[name]
    return None


def _normalise_symbol(symbol) -> str:
    """Reduce a space-group symbol to a comparable form

    :param symbol: Symbol as written in the file
    :returns: Normalized symbol
    """
    return re.sub(r"[\s_]", "", str(symbol)).lower()


def _hall_from_hm(symbol: str) -> str | None:
    """Map a Hermann-Mauguin symbol to the Hall symbol naming the same setting

    :param symbol: Hermann-Mauguin symbol as written in the file
    :returns: Hall symbol, or None if unknown or ambiguous
    """
    if not _HM_INDEX:
        if SpaceGroup._data == {}:
            SpaceGroup.load_data()

        collisions: set[str] = set()
        for key, entry in SpaceGroup._data.items():
            if key.isdigit():  # number-keyed duplicate of a Hall-keyed entry
                continue
            for field in ("short", "Hermann-Mauguin"):
                value = entry["symbol"].get(field)
                if not value:
                    continue
                norm = _normalise_symbol(value)
                if norm in _HM_INDEX and _HM_INDEX[norm] != key:
                    collisions.add(norm)
                _HM_INDEX.setdefault(norm, key)

        for norm in collisions:
            del _HM_INDEX[norm]

    return _HM_INDEX.get(_normalise_symbol(symbol))


def _listed_sym_ops(block: dict) -> list[SymOp] | None:
    """Return the operations the file lists explicitly, if it lists any

    :param block: Cleaned data block
    :returns: Operations as :class:`SymOp`, or ``None`` if the file lists none
    """
    raw = _first(block, _SYMOP_FIELDS)
    if raw is None:
        return None  # no operations listed

    entries = raw if isinstance(raw, list) else [raw]
    ops = []
    for entry in entries:
        text = str(entry).strip().strip("'\"")

        # some writers prefix each operation with its loop index ("1 x,y,z").
        text = re.sub(r"^\d+\s+", "", text)
        if text:
            ops.append(SymOp(text))
    return ops or None


def _op_signature(ops) -> set[tuple[bytes, bytes]]:
    """Return a comparable signature for a set of operations, e.g. different coset representatives of the same setting

    :param ops: Operations to summarize
    :returns: Set of ``(rotation, translation % 1)`` byte pairs
    """
    return {
        (
            np.asarray(op.rot, dtype=np.int64).tobytes(),
            (np.asarray(op.tr, dtype=np.int64) % 12).tobytes(), # mod 12 to match int convention
        )
        for op in ops
    }


def _verify_against_listed_ops(space_group: SpaceGroup, listed, source: str) -> SpaceGroup:
    """Check a resolved space group against the file's own operation list

    :param space_group: :class:`SpaceGroup` resolved from symbol or number
    :param listed: Operations the file lists, or None
    :param source: How the :class:`SpaceGroup` was resolved for the warning text
    :returns: ``space_group``, or one rebuilt from ``listed`` on disagreement
    """
    if not listed or _op_signature(space_group.sym_ops) == _op_signature(listed):
        return space_group

    warnings.warn(
        f"CIF declares a space group ({source}) whose {len(space_group.sym_ops)} "
        f"operations do not match the {len(listed)} operations the file itself "
        "lists. Using the listed operations...",
        RuntimeWarning,
        stacklevel=3,
    )
    return SpaceGroup.from_sym_ops(
        listed, number=space_group.number, symbol=space_group.symbol
    )


def _resolve_space_group(block: dict) -> SpaceGroup:
    """Determine the space group of one data block

    :param block: Cleaned data block
    :returns: Resolved :class:`SpaceGroup`
    :raises ValueError: If the block carries no usable symmetry information
    """
    if SpaceGroup._data == {}:
        SpaceGroup.load_data()

    # resolve Hall and Hermann-Mauguin symbols, and the international number, if present
    hall = _first(block, _HALL_FIELDS)
    hall = hall.strip() if isinstance(hall, str) else None
    hm = _first(block, _HM_FIELDS)
    hm = hm.strip() if isinstance(hm, str) else None
    raw_number = _first(block, _NUMBER_FIELDS)

    # grab the operations the file lists, if any
    listed = _listed_sym_ops(block)

    # source: Hall -> Hermann-Mauguin -> number, in that order
    source = "Hall symbol" if hall else None
    if hall is None and hm is not None:
        hall = _hall_from_hm(hm)
        if hall is not None:
            source = f"Hermann-Mauguin symbol {hm!r}"

    number = int(raw_number) if raw_number is not None else None
    if number is None and hall is not None:
        entry = SpaceGroup._data.get(hall)
        number = entry["number"] if entry else None

    if number is not None:
        if source is None:
            source = "International number (standard setting)"
        return _verify_against_listed_ops(SpaceGroup(number, hall=hall), listed, source)

    # last resort, match based on the listed operations
    if listed:
        return SpaceGroup.from_sym_ops(
            listed, symbol={"Hermann-Mauguin": hm} if hm else None
        )

    # we've tried all we can
    raise ValueError(
        "CIF block carries no usable symmetry information: no space group number "
        f"({' / '.join(_NUMBER_FIELDS)}), no Hall symbol "
        f"({' / '.join(_HALL_FIELDS)}), no recognised Hermann-Mauguin symbol "
        f"(found {hm!r})" + (", " if hm else " ") + "and no explicit operation "
        f"list ({' / '.join(_SYMOP_FIELDS)})."
    )


def _site_occupancies(block: dict, n_sites: int) -> NDArray[np.float64]:
    """Return per-site occupancies from the block, defaulting to 1.0 if absent

    :param block: Cleaned data block
    :param n_sites: How many atom sites the block describes
    :returns: An ``(n_sites,)`` float array of occupancies
    """
    raw = _first(block, ("_atom_site_occupancy",))
    if raw is None:
        return np.ones(n_sites)  # default to fully occupied if the field is absent

    values = raw if isinstance(raw, list) else [raw]
    occupancies = np.array(
        [float(v) if not isinstance(v, str) else 1.0 for v in values], dtype=float
    )
    if len(occupancies) != n_sites:
        return np.ones(
            n_sites
        )  # default to fully occupied if the field is present but has the wrong length

    return occupancies


def _apply_disorder_policy(block: dict, coords: NDArray[np.float64], policy: str) -> np.ndarray:
    """Resolve partially occupied sites according to ``policy``

    :param block: Cleaned data block
    :param coords: Fractional coordinates, shape ``(n, 3)``
    :param policy: One of :data:`DISORDER_POLICIES`
    :returns: Boolean mask selecting the sites to keep
    :raises ValueError: If ``policy`` is unknown or is ``"error"`` and the block is disordered
    """
    if policy not in DISORDER_POLICIES:
        raise ValueError(
            f"unknown disorder policy {policy!r}; expected one of {DISORDER_POLICIES}"
        )

    occupancies = _site_occupancies(block, len(coords))
    partial = occupancies < _FULL_OCCUPANCY
    if not partial.any():
        return np.ones(len(coords), dtype=bool)  # found no partial occupancy

    labels = _first(block, ("_atom_site_label",)) or []
    labels = labels if isinstance(labels, list) else [labels]
    named = ", ".join(
        f"{labels[i] if i < len(labels) else f'site {i}'}={occupancies[i]:g}"
        for i in np.flatnonzero(partial)
    )

    # the user has asked us to error on partial occupancy, so do that
    if policy == "error":
        raise ValueError(
            f"CIF contains {int(partial.sum())} partially occupied site(s) "
            f"({named}), but the disorder policy is set to 'error'."
        )

    # the user has asked us to ignore partial occupancy, so warn and keep all sites
    if policy == "ignore":
        warnings.warn(
            f"CIF contains partially occupied site(s) ({named}); reading them as "
            "ordinary atoms, which can produce molecules that do not exist.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.ones(len(coords), dtype=bool)

    # the user has asked us to drop partial occupancy, so warn and keep only the fully occupied sites
    warnings.warn(
        f"CIF contains partially occupied site(s) ({named}); dropping them and "
        "keeping only the fully occupied substructure.",
        RuntimeWarning,
        stacklevel=3,
    )
    return ~partial


def _resolve_asu(block: dict, coords: NDArray[np.float64], keep: NDArray[np.int16]) -> ASU:
    """Build the asymmetric unit, deriving element symbols however the file allows

    :param block: Cleaned data block
    :param coords: Fractional coordinates, shape ``(n, 3)``
    :param keep: Boolean mask of the sites to retain, from the disorder policy
    :returns: Assembled :class:`ASU`
    :raises ValueError: If neither field is present
    """
    symbols = None
    if "_atom_site_type_symbol" in block:
        symbols = [
            re.match(r"[A-Za-z]+", s).group() for s in block["_atom_site_type_symbol"]
        ]
    elif "_atom_site_label" in block:
        symbols = [
            re.sub(r"[\d\(\)]", "", label) for label in block["_atom_site_label"]
        ]

    if symbols is not None:
        return ASU([s for s, k in zip(symbols, keep) if k], coords[keep])

    raise ValueError(
        "CIF file must contain either _atom_site_type_symbol or _atom_site_label "
        "to determine atomic symbols."
    )


def from_cif(
    cif_file: str, disorder: str = "error"
) -> Generator[tuple[Crystal, Monomer]]:
    """Parse a CIF file into :class:`Crystal` / :class:`Monomer` pairs.

    .. note::
        The CIF format allows multiple data blocks per file, so this function yields instead
        of returning a single crystal

    :param cif_file: Path to the CIF file
    :param disorder: Policy for partially occupied sites ("error", "ignore", or "drop")
    :returns: Generator of ``(Crystal, Monomer)``, one per data block
    :raises ValueError: If block information is generally inconsistent or uninterpretable
    """
    # grab the raw CIF data using pycifrw
    cif = ReadCif(cif_file)
    blocks = [item[-1] for item in cif.items()]

    # for each block, clean it up and extract the crystal information
    for block in blocks:
        block = _clean_block(block)

        missing = [
            f
            for f in (*_CELL_LENGTH_FIELDS, *_CELL_ANGLE_FIELDS)
            if _first(block, (f,)) is None
        ]
        if missing:
            raise ValueError(f"CIF block is missing cell parameters: {missing}")

        a, b, c = (
            float(_first(block, (f,))) for f in _CELL_LENGTH_FIELDS
        )  # lattice lengths
        alpha, beta, gamma = (
            float(_first(block, (f,))) for f in _CELL_ANGLE_FIELDS
        )  # lattice angles

        # attempt to resolve the space group
        space_group = _resolve_space_group(block)

        # grab the fractional coordinates and apply the disorder policy
        coords = np.array(
            [
                block["_atom_site_fract_x"],
                block["_atom_site_fract_y"],
                block["_atom_site_fract_z"],
            ]
        ).T
        keep = _apply_disorder_policy(block, coords, disorder)

        # finally, assemble the crystal and its asymmetric unit
        crystal = Crystal(
            (a, b, c),
            (alpha, beta, gamma),
            space_group,
            _resolve_asu(block, coords, keep),
        )

        # cross-check the declared volume against the one the cell implies
        declared = _first(block, ("_cell_volume",))
        if declared is not None and abs(crystal.volume - float(declared)) > 0.1:
            raise ValueError(
                f"Discordant unit cell volumes:\nDeclared in CIF: {float(declared):.2f} A^3\n"
                f"Calculated from primitive vectors: {crystal.volume:.2f} A^3"
            )

        # since the CIF format allows multiple data blocks, yield each crystal
        yield crystal
