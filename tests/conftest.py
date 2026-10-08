import warnings
from pathlib import Path

import pytest

import crystalatte as cle

# root of fixture tree, crystals live in x23/ and other/
DATA = Path(__file__).parent

GENERATE_KW = {"use_origin": True, "n_jobs": 1, "verbose": 0}

# ice is proton disordered (4 sites with 1/2 occupancy each)
DISORDERED = {"ice"}

# mark tests that are slow on some fixtures, so they can be deselected with -m "not slow"
_SLOW_CRYSTALS = frozenset({"hexamine"})
_SLOW_TESTS = frozenset({
    "test_full_congruence_check_is_unconditional",
    "test_generate_runs_on_high_symmetry_crystals",
    "test_get_unique_sym_ops_ambiguous_wyckoff_position",
})

def cif(name: str) -> str:
    """Return the path to a crystal fixture by bare name

    :param name: crystal directory name, e.g. "benzene"
    :returns: absolute path to ``<name>/<name>.cif`` as a string
    :raises FileNotFoundError: if there is no such fixture directory, or it does not hold a CIF
    """
    for sub_dir in ("x23", "other"):
        folder = DATA / sub_dir / name
        if not folder.is_dir():
            continue
        path = folder / f"{name}.cif"
        if path.exists():
            return str(path)
        others = sorted(p.name for p in folder.glob("*.cif"))
        raise FileNotFoundError(
            f"fixture directory {sub_dir}/{name}/ exists but holds no "
            f"{name}.cif" + (f" (found {others})" if others else " at all")
        )
    available = sorted(
        p.name for d in ("x23", "other") for p in (DATA / d).iterdir() if p.is_dir()
    )
    raise FileNotFoundError(
        f"no CIF fixture named {name!r}; available fixtures are {available}"
    )

def load(name: str, disorder: str | None = None):
    """Parse a fixture and return its first ``(crystal, reference)`` block

    :param name: crystal directory name, e.g. "benzene"
    :param disorder: policy for partially occupied sites
    :returns: ``(Crystal, Monomer)`` pair for the file's first block
    """
    if disorder is None:
        # if known to be disordered, ignore the warning; otherwise, raise it as an error
        disorder = "ignore" if name in DISORDERED else "error"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*partially occupied.*")
        return next(cle.from_cif(cif(name), disorder=disorder))


def crystal(name: str, disorder: str | None = None):
    """Parse a fixture and return the :class:`Crystal`

    :param name: crystal directory name, e.g. "benzene"
    :param disorder: passed through to :func:`load`
    :returns: the parsed :class:`~crystalatte.core.crystal.Crystal`
    """
    return load(name, disorder)[0]


@pytest.fixture
def cif_path():
    """fixture form of :func:`cif` for tests that prefer injection"""
    return cif


def pytest_configure(config):
    """register the suite's custom markers so ``--strict-markers`` stays usable"""
    config.addinivalue_line(
        "markers",
        "slow: takes roughly a second or more. Deselect with -m 'not slow' for a "
        "fast run that still covers every module.",
    )


def pytest_collection_modifyitems(config, items):
    """Mark the known-expensive cases ``slow``

    ``-m "not slow"`` still exercises every test module and every code path that is not specific to a slow fixture
    """
    for item in items:
        original = getattr(item, "originalname", None) or item.name
        if original in _SLOW_TESTS or any(c in item.name for c in _SLOW_CRYSTALS):
            item.add_marker(pytest.mark.slow)
