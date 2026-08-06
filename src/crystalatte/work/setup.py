from collections import defaultdict
from pathlib import Path

from ..core.multimer import Multimer
from ..io.writers import get_writer

_NAMES = {1: "monomer", 2: "dimer", 3: "trimer", 4: "tetramer", 5: "pentamer"}


def _order_name(N: int) -> str:
    return _NAMES.get(N, f"{N}mer")


def write_inputs(
    multimers: list[Multimer],
    output_dir: str | Path,
    software: str,
    config: dict,
) -> None:
    output_dir = Path(output_dir)
    writer = get_writer(software)

    by_order: dict[int, list[Multimer]] = defaultdict(list)
    for m in multimers:
        by_order[len(m.monomers)].append(m)

    for N, group in sorted(by_order.items()):
        order_dir = output_dir / f"{_order_name(N)}s"
        order_dir.mkdir(parents=True, exist_ok=True)
        for idx, multimer in enumerate(group, start=1):
            path = order_dir / f"{_order_name(N)}_{idx:03d}.inp"
            writer.write(path, multimer.monomers, config)
