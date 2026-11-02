from collections import defaultdict
from pathlib import Path

from ..core.multimer import Multimer, order_name
from ..io.writers import get_writer


def write_inputs(
    multimers: list[Multimer],
    output_dir: str | Path,
    software: str,
    config: dict,
) -> None:
    """Write input files for a list of multimers"""
    output_dir = Path(output_dir)
    writer = get_writer(software)

    # TODO: maybe improve writing by grouping multimers by order and writing them in subdirectories?
    by_order: dict[int, list[Multimer]] = defaultdict(list)
    for m in multimers:
        by_order[len(m.monomers)].append(m)

    for N, group in sorted(by_order.items()):
        order_dir = output_dir / f"{order_name(N)}s"
        order_dir.mkdir(parents=True, exist_ok=True)
        for idx, multimer in enumerate(group, start=1):
            path = order_dir / f"{order_name(N)}_{idx:03d}.in"
            writer.write(path, multimer.monomers, config)
