"""Check that a rerun found the same multimers as a reference run of the same code

Compares unique and summed-multiplicity counts per (crystal, N, R). Timings are not compared.
Exits 1 if any count differs. Run with the reference first::

    python benchmark_rerun/compare_counts.py benchmark_rerun/reference/v2_fee5bc6_m1max.jsonl ~/cle_benchmarks/<host>/v2_fee5bc6/timings.jsonl
"""

import collections
import json
import sys


def load(path: str) -> tuple[dict, dict]:
    """Return counts per cell from ok records, and the statuses seen per cell"""
    counts = collections.defaultdict(set)
    statuses = collections.defaultdict(set)
    for line in open(path):
        record = json.loads(line)
        cell = record["crystal"], record["N"], record["R"]
        statuses[cell].add(record["status"])
        if record["status"] == "ok":
            counts[cell].add((record["unique"], record["multiplicity"]))
    return counts, statuses


def main() -> None:
    ref_counts, ref_status = load(sys.argv[1])
    new_counts, new_status = load(sys.argv[2])
    failed = False

    for name, counts in (("reference", ref_counts), ("rerun", new_counts)):
        for cell, values in sorted(counts.items()):
            if len(values) > 1:
                failed = True
                print(f"REPEATS DISAGREE in {name}: {cell} {sorted(values)}")

    shared = sorted(ref_counts.keys() & new_counts.keys())
    mismatched = [cell for cell in shared if ref_counts[cell] != new_counts[cell]]
    for cell in mismatched:
        failed = True
        print(f"COUNT MISMATCH {cell}: reference {sorted(ref_counts[cell])} rerun {sorted(new_counts[cell])}")

    missing = sorted(ref_counts.keys() - new_counts.keys())
    gained = sorted(new_counts.keys() - ref_counts.keys())
    print(f"\n{len(shared)} cells measured in both, {len(mismatched)} with different counts")
    print(f"{len(missing)} cells ok in reference but not in rerun:")
    for cell in missing:
        print(f"  {cell} rerun status: {sorted(new_status.get(cell, {'not run'}))}")
    print(f"{len(gained)} cells ok in rerun but not in reference:")
    for cell in gained:
        print(f"  {cell} reference status: {sorted(ref_status.get(cell, {'not run'}))} counts {sorted(new_counts[cell])}")

    print("\nFAIL" if failed else "\nPASS: every shared cell has identical counts")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
