"""Run the v2 generation timing grid one process at a time, appending one JSON record per run

Reruns resume from ``--out``: finished measurements are not repeated. Run from the repository root::

    PYTHONPATH=src python scripts/benchmark_driver.py --out ~/cle_benchmarks/v2_1a042c5/timings.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CRYSTALS = {
    "carbon_dioxide": "tests/x23/carbon_dioxide/carbon_dioxide.cif",
    "urea": "tests/x23/urea/urea.cif",
    "triazine": "tests/x23/triazine/triazine.cif",
    "trioxane": "tests/x23/trioxane/trioxane.cif",
    "benzene": "tests/x23/benzene/benzene.cif",
    "ethyl_carbamate": "tests/x23/ethyl_carbamate/ethyl_carbamate.cif",
    "14_cyclohexanedione": "tests/x23/14_cyclohexanedione/14_cyclohexanedione.cif",
    "hexamine": "tests/x23/hexamine/hexamine.cif",
    "anthracene": "tests/x23/anthracene/anthracene.cif",
    "adamantane": "tests/x23/adamantane/adamantane.cif",
    "ritonavir_c2": "tests/other/ritonavir_c2/ritonavir_c2.cif",
    "pyrazole": "tests/x23/pyrazole/pyrazole.cif",
    "fomepizole": "tests/other/fomepizole/fomepizole.cif",
    "gamma_glycine": "tests/other/gamma_glycine/gamma_glycine.cif",
    "l_alanine": "tests/other/l_alanine/l_alanine.cif",
    "phenol": "tests/other/phenol/phenol.cif",
    "aspirin": "tests/other/aspirin/aspirin.cif",
    "pentaerythritol": "tests/other/pentaerythritol/pentaerythritol.cif",
    "beta_hmx": "tests/other/beta_hmx/beta_hmx.cif",
    "hexamethylbenzene": "tests/other/hexamethylbenzene/hexamethylbenzene.cif",
    "ibuprofen": "tests/other/ibuprofen/ibuprofen.cif",
    "pentacene": "tests/other/pentacene/pentacene.cif",
    "sucrose": "tests/other/sucrose/sucrose.cif",
    "rubrene": "tests/other/rubrene/rubrene.cif",
    "coronene": "tests/other/coronene/coronene.cif",
    "cholesterol": "tests/other/cholesterol/cholesterol.cif",
}
CUTOFFS = {
    2: [8, 10, 12, 15, 20, 30, 40, 50, 60],
    3: [6, 8, 10, 12, 15, 20, 25, 30],
    4: [4, 5, 6, 8, 10, 12, 15, 20],
    5: [3, 4, 5, 6, 8, 10, 12, 15],
}
POLL_SECONDS = 2.0
SINGLE_THREAD = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def resident_gb(pid: int) -> float:
    """Return a process's resident memory in GB, or 0 if it has already exited"""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024**2 if out.strip() else 0.0


def measure(worker: Path, cif: str, N: int, R: float, timeout: float, max_gb: float) -> dict:
    """Run one measurement in a fresh process and return its result fields"""
    env = {**os.environ, **SINGLE_THREAD}
    peak = 0.0
    with tempfile.TemporaryFile("w+") as stderr:
        proc = subprocess.Popen(
            [sys.executable, str(worker), cif, str(N), str(R)],
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            env=env,
        )
        start = time.monotonic()
        status = None
        while status is None:
            try:
                proc.wait(POLL_SECONDS)
                status = "ok" if proc.returncode == 0 else "error"
            except subprocess.TimeoutExpired:
                peak = max(peak, resident_gb(proc.pid))
                if peak > max_gb:
                    status = "memory_limit"
                elif time.monotonic() - start > timeout:
                    status = "timeout"
                if status is not None:
                    proc.kill()
                    proc.wait()
        stdout = proc.stdout.read()
        stderr.seek(0)
        message = stderr.read()[-500:]

    result = {"status": status, "peak_gb": round(peak, 2)}
    if status == "ok":
        result.update(json.loads(stdout))
    elif status == "error":
        result["message"] = message
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="JSON Lines results file")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds allowed per run")
    parser.add_argument("--slow", type=float, default=600.0, help="seconds beyond which larger cutoffs are skipped")
    parser.add_argument("--max-gb", type=float, default=16.0, help="resident memory allowed per run")
    parser.add_argument("--only", nargs="*", default=list(CRYSTALS), help="crystal names to run")
    parser.add_argument("--N", nargs="*", type=int, default=list(CUTOFFS), dest="orders")
    parser.add_argument("--max-R", type=float, default=float("inf"), help="largest cutoff to run")
    parser.add_argument("--cif-root", type=Path, default=Path("."), help="directory the CIF paths are relative to")
    parser.add_argument("--skip", nargs="*", default=[], help="N:R pairs not to run, e.g. 4:20 5:15")
    parser.add_argument("--worker", type=Path, default=Path(__file__).with_name("benchmark_generate.py"), help="script timing one run")
    parser.add_argument("--code-commit", help="commit to record; defaults to the last commit touching src")
    parser.add_argument("--retry-errors", action="store_true", help="rerun measurements recorded as errors")
    args = parser.parse_args()
    skipped = {tuple(float(part) for part in pair.split(":")) for pair in args.skip}

    commit = args.code_commit or subprocess.run(["git", "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True).stdout.strip()
    args.out = args.out.expanduser()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    finished = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            record = json.loads(line)
            if args.retry_errors and record["status"] == "error":
                continue
            finished[record["crystal"], record["N"], record["R"], record["repeat"]] = record

    stopped = set()
    for record in finished.values():
        if record["status"] != "ok" or record.get("seconds", 0) > args.slow:
            stopped.add((record["crystal"], record["N"]))

    # ascending cutoffs across all crystals, so every crystal has cheap data before any costly run
    for N in args.orders:
        for R in (r for r in CUTOFFS[N] if r <= args.max_R and (N, r) not in skipped):
            for name in args.only:
                if (name, N) in stopped:
                    continue
                for repeat in range(args.repeats):
                    if (name, N, R, repeat) in finished:
                        continue
                    result = measure(args.worker, str(args.cif_root / CRYSTALS[name]), N, R, args.timeout, args.max_gb)
                    record = {"crystal": name, "N": N, "R": R, "repeat": repeat, "code_commit": commit, **result}
                    with args.out.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
                    if result["status"] != "ok" or result["seconds"] > args.slow:
                        stopped.add((name, N))
                        break


if __name__ == "__main__":
    main()
