# Generation benchmark rerun on Linux (v1 + v2)

These are instructions for the person and the Claude session that run the benchmark on the Ubuntu x86-64
machine. The goal is to repeat the M1 Max benchmark with the same code, the same inputs and the same protocol,
so that v1 and v2 timings from this machine can be compared with each other. The M1 Max timings are not
compared with these. The only cross-machine check is that every multimer count is the same.

Do not edit `scripts/benchmark_driver.py`, `scripts/benchmark_generate*.py`, the CIFs, or the code under test.
If something fails, stop and report it rather than working around it. A silent workaround changes the benchmark.

## What is fixed

| Item | Value |
|---|---|
| v2 code under test | commit `fee5bc6` (`origin/v2`); `src/` is unchanged up to `v2` HEAD |
| v1 code under test | commit `080421e` (`origin/master`), package `crystalatte/` at the repo root |
| Driver and worker scripts | branch `bench/v2-generation-timing` (this checkout) |
| CIFs | 26 files under `tests/` on `origin/v2` (checksums in `cifs_v2.sha256`) |
| v1-only CIF copies | `v1_cifs/`: ritonavir_c2 as one ordered state (A1 B1); pentaerythritol with its symmetry operations written out (v1 cannot use a space-group name alone). Checksums are in `cifs_v1_copies.sha256` |
| Python environment | `environment.yml` (versions match the macOS `cle` env). v1 and v2 run on the same interpreter: `run_all.sh` resolves `python` once, checks that it is the `cle-bench` env with the pinned versions, and passes it to both drivers. Each driver starts its workers with that same interpreter |
| Process | one fresh process per measurement, single-threaded BLAS (driver sets `OMP/OPENBLAS/MKL_NUM_THREADS=1`) |
| v2 grid | all 26 crystals, N = 2–5, all cutoffs in `CUTOFFS`, 3 repeats |
| v1 grid | 21 crystals: the 20 from the M1 Max plus carbon_dioxide (still no pyrazole, fomepizole, phenol, pentacene or cholesterol), N = 2–5, all cutoffs, 3 repeats |
| 10 min rule | after a run longer than 600 s, or a failed run, larger cutoffs for that crystal and N are skipped (any remaining repeats of that cutoff too). Exception: v1 ritonavir_c2 at N=5 runs without the rule, up to R = 10 Å, as on the M1 Max |

## Deliberate changes from the M1 Max protocol

1. **No time limit for v1** (`--timeout inf`). On the M1 Max, six v1 runs at N=5, R=10 Å hit the 1 h limit:
   urea, pentaerythritol, gamma_glycine, l_alanine, coronene and ritonavir_c2. They now run to completion, which can
   take many hours each. v2 keeps the 1 h limit; its only timeout was carbon_dioxide N=5 R=15.
2. **Larger memory limit.** `MAX_GB` defaults to 80% of RAM (it was 16 GB). On the M1 Max, five v2 runs at N=5,
   R=15 Å hit the limit. Their output alone is about 24 GB (about 7.4 KB per multimer, 3.25 M multimers). Here they
   should finish if the machine has enough RAM. Linux has no memory compression, so the `ps rss` peak in
   `peak_gb` is a real number here. On macOS it was not.
3. **carbon_dioxide added to v1.** It was left out on the M1 Max, so v1 has no reference for it. A check on the M1 Max
   found v1 counts equal to v2's at N=2 and N=3 up to R=10 Å.
4. **One Python for both codes.** On the M1 Max, v1 ran in the `cle` env (Python 3.13). v2 probably ran in base
   miniconda (Python 3.11, numpy 2.1). Here both use `cle-bench`, so their timings can be compared with each other.

Because the 10 min rule depends on speed, a faster or slower CPU can change which large cutoffs get run. That is
expected. The count check only compares cells measured on both machines, and it lists the other cells.

## Setup

```bash
git clone git@github.com:j-yon/CrystaLattE.git ~/CrystaLattE        # REPO: CIF root, stays on v2
cd ~/CrystaLattE && git checkout v2
git worktree add ~/cle-bench bench/v2-generation-timing              # scripts + this bundle
git worktree add --detach ~/cle-v2src fee5bc6                         # V2SRC
git worktree add --detach ~/cle-v1src 080421e                         # V1SRC

conda env create -f ~/cle-bench/benchmark_rerun/environment.yml
conda activate cle-bench
```

If conda cannot solve `environment.yml`, relax `psi4` first (v1 only imports it). Relax `python` second. Do not
change the numpy, scipy or spglib versions without asking the user. Do not `pip install` crystalatte into the env,
because each code is chosen with `PYTHONPATH`, and an installed copy could be imported instead.

## Run

Run everything from `~/cle-bench` inside `tmux` or with `nohup`. The v1 stage can take days. Keep the machine
free of other heavy jobs while timings are recorded.

```bash
cd ~/cle-bench
export REPO=~/CrystaLattE V2SRC=~/cle-v2src V1SRC=~/cle-v1src
benchmark_rerun/run_all.sh check                 # preflight only: commits, clean trees, CIF checksums, import paths
tmux new -s bench 'benchmark_rerun/run_all.sh all; exec bash'
```

- Results go to `~/cle_benchmarks/<hostname>/{v2_fee5bc6,v1_080421e}/timings.jsonl`, with each stage's
  `driver.log` beside it. `machine_<timestamp>.txt` records the interpreter used for both codes, the CPU, RAM, OS, script commit and `conda list`.
  Override the paths with `OUT=...` and `MAX_GB=...`.
- The run resumes. After an interruption, run the same command again and finished measurements are skipped.
  `run_all.sh v2` and `run_all.sh v1` run a single stage.
- Watch progress with `tail -f ~/cle_benchmarks/$(hostname -s)/v1_080421e/driver.log`.
- If a run fails with `status: "error"` because of the environment (for example an import error), fix the
  environment. Then rerun that stage by hand, adding `--retry-errors` to the same driver command. Errors from the
  code itself are results: report them, and do not retry.

## Verify

Check each code against its M1 Max reference after each stage, and again at the end:

```bash
OUTDIR=~/cle_benchmarks/$(hostname -s)
python benchmark_rerun/compare_counts.py benchmark_rerun/reference/v2_fee5bc6_m1max.jsonl $OUTDIR/v2_fee5bc6/timings.jsonl
python benchmark_rerun/compare_counts.py benchmark_rerun/reference/v1_080421e_m1max.jsonl $OUTDIR/v1_080421e/timings.jsonl
```

`PASS` means that every (crystal, N, R) measured on both machines has the same unique count and summed
multiplicity, and that repeats agree. Any `COUNT MISMATCH` or `REPEATS DISAGREE` means the benchmark is not the
same. Stop and report it. The lists of cells that are ok on only one machine are expected to contain:

- v2: the N=5 R=15 cells that hit the memory limit on the M1 Max (pyrazole, fomepizole, gamma_glycine,
  l_alanine, phenol)
- v1: the six N=5 R=10 cells that timed out on the M1 Max, and every carbon_dioxide cell (no v1 reference)
- cells that the 10 min rule keeps or drops differently on this machine

Do not compare v1 with v2 counts here. They are known to differ for hexamethylbenzene, where v1 is wrong, and by one
cluster at the cutoff edge for adamantane N=2 R=50 and ethyl_carbamate N=2 R=40.

## Report back

Report the paths of both `timings.jsonl` files and the `machine_*.txt` file, the output of both comparisons,
the total wall time per stage, and every non-ok record.
