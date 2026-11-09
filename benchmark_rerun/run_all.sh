#!/bin/bash
# Full v1 + v2 generation benchmark rerun on one Linux machine. See benchmark_rerun/HANDOFF.md.
# Resumable: the driver skips measurements already in the output files, so rerunning continues where it stopped.
#
#   REPO=~/CrystaLattE V2SRC=~/cle-v2src V1SRC=~/cle-v1src benchmark_rerun/run_all.sh [check|v2|v1|all]
#
# Run from the root of the bench/v2-generation-timing checkout, inside the activated cle-bench env.
set -euo pipefail

STAGE=${1:-all}
REPO=${REPO:?set REPO to the v2 checkout that holds tests/ (CIF root)}
V2SRC=${V2SRC:?set V2SRC to a checkout of fee5bc6}
V1SRC=${V1SRC:?set V1SRC to a checkout of 080421e}
BUNDLE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/cle_benchmarks/$(hostname -s)}
MEM_GB=$(awk '/MemTotal/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo)
MAX_GB=${MAX_GB:-$((MEM_GB * 80 / 100))}

# v1 crystals: the M1 Max queue plus carbon_dioxide. ritonavir_c2 and pentaerythritol read edited copies from v1_cifs/.
COPIES="ritonavir_c2 pentaerythritol"
OTHERS="carbon_dioxide urea trioxane benzene hexamine aspirin anthracene adamantane sucrose rubrene \
triazine gamma_glycine l_alanine ethyl_carbamate 14_cyclohexanedione \
beta_hmx hexamethylbenzene ibuprofen coronene"

# One interpreter for both codes. The driver starts every worker with its own interpreter, so this is the only Python used.
PY=$(command -v python)
# Must match environment.yml
EXPECTED_VERSIONS="python=3.13.12 numpy=2.4.3 scipy=1.17.1 spglib=2.7.0"

die() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }

check() {
    [[ ${CONDA_DEFAULT_ENV:-} == cle-bench ]] || die "activate the cle-bench env first"
    [[ $PY == "$CONDA_PREFIX"/bin/python ]] || die "python is $PY, not the cle-bench interpreter"
    local versions
    versions=$("$PY" -c "import platform, numpy, scipy, spglib; print(f'python={platform.python_version()} numpy={numpy.__version__} scipy={scipy.__version__} spglib={spglib.__version__}')")
    [[ $versions == "$EXPECTED_VERSIONS" ]] || die "package versions are '$versions', expected '$EXPECTED_VERSIONS'"
    "$PY" -c "import psi4" || die "psi4 must be importable for v1"
    [[ -f scripts/benchmark_driver.py ]] || die "run from the bench/v2-generation-timing checkout root"
    [[ $(git -C "$V2SRC" rev-parse --short=7 HEAD) == fee5bc6 ]] || die "V2SRC is not at fee5bc6"
    [[ $(git -C "$V1SRC" rev-parse --short=7 HEAD) == 080421e ]] || die "V1SRC is not at 080421e"
    git -C "$V2SRC" diff --quiet HEAD -- src || die "V2SRC has local changes in src/"
    git -C "$V1SRC" diff --quiet HEAD -- crystalatte || die "V1SRC has local changes in crystalatte/"
    (cd "$REPO" && sha256sum --quiet -c "$BUNDLE/cifs_v2.sha256") || die "CIF checksums differ in REPO"
    (cd "$BUNDLE/v1_cifs" && sha256sum --quiet -c "$BUNDLE/cifs_v1_copies.sha256") || die "v1 CIF copy checksums differ"
    local where
    where=$(PYTHONPATH="$V2SRC/src" "$PY" -c "import crystalatte; print(crystalatte.__file__)")
    [[ $where == "$(cd "$V2SRC" && pwd)"/src/* ]] || die "v2 import resolves to $where"
    where=$(PYTHONPATH="$V1SRC" "$PY" -c "import crystalatte.crystalatte as c; print(c.__file__)")
    [[ $where == "$(cd "$V1SRC" && pwd)"/* ]] || die "v1 import resolves to $where"
    echo "preflight ok: $PY ($versions), fee5bc6, 080421e, 28 CIFs verified, imports resolve correctly, MAX_GB=$MAX_GB of ${MEM_GB} GB"
}

record_machine() {
    mkdir -p "$OUT"
    {
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "max_gb: $MAX_GB"
        echo "python (v1 and v2): $PY"
        "$PY" -VV
        uname -a
        lsb_release -ds 2>/dev/null || true
        lscpu
        free -g
        echo "--- bench scripts commit: $(git rev-parse HEAD)"
        echo "--- conda list"
        conda list
    } > "$OUT/machine_$(date +%Y%m%dT%H%M%S).txt"
}

run_v2() {
    mkdir -p "$OUT/v2_fee5bc6"
    # Same rules as the M1 Max run (1 h timeout, 10 min rule, 3 repeats); only the memory limit is raised.
    PYTHONPATH="$V2SRC/src" "$PY" scripts/benchmark_driver.py \
        --code-commit fee5bc6 --cif-root "$REPO" --max-gb "$MAX_GB" \
        --out "$OUT/v2_fee5bc6/timings.jsonl" >> "$OUT/v2_fee5bc6/driver.log" 2>&1
}

run_v1() {
    mkdir -p "$OUT/v1_080421e"
    # No per-run time limit for v1. The 10 min rule still stops larger cutoffs after a slow run.
    PYTHONPATH="$V1SRC" "$PY" scripts/benchmark_driver.py --worker scripts/benchmark_generate_v1.py \
        --code-commit 080421e --timeout inf --max-gb "$MAX_GB" \
        --out "$OUT/v1_080421e/timings.jsonl" "$@" >> "$OUT/v1_080421e/driver.log" 2>&1
}

v1_all() {
    # Every order across all crystals before the next, as on the M1 Max
    for N in 2 3 4 5; do
        if [[ $N == 5 ]]; then
            # On the M1 Max, ritonavir N=5 ran without the 10 min rule up to R=10 A
            run_v1 --cif-root "$BUNDLE/v1_cifs" --only ritonavir_c2 --N 5 --slow inf --max-R 10
            run_v1 --cif-root "$BUNDLE/v1_cifs" --only pentaerythritol --N 5
        else
            run_v1 --cif-root "$BUNDLE/v1_cifs" --only $COPIES --N "$N"
        fi
        run_v1 --cif-root "$REPO" --only $OTHERS --N "$N"
    done
}

check
case $STAGE in
    check) ;;
    v2) record_machine; run_v2 ;;
    v1) record_machine; v1_all ;;
    all) record_machine; run_v2; v1_all ;;
    *) die "unknown stage $STAGE" ;;
esac
echo "done: $STAGE"
