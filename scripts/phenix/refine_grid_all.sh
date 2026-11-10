#!/bin/bash
# Drive the full 2 x 2 x 2 refinement grid for one PDB entry and collect the stats.
#
#   waters kept | stripped  x  ordered_solvent true | false  x  default | rigid
#
# = 8 trials, each starting from the entry's own deposited model against its own
# deposited data. Trials are independent, so they run JOBS at a time via xargs.
#
# usage: PDBID=3b3a [NUMCYCLE=10] [JOBS=4] [NPROC=1] refine_grid_all.sh
#    or: refine_grid_all.sh 3b3a
#
# Inputs under ALL_PDB_REDO_DIR are read-only; every artifact is written under
# $DATA_DIR/<cohort>_grid_phenix/.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

PDBID="${1:-${PDBID:-3b3a}}"
PDBID="$(printf '%s' "$PDBID" | tr '[:upper:]' '[:lower:]')"

export NPROC="${NPROC:-1}"
export NUMCYCLE="${NUMCYCLE:-10}"
export MAPCUTOFF="${MAPCUTOFF:-3.5}"
export JOBS="${JOBS:-4}"
export COHORT_ID="${COHORT_ID:-$PDBID}"

OUT_DIR="${DATA_DIR}/${COHORT_ID}_grid_phenix/refinement_results/${PDBID}"
PREP_DIR="${OUT_DIR}/prepared_models"
mkdir -p "$PREP_DIR"

START_CIF="${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.cif"
[ -f "$START_CIF" ] || { echo "error: no starting model: $START_CIF" >&2; exit 1; }
[ -f "${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.mtz" ] || { echo "error: no mtz for $PDBID" >&2; exit 1; }

if [ -n "$PHENIX_ENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$PHENIX_ENV_PATH"
fi

# Pass 0: prepare both starting models once, serially, before any trial runs — four
# trials share each prepared cif and parallel phenix.pdbtools writes would race.
# pdbtools also re-emits the model without the deposited TLS metadata that
# phenix.refine asserts on.
for water in kept stripped; do
    prepared="${PREP_DIR}/${PDBID}_water${water}.cif"
    [ -f "$prepared" ] && continue
    remove=()
    [ "$water" = "stripped" ] && remove=(remove="resname HOH")
    echo "Preparing ${PDBID} starting model (waters ${water})"
    ( cd "$PREP_DIR" && phenix.pdbtools "$START_CIF" "${remove[@]}" \
        output.file_name="${PDBID}_water${water}.cif" > "${PDBID}_water${water}_pdbtools.log" 2>&1 )
    [ -f "$prepared" ] || { echo "error: pdbtools failed; see ${PREP_DIR}/${PDBID}_water${water}_pdbtools.log" >&2; exit 1; }
done

# Pass 1: the 8 trials, JOBS at a time.
echo "Running 8 trials for ${PDBID} (JOBS=${JOBS}, NPROC=${NPROC}, NUMCYCLE=${NUMCYCLE})"
for water in kept stripped; do
    for ordered in true false; do
        for strategy in default rigid; do
            printf '%s\t%s\t%s\t%s\n' "$PDBID" "$water" "$ordered" "$strategy"
        done
    done
done | xargs -P "$JOBS" -n4 bash "${SCRIPT_DIR}/refine_grid.sh"

# Pass 2: collect starting/final R-work, R-free and n_water into one CSV.
bash "${SCRIPT_DIR}/collect_grid_stats.sh" "$PDBID" "${DATA_DIR}/${COHORT_ID}_grid_phenix/grid_stats.csv"
