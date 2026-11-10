#!/bin/bash
# One cell of the water/solvent/strategy refinement grid for a single PDB entry.
#
# Unlike re-refine.sh (which refines an N x N matrix of target-data x starting-model
# pairs), this refines one entry against its OWN data starting from its OWN deposited
# model — the diagonal case, so no alignment is needed — and sweeps the three binary
# knobs the grid is about:
#
#   WATER    kept | stripped   starting model keeps or drops its deposited waters
#   ORDERED  true | false      refinement.main.ordered_solvent (water picking)
#   STRATEGY default | rigid   which eff template (individual sites vs rigid body)
#
# Normally invoked by refine_grid_all.sh, which sources setup_env.sh, exports the
# phil variables, and prepares the starting models once before fanning out. The
# prepared models are read-only inputs here so parallel trials never race.
#
# usage: refine_grid.sh <pdb_id> <kept|stripped> <true|false> <default|rigid>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "NPROC: ${NPROC} NUMCYCLE: ${NUMCYCLE} MAPCUTOFF: ${MAPCUTOFF}"

PDBID="${1:?usage: refine_grid.sh <pdb_id> <kept|stripped> <true|false> <default|rigid>}"
WATER="${2:?water must be 'kept' or 'stripped'}"
ORDERED="${3:?ordered must be 'true' or 'false'}"
STRATEGY="${4:?strategy must be 'default' or 'rigid'}"

: "${COHORT_ID:?not set — refine_grid_all.sh exports it; set it for standalone runs}"
: "${DATA_DIR:?not set — source scripts/phenix/setup_env.sh first}"

case "$WATER" in kept|stripped) ;; *) echo "error: water must be 'kept' or 'stripped'" >&2; exit 1 ;; esac
case "$ORDERED" in true|false) ;; *) echo "error: ordered must be 'true' or 'false'" >&2; exit 1 ;; esac
case "$STRATEGY" in
    rigid)   EFF_TEMPLATE="${SCRIPT_DIR}/refine_template_rigid.eff" ;;
    default) EFF_TEMPLATE="${SCRIPT_DIR}/refine_template_default.eff" ;;
    *) echo "error: STRATEGY must be 'rigid' or 'default'" >&2; exit 1 ;;
esac

# All 8 trials share one tree so the grid can be collected in a single pass.
OUT_DIR="${DATA_DIR}/${COHORT_ID}_grid_phenix/refinement_results/${PDBID}"
PREP_DIR="${OUT_DIR}/prepared_models"
TRIAL="water${WATER}_ordered${ORDERED}_${STRATEGY}"
OUTPREFIX="${PDBID}_${TRIAL}"
REFINE_DIR="${OUT_DIR}/${TRIAL}"

# Read-only input: the deposited MTZ, referenced by $MTZFILE in the eff templates.
export MTZFILE="${ALL_PDB_REDO_DIR}/${PDBID}/${PDBID}_final.mtz"
[ -f "$MTZFILE" ] || { echo "error: no mtz: $MTZFILE" >&2; exit 1; }

EFF_CIFFILE="${PREP_DIR}/${PDBID}_water${WATER}.cif"
[ -f "$EFF_CIFFILE" ] || { echo "error: prepared model missing (run refine_grid_all.sh): $EFF_CIFFILE" >&2; exit 1; }

echo "================================================"
echo "Refining ${PDBID}: waters=${WATER} ordered_solvent=${ORDERED} strategy=${STRATEGY}"
echo "================================================"

if [ -f "${REFINE_DIR}/${OUTPREFIX}_001.log" ] && grep -q "Final R-work" "${REFINE_DIR}/${OUTPREFIX}_001.log"; then
    echo "Already refined, skipping"
    grep "Final R-work" "${REFINE_DIR}/${OUTPREFIX}_001.log"
    exit 0
fi

mkdir -p "$REFINE_DIR"
cd "$REFINE_DIR" || exit 1
phenix.refine "${EFF_CIFFILE}" "${EFF_TEMPLATE}" "ordered_solvent=${ORDERED}" \
    output.prefix="${OUTPREFIX}" > "${OUTPREFIX}_stdout.txt" 2>&1
status=$?
if [ ! -f "${OUTPREFIX}_001.log" ]; then
    echo "error: ${TRIAL} produced no log (exit ${status}); see ${REFINE_DIR}/${OUTPREFIX}_stdout.txt" >&2
    exit 1
fi
grep "Final R-work" "${OUTPREFIX}_001.log"
