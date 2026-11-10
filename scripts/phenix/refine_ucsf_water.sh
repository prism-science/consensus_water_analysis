#!/bin/bash
# Two-stage Phenix re-refinement of one UCSF NSP3 Mac1 (P43) Dimple dataset with
# automatic ordered-solvent placement, parameterised to match Dimple's REFMAC run
# (isotropic individual B) so the final R-factors are comparable to the header ones.
#
#   stage1_rigid  rigid body only, 1 macrocycle (2-3 changed R-free by <0.001 on
#                 P0067/P0068/P0070), no water picking
#   stage2_water  individual sites + isotropic individual ADP, 8 macrocycles, water
#                 picking every macrocycle with Phenix's default peak/distance
#                 criteria, water occupancies fixed at 1.0
#
# The Dimple PDB/MTZ are copied (read-only) into <out_root>/<dataset>/inputs and both
# stages read only those copies. The MTZ FreeR_flag column follows the CCP4/REFMAC
# convention (0 = test set), so test_flag_value=0 is passed explicitly.
#
# RIDING_H=1 adds hydrogens (phenix.ready_set) to the stage-1 model and refines them as
# riding in stage 2, mirroring REFMAC's default riding H; everything else is unchanged.
# It defaults out_root to a separate tree so the no-H results are never overwritten.
#
# HIGH_RESOLUTION=<d> truncates the data to d A in both stages (everything else is
# unchanged) and appends _d<d> to the default out_root.
#
# usage: [RIDING_H=1] [HIGH_RESOLUTION=d] refine_ucsf_water.sh <dataset> [out_root]

DATASET="${1:?usage: [RIDING_H=1] [HIGH_RESOLUTION=d] refine_ucsf_water.sh <dataset> [out_root]}"
BASE_DIR="/mnt/diffuse-shared/users/dorismai/water_datasets/ucsf_nsp3_mac1_P43"
RIDING_H="${RIDING_H:-0}"
HIGH_RESOLUTION="${HIGH_RESOLUTION:-}"
if [ "$RIDING_H" = 1 ]; then
    DEFAULT_OUT_ROOT="${BASE_DIR}/phenix_water_rerefine_riding_h"
else
    DEFAULT_OUT_ROOT="${BASE_DIR}/phenix_water_rerefine"
fi
if [ -n "$HIGH_RESOLUTION" ]; then
    DEFAULT_OUT_ROOT="${DEFAULT_OUT_ROOT}_d${HIGH_RESOLUTION}"
fi
OUT_ROOT="${2:-${DEFAULT_OUT_ROOT}}"
STAGE1_CYCLES=1
STAGE2_CYCLES=8

SOURCE_PDB="${BASE_DIR}/${DATASET}/${DATASET}_dimple.pdb"
SOURCE_MTZ="${BASE_DIR}/${DATASET}/${DATASET}_dimple.mtz"
for source_file in "$SOURCE_PDB" "$SOURCE_MTZ"; do
    if [ ! -f "$source_file" ]; then
        echo "error: missing input: $source_file" >&2
        exit 1
    fi
done

OUT_DIR="${OUT_ROOT}/${DATASET}"
INPUT_DIR="${OUT_DIR}/inputs"
mkdir -p "$INPUT_DIR"
INPUT_PDB="${INPUT_DIR}/${DATASET}_dimple.pdb"
INPUT_MTZ="${INPUT_DIR}/${DATASET}_dimple.mtz"
if [ ! -f "$INPUT_PDB" ]; then
    cp "$SOURCE_PDB" "$INPUT_PDB"
    chmod a-w "$INPUT_PDB"
fi
if [ ! -f "$INPUT_MTZ" ]; then
    cp "$SOURCE_MTZ" "$INPUT_MTZ"
    chmod a-w "$INPUT_MTZ"
fi

# Shared by both stages: the deposited data, the REFMAC free set, and no outlier
# rejection so the work/test reflection counts equal the PDB header's.
COMMON_ARGS=(
    "data_manager.miller_array.labels.name=F,SIGF"
    "data_manager.miller_array.labels.name=FreeR_flag"
    "data_manager.fmodel.xray_data.r_free_flags.test_flag_value=0"
    "data_manager.fmodel.xray_data.r_free_flags.ignore_r_free_flags=false"
    "data_manager.fmodel.xray_data.outliers_rejection=false"
    "output.serial=1"
    "output.overwrite=true"
    "write_model_cif_file=true"
    "write_final_pdb_file=true"
    "write_maps=false"
)
if [ -n "$HIGH_RESOLUTION" ]; then
    COMMON_ARGS+=("data_manager.fmodel.xray_data.high_resolution=${HIGH_RESOLUTION}")
fi

STAGE1_ARGS=(
    "strategy=rigid_body"
    "main.number_of_macro_cycles=${STAGE1_CYCLES}"
    "main.ordered_solvent=false"
)

# strategy omits occupancies and ordered_solvent.refine_occupancies is off: both are
# needed to keep picked waters at occupancy 1.0.
STAGE2_ARGS=(
    "strategy=individual_sites+individual_adp"
    "refine.adp.individual.isotropic=all"
    "main.number_of_macro_cycles=${STAGE2_CYCLES}"
    "main.ordered_solvent=true"
    "ordered_solvent.mode=every_macro_cycle"
    "ordered_solvent.new_solvent=isotropic"
    "ordered_solvent.refine_occupancies=false"
)

run_stage() {
    local stage="$1" model="$2"
    shift 2
    local stage_dir="${OUT_DIR}/${stage}"
    local prefix="${DATASET}_${stage}"
    if [ -f "${stage_dir}/${prefix}_001.log" ] && grep -q "Final R-work" "${stage_dir}/${prefix}_001.log"; then
        echo "${stage}: already refined, skipping"
        return 0
    fi
    mkdir -p "$stage_dir"
    local command=(phenix.refine "$model" "$INPUT_MTZ" "${COMMON_ARGS[@]}" "$@" "output.prefix=${prefix}")
    printf '%q ' "${command[@]}" > "${stage_dir}/command.txt"
    echo >> "${stage_dir}/command.txt"
    echo "${DATASET} ${stage}: $(date -Is)"
    (cd "$stage_dir" && "${command[@]}" > "${prefix}_stdout.txt" 2>&1)
    if ! grep -q "Final R-work" "${stage_dir}/${prefix}_001.log" 2>/dev/null; then
        echo "error: ${stage} did not finish; see ${stage_dir}/${prefix}_stdout.txt" >&2
        return 1
    fi
    grep "Final R-work" "${stage_dir}/${prefix}_001.log"
}

run_stage stage1_rigid "$INPUT_PDB" "${STAGE1_ARGS[@]}" || exit 1
STAGE1_MODEL="${OUT_DIR}/stage1_rigid/${DATASET}_stage1_rigid_001.cif"
if [ "$RIDING_H" = 1 ]; then
    # ready_set rejects the stage-1 CIF (hyphenated data block name), so it reads the PDB.
    HYDROGEN_PREFIX="${DATASET}_stage1_rigid_001_H"
    HYDROGEN_MODEL="${OUT_DIR}/stage1_rigid/${HYDROGEN_PREFIX}.pdb"
    if [ ! -f "$HYDROGEN_MODEL" ]; then
        (cd "${OUT_DIR}/stage1_rigid" && phenix.ready_set "${STAGE1_MODEL%.cif}.pdb" \
            "output_file_name=${HYDROGEN_PREFIX}" > ready_set_stdout.txt 2>&1) || exit 1
    fi
    STAGE1_MODEL="$HYDROGEN_MODEL"
    # Riding H take their parent's B, so they must be outside the individual-ADP set;
    # the later isotropic selection overrides "all" above.
    STAGE2_ARGS+=("hydrogens.refine=riding" "refine.adp.individual.isotropic=not element H")
fi
run_stage stage2_water "$STAGE1_MODEL" "${STAGE2_ARGS[@]}" || exit 1
