#!/usr/bin/env bash
# Summarise refine_ucsf_water.sh runs into one CSV row per dataset.
#
# For each <out_root>/<dataset> with a finished stage2_water log, write the full
# step-by-step table of both stages beside each log (<log>.steps.csv) and emit:
#   header_*        R-factors and work/test reflection counts from the Dimple REMARK 3
#   n_refl, pct_free  reflections Phenix actually used (from the stage-2 log)
#   *_bss           stage 1 after bulk-solvent/scaling: the fair starting point
#   *_rigid         end of stage 1 (rigid body)
#   *_cycle_last    stage 2 after the last macrocycle, before the final water filter
#   *_final         end of stage 2
#   d_r_*           final minus header
#   rmsd_*, max_shift, n_flips  protein shift of the final model vs the Dimple input
#                   (protein_rmsd.py)
#
# usage: collect_ucsf_water.sh [out_root] [out.csv]
set -uo pipefail

BASE_DIR="/mnt/diffuse-shared/users/dorismai/water_datasets/ucsf_nsp3_mac1_P43"
OUT_ROOT="${1:-${BASE_DIR}/phenix_water_rerefine}"
out="${2:-${OUT_ROOT}/summary.csv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Every row of the "stage r-work r-free ..." table as stage,r_work,r_free,n_water.
step_table() {
    awk '
        / stage r-work r-free/ { inblock = 1; next }
        !inblock { next }
        /^[[:space:]]*-+[[:space:]]*$/ { inblock = 0; next }
        {
            colon = index($0, ":")
            if (colon == 0) next
            stage = substr($0, 1, colon - 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", stage)
            n = split(substr($0, colon + 1), f)
            if (n < 8) next
            print stage "," f[1] "," f[2] "," f[8]
        }
    ' "$1"
}

row_for() { awk -F, -v s="$2" '$1 == s { print $2 "," $3 "," $4 }' "$1"; }
remark() { sed -n "s/^REMARK   3   *$2 *: *\([0-9.]*\).*/\1/p" "$1" | head -1; }

echo "dataset,header_r_work,header_r_free,header_n_work,header_n_free,n_refl,pct_free,r_work_bss,r_free_bss,r_work_rigid,r_free_rigid,r_work_cycle_last,r_free_cycle_last,n_water_cycle_last,r_work_final,r_free_final,n_water_final,d_r_work,d_r_free,rmsd_ca,rmsd_heavy,max_shift,n_flips" > "$out"

found=0
for dataset_dir in "${OUT_ROOT}"/UCSF-P*/; do
    dataset="$(basename "$dataset_dir")"
    stage1_log="${dataset_dir}stage1_rigid/${dataset}_stage1_rigid_001.log"
    stage2_log="${dataset_dir}stage2_water/${dataset}_stage2_water_001.log"
    if ! grep -q "Final R-work" "$stage2_log" 2>/dev/null; then
        echo "skip: ${dataset} stage 2 not finished" >&2
        continue
    fi

    step_table "$stage1_log" > "${stage1_log}.steps.csv"
    step_table "$stage2_log" > "${stage2_log}.steps.csv"
    bss="$(row_for "${stage1_log}.steps.csv" 1_bss | cut -d, -f1,2)"
    rigid="$(row_for "${stage1_log}.steps.csv" end | cut -d, -f1,2)"
    final="$(row_for "${stage2_log}.steps.csv" end)"
    cycle_last="$(grep -v '^end,' "${stage2_log}.steps.csv" | tail -1 | cut -d, -f2-)"

    pdb="${dataset_dir}inputs/${dataset}_dimple.pdb"
    header_r_work="$(remark "$pdb" 'R VALUE            (WORKING SET)')"
    header_r_free="$(remark "$pdb" 'FREE R VALUE')"
    header_n_work="$(remark "$pdb" 'NUMBER OF REFLECTIONS')"
    header_n_free="$(remark "$pdb" 'FREE R VALUE TEST SET COUNT')"
    refl="$(sed -n 's/.*n_refl\.=\([0-9]*\) (all), *\([0-9.]*\) *% free.*/\1,\2/p' "$stage2_log" | head -1)"

    deltas="$(awk -F, -v hw="$header_r_work" -v hf="$header_r_free" \
        '{ printf "%.4f,%.4f", $1 - hw, $2 - hf }' <<< "$final")"

    shift_stats="$(phenix.python "${SCRIPT_DIR}/protein_rmsd.py" "$dataset_dir")"

    echo "${dataset},${header_r_work},${header_r_free},${header_n_work},${header_n_free},${refl},${bss},${rigid},${cycle_last},${final},${deltas},${shift_stats}" >> "$out"
    found=$((found + 1))
done

echo "wrote $out: $found datasets" >&2
