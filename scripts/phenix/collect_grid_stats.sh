#!/usr/bin/env bash
# Collect the 8-trial refinement grid for one PDB entry into a single CSV.
#
# For every (waters, ordered_solvent, strategy) trial written by refine_grid.sh, parse
# the phenix log with parse_phenix_log.sh, cache the per-log CSV beside the log, and
# emit the starting and final R-work / R-free / n_water.
#
# Two "starting" points are reported, both straight from the log's step-by-step table:
#   *_start   stage 0     — model as read in, before bulk-solvent correction/scaling
#   *_bss     stage 1_bss — same model after the first bulk-solvent/scaling step
# Stage 0's R-factors are inflated by the missing bulk-solvent term, so 1_bss is the
# number to compare against the deposited R-work/R-free.
#
# usage: collect_grid_stats.sh <pdb_id> <out.csv>
#   DATA_DIR / COHORT_ID env override the data root and cohort name (as elsewhere).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER="$SCRIPT_DIR/parse_phenix_log.sh"
source "$SCRIPT_DIR/setup_env.sh"

[ "$#" -eq 2 ] || { echo "usage: $(basename "$0") <pdb_id> <out.csv>" >&2; exit 1; }
PDBID="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
out="$2"
COHORT_ID="${COHORT_ID:-$PDBID}"
RESULTS_DIR="${DATA_DIR}/${COHORT_ID}_grid_phenix/refinement_results/${PDBID}"

mkdir -p "$(dirname "$out")"
echo "pdb_id,waters,ordered_solvent,strategy,r_work_start,r_free_start,n_water_start,r_work_bss,r_free_bss,r_work_final,r_free_final,n_water_final" > "$out"

# Pull one stage's row out of a parsed per-log CSV.
row_for() { awk -F, -v s="$2" '$1 == s { print $2 "," $3 "," $4 }' "$1"; }

found=0; missing=0
for water in kept stripped; do
    for ordered in true false; do
        for strategy in default rigid; do
            trial="water${water}_ordered${ordered}_${strategy}"
            log="${RESULTS_DIR}/${trial}/${PDBID}_${trial}_001.log"
            if [ ! -f "$log" ]; then
                echo "skip: no log for $trial" >&2
                missing=$((missing + 1))
                continue
            fi

            csv="${log%.log}.csv"
            [ -f "$csv" ] || bash "$PARSER" "$log" > "$csv"

            start="$(row_for "$csv" 0)"
            bss="$(row_for "$csv" 1_bss)"
            final="$(row_for "$csv" end)"
            if [ -z "$start" ] || [ -z "$final" ]; then
                echo "warn: incomplete stats in $csv, skipping $trial" >&2
                missing=$((missing + 1))
                continue
            fi
            # bss carries r_work,r_free,n_water; only the R-factors are of interest.
            bss_r="$(printf '%s' "${bss:-,,}" | cut -d, -f1,2)"

            echo "${PDBID},${water},${ordered},${strategy},${start},${bss_r},${final}" >> "$out"
            found=$((found + 1))
        done
    done
done

echo "wrote $out: $found trials ($missing missing)" >&2
