#!/usr/bin/env bash
# Extract r-work, r-free, and n_water for the 0, 1_bss, N_occ, and end stages from
# the "overall refinement statistics: step by step" table of a phenix refinement log.
# Stage 0 is the model as read in, before bulk-solvent correction and scaling, so its
# R-factors sit well above the deposited ones; 1_bss is the same model after the first
# bulk-solvent/scaling step and is the fair "starting point" to compare against.
set -euo pipefail

usage() {
    echo "usage: $(basename "$0") <phenix_log>" >&2
    exit 1
}

[ "$#" -eq 1 ] || usage
log="$1"
[ -r "$log" ] || { echo "error: cannot read log file: $log" >&2; exit 1; }

echo "stage,r_work,r_free,n_water"

awk '
    # Capture begins only after the column-header line, which excludes the
    # legend lines (e.g. "1_occ: refinement of occupancies") above it.
    /r-work/ && /r-free/ { inblock = 1; next }
    !inblock { next }

    # Closing dashed line or a new ===== section ends the table.
    /^[[:space:]]*-+[[:space:]]*$/ || /====/ { inblock = 0; next }

    {
        colon = index($0, ":")
        if (colon == 0) next
        stage = substr($0, 1, colon - 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", stage)
        if (stage != "0" && stage != "1_bss" && stage != "end" && stage !~ /^[0-9]+_occ$/) next

        n = split(substr($0, colon + 1), f)
        if (n < 8 || f[1] !~ /^[0-9.]+$/) next
        print stage "," f[1] "," f[2] "," f[8]
    }
' "$log"
