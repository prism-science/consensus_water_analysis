#!/usr/bin/env phenix.python
"""Recalculate R-work/R-free without refining a deposited structural model.

Run on a Phenix-enabled host/pod.  For each UCSF-P* dataset, this script uses
the Dimple MTZ F/SIGF observations and the CCP4/REFMAC FreeR convention
(FreeR_flag == 0 is the test set).  Phenix performs one macrocycle containing
only bulk-solvent modelling and overall scaling; coordinates, ADPs,
occupancies, TLS parameters, and ordered solvent are not refined.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from iotbx import mtz


DEFAULT_BASE = Path(
    "/mnt/diffuse-shared/users/dorismai/water_datasets/ucsf_nsp3_mac1_P43"
)


@dataclass(frozen=True)
class Dataset:
    name: str
    pdb: Path
    reflections: Path
    original_r_work: float
    original_r_free: float
    pdb_work_count: int
    pdb_free_count: int
    pdb_header_atom_count: int
    coordinate_atom_count: int
    mtz_observation_count: int
    mtz_free_count: int


def remark_value(text: str, label: str, cast):
    pattern = rf"^REMARK   3\s+{re.escape(label)}\s*:\s*([0-9.]+)"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing PDB REMARK 3 field: {label}")
    return cast(match.group(1))


def mtz_column(path: Path, label: str):
    obj = mtz.object(str(path))
    indices = obj.extract_miller_indices()
    extracted = obj.get_column(label).extract_values_and_selection_valid()
    return {
        tuple(index): value
        for index, value, valid in zip(
            indices, extracted.values, extracted.selection_valid
        )
        if valid
    }


def inspect_dataset(directory: Path) -> Dataset:
    name = directory.name
    pdb = directory / f"{name}_dimple.pdb"
    reflections = directory / f"{name}_dimple.mtz"
    if not pdb.is_file() or not reflections.is_file():
        raise RuntimeError(f"{name}: missing Dimple PDB or MTZ")

    text = pdb.read_text(errors="replace")
    original_r_work = remark_value(text, "R VALUE            (WORKING SET)", float)
    original_r_free = remark_value(text, "FREE R VALUE", float)
    pdb_work_count = remark_value(text, "NUMBER OF REFLECTIONS", int)
    pdb_free_count = remark_value(text, "FREE R VALUE TEST SET COUNT", int)
    pdb_header_atom_count = remark_value(text, "ALL ATOMS", int)
    coordinate_atom_count = sum(
        line.startswith(("ATOM  ", "HETATM")) for line in text.splitlines()
    )

    observations = mtz_column(reflections, "F")
    free_flags = mtz_column(reflections, "FreeR_flag")
    common_indices = observations.keys() & free_flags.keys()
    mtz_observation_count = len(common_indices)
    # CCP4/REFMAC convention: value 0 is the selected test set; values 1..19
    # are work-set partitions.  Phenix's internal boolean convention differs,
    # so the command below explicitly supplies test_flag_value=0.
    mtz_free_count = sum(int(free_flags[index]) == 0 for index in common_indices)

    expected_total = pdb_work_count + pdb_free_count
    if mtz_observation_count != expected_total:
        raise RuntimeError(
            f"{name}: Dimple MTZ has {mtz_observation_count} usable Fobs, "
            f"but PDB reports {pdb_work_count} work + {pdb_free_count} free "
            f"= {expected_total}"
        )
    if mtz_free_count != pdb_free_count:
        raise RuntimeError(
            f"{name}: FreeR_flag==0 selects {mtz_free_count} observations, "
            f"but PDB reports {pdb_free_count} test reflections"
        )

    return Dataset(
        name=name,
        pdb=pdb,
        reflections=reflections,
        original_r_work=original_r_work,
        original_r_free=original_r_free,
        pdb_work_count=pdb_work_count,
        pdb_free_count=pdb_free_count,
        pdb_header_atom_count=pdb_header_atom_count,
        coordinate_atom_count=coordinate_atom_count,
        mtz_observation_count=mtz_observation_count,
        mtz_free_count=mtz_free_count,
    )


def run_one(dataset: Dataset, output_root: Path):
    output_dir = output_root / dataset.name
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{dataset.name}_phenix_unrefined"
    command = [
        "phenix.refine",
        str(dataset.pdb),
        str(dataset.reflections),
        "data_manager.miller_array.labels.name=F,SIGF",
        "data_manager.miller_array.labels.name=FreeR_flag",
        "data_manager.fmodel.xray_data.r_free_flags.test_flag_value=0",
        # Preserve the exact deposited work/test reflection counts.
        "data_manager.fmodel.xray_data.outliers_rejection=false",
        # Only bulk-solvent modelling and overall scaling are performed.
        "strategy=none",
        "main.number_of_macro_cycles=1",
        "main.ordered_solvent=false",
        # Do not transfer the isotropic part of the overall scale into atom Bs.
        "bulk_solvent_and_scale.apply_back_trace=false",
        f"output.prefix={prefix}",
        "output.serial=1",
        "output.overwrite=true",
        "write_geo_file=false",
        "write_def_file=false",
        "write_maps=false",
        "write_map_coefficients=false",
        "write_model_cif_file=false",
        "write_final_pdb_file=false",
        "write_mtz_file=false",
    ]
    stdout_path = output_dir / "phenix_refine_stdout.log"
    with stdout_path.open("w") as stdout:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{dataset.name}: phenix.refine exited {completed.returncode}; "
            f"see {stdout_path}"
        )

    log_path = output_dir / f"{prefix}_001.log"
    if not log_path.is_file():
        raise RuntimeError(f"{dataset.name}: missing expected log {log_path}")
    log_text = log_path.read_text(errors="replace")
    match = re.search(
        r"^\s*end:\s+([0-9.]+)\s+([0-9.]+)", log_text, flags=re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"{dataset.name}: cannot parse final R values")
    r_work, r_free = map(float, match.groups())

    logged_counts = {int(value) for value in re.findall(r"n_refl\.=([0-9]+)", log_text)}
    if logged_counts != {dataset.mtz_observation_count}:
        raise RuntimeError(
            f"{dataset.name}: Phenix reflection counts {sorted(logged_counts)} "
            f"do not equal validated MTZ count {dataset.mtz_observation_count}"
        )

    verification = output_dir / "verification.txt"
    verification.write_text(
        "\n".join(
            [
                f"dataset={dataset.name}",
                f"pdb={dataset.pdb}",
                f"mtz={dataset.reflections}",
                "observed_data_labels=F,SIGF",
                "r_free_flag_label=FreeR_flag",
                "r_free_test_flag_value=0",
                "r_free_convention=CCP4/REFMAC integer flag 0 mapped to Phenix test set",
                f"pdb_work_reflections={dataset.pdb_work_count}",
                f"pdb_test_reflections={dataset.pdb_free_count}",
                f"mtz_observations={dataset.mtz_observation_count}",
                f"mtz_test_reflections_flag_0={dataset.mtz_free_count}",
                "test_reflection_count_matches_pdb=yes",
                "phenix_outlier_rejection=false",
                f"pdb_header_atom_count={dataset.pdb_header_atom_count}",
                f"coordinate_record_atom_count={dataset.coordinate_atom_count}",
                f"original_R_work/free={dataset.original_r_work:.5f}/{dataset.original_r_free:.5f}",
                f"phenix_unrefined_R_work/free={r_work:.4f}/{r_free:.4f}",
                "command=" + " ".join(command),
                "",
            ]
        )
    )
    return dataset, r_work, r_free


def main() -> int:
    base = Path(os.environ.get("BASE_DIR", DEFAULT_BASE)).resolve()
    output_root = Path(
        os.environ.get("OUTPUT_DIR", base / "my_sanity_check")
    ).resolve()
    jobs = int(os.environ.get("JOBS", "4"))

    directories = sorted(path for path in base.glob("UCSF-P*") if path.is_dir())
    if not directories:
        raise RuntimeError(f"no UCSF-P* dataset directories under {base}")
    datasets = [inspect_dataset(directory) for directory in directories]
    print(
        f"Validated {len(datasets)} PDB/Dimple-MTZ pairs; FreeR_flag==0 "
        "matches every PDB test-set count.",
        flush=True,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    script_copy = output_root / Path(__file__).name
    if Path(__file__).resolve() != script_copy.resolve():
        shutil.copy2(__file__, script_copy)

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_dataset = {
            executor.submit(run_one, dataset, output_root): dataset
            for dataset in datasets
        }
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            try:
                result = future.result()
            except Exception as error:
                failures.append((dataset.name, str(error)))
                print(f"FAILED {dataset.name}: {error}", file=sys.stderr, flush=True)
            else:
                results.append(result)
                _, r_work, r_free = result
                print(
                    f"DONE {dataset.name}: {r_work:.4f}/{r_free:.4f}", flush=True
                )

    if failures:
        detail = "; ".join(f"{name}: {error}" for name, error in failures)
        raise RuntimeError(f"{len(failures)} dataset(s) failed: {detail}")

    summary_path = output_root / "summary.csv"
    temporary_summary = output_root / "summary.csv.tmp"
    with temporary_summary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "original_R_work/free",
                "phenix_unrefined_R_work/free",
            ]
        )
        for dataset, r_work, r_free in sorted(results, key=lambda item: item[0].name):
            writer.writerow(
                [
                    dataset.name,
                    f"{dataset.original_r_work:.5f}/{dataset.original_r_free:.5f}",
                    f"{r_work:.4f}/{r_free:.4f}",
                ]
            )
    os.replace(temporary_summary, summary_path)
    print(f"Wrote {summary_path} ({len(results)} datasets).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
