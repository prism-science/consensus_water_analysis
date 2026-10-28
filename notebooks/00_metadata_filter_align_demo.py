import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 00 — metadata / filter / align demo

    Single-structure sanity checks for the core pipeline modules, coalesced from
    the per-stage notebooks:

    1. **metadata** — `cw.io` + `cw.metadata`
    2. **filter by distance** — `cw.filter`
    3. **align a pair** — `cw.align`

    Each part has its own input + example structure; nothing is shared between
    parts except the imports below.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import biotite.structure.io.pdbx as pdbx
    import gemmi
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist

    import config
    from cw.align import align_to_reference
    from cw.filter import best_sym_positions, filter_waters
    from cw.io import (
        count_water_oxygens,
        load_protein,
        load_structure_waters,
        water_oxygen_mask,
    )
    from cw.metadata import metadata_row

    return (
        Path,
        align_to_reference,
        best_sym_positions,
        cdist,
        config,
        count_water_oxygens,
        filter_waters,
        gemmi,
        load_protein,
        load_structure_waters,
        metadata_row,
        np,
        pdbx,
        plt,
        water_oxygen_mask,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Part 1 — metadata (`cw.io` + `cw.metadata`)

    Default **6YBF** (chosen for its altloc waters).

    - gemmi surfaces space group / cell / resolution / unit-cell volume.
    - gemmi raw-CIF read surfaces `r_work` / `r_free` / `ligand_names`; the RCSB Data
      API supplies `experiment_condition` and `starting_model`.
    - gemmi water-O count agrees with biotite `count_water_oxygens`.
    - EDIA keys (chain, res_id, ins_code) match the water atoms.
    """)
    return


@app.cell
def _(config, mo):
    meta_pdb_id = "6ybf"
    meta_cif_input = mo.ui.text(
        value=f"{config.ALL_PDB_REDO_DIR}/{config.CIF_TEMPLATE.format(pdb_id=meta_pdb_id)}",
        placeholder="/path/to/<id>/<id>_final.cif",
        label="CIF path",
        full_width=True,
    )
    meta_edia_input = mo.ui.text(
        value=f"{config.ALL_PDB_REDO_DIR}/{config.EDIA_TEMPLATE.format(pdb_id=meta_pdb_id)}",
        placeholder="/path/to/<id>/<id>_final.json",
        label="EDIA JSON path",
        full_width=True,
    )
    mo.vstack([meta_cif_input, meta_edia_input])
    return meta_cif_input, meta_edia_input


@app.cell
def _(Path, meta_cif_input, meta_edia_input, mo):
    meta_cif_path = Path(meta_cif_input.value.strip()) if meta_cif_input.value.strip() else None
    meta_edia_path = Path(meta_edia_input.value.strip()) if meta_edia_input.value.strip() else None

    mo.stop(meta_cif_path is None, mo.md("**Set a CIF path above to continue.**"))
    mo.stop(not meta_cif_path.exists(), mo.md(f"CIF not found: `{meta_cif_path}`"))
    return meta_cif_path, meta_edia_path


@app.cell
def _(meta_cif_path, metadata_row, mo):
    meta_row = metadata_row(meta_cif_path)
    mo.md("### metadata row\n\n" + "\n".join(f"- **{k}**: `{v}`" for k, v in meta_row.items()))
    return (meta_row,)


@app.cell
def _(count_water_oxygens, meta_cif_path, meta_row, mo, pdbx):
    _cif = pdbx.CIFFile.read(meta_cif_path)
    _atoms = pdbx.get_structure(
        _cif,
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )
    _biotite_count = count_water_oxygens(_atoms)
    _gemmi_count = meta_row["num_water"]
    mo.md(
        "### water count\n\n"
        f"- **gemmi (all altlocs):** {_gemmi_count}\n"
        f"- **biotite count_water_oxygens:** {_biotite_count}\n"
        + (
            "- **Agree** ✓"
            if _gemmi_count == _biotite_count
            else f"- **Differ** — gemmi count ({_gemmi_count}) is canonical"
        )
    )
    return


@app.cell
def _(load_structure_waters, meta_cif_path, meta_edia_path, meta_row, mo):
    meta_df = load_structure_waters(meta_cif_path, meta_edia_path)
    _n_edia = int(meta_df["edia"].notna().sum())
    _n_water = meta_row["num_water"]
    _check = (
        f"**Match** ✓ ({_n_edia} EDIA values for {_n_water} waters)"
        if _n_edia == _n_water
        else f"**Mismatch** — {_n_edia} valid EDIA values for {_n_water} waters"
    )
    mo.md(f"### EDIA\n\n{_check}")
    return (meta_df,)


@app.cell
def _(meta_df):
    meta_df.head(20)
    return


@app.cell
def _(gemmi, meta_cif_path, meta_row, mo):
    _block = gemmi.cif.read(str(meta_cif_path)).sole_block()
    _nonpoly_names = list(_block.find_loop("_pdbx_entity_nonpoly.name"))
    _nonpoly_comps = list(_block.find_loop("_pdbx_entity_nonpoly.comp_id"))
    mo.md(
        "### ligands + RCSB fields\n\n"
        f"**`_pdbx_entity_nonpoly` names:** `{_nonpoly_names}`\n\n"
        f"**`_pdbx_entity_nonpoly` comp_ids:** `{_nonpoly_comps}`\n\n"
        f"**ligand_names (row):** `{meta_row['ligand_names']}`\n\n"
        f"**experiment_condition:** `{meta_row['experiment_condition']}`\n\n"
        f"**starting_model:** `{meta_row['starting_model']}`"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2 — filter by distance (`cw.filter`)

    Default **5F14** (many waters get relocated to symmetry images as well as
    removed).

    - removed-vs-kept counts look plausible
    - water→protein min-distance distribution improves after symmetry-aware
      relocation + filtering
    """)
    return


@app.cell
def _(config, mo):
    filt_pdb_id = "5f14"
    filt_cif_input = mo.ui.text(
        value=f"{config.ALL_PDB_REDO_DIR}/{config.CIF_TEMPLATE.format(pdb_id=filt_pdb_id)}",
        placeholder="/path/to/<id>/<id>_final.cif",
        label="CIF path",
        full_width=True,
    )
    filt_cutoff_input = mo.ui.number(
        value=float(config.WATER_PROT_DIST_CUTOFF),
        start=1.0,
        stop=10.0,
        step=0.5,
        label="Distance cutoff (Å)",
    )
    mo.vstack([filt_cif_input, filt_cutoff_input])
    return filt_cif_input, filt_cutoff_input


@app.cell
def _(Path, filt_cif_input, filt_cutoff_input, mo):
    filt_cif_path = Path(filt_cif_input.value.strip()) if filt_cif_input.value.strip() else None
    filt_cutoff = float(filt_cutoff_input.value)

    mo.stop(filt_cif_path is None, mo.md("**Set a CIF path above to continue.**"))
    mo.stop(not filt_cif_path.exists(), mo.md(f"CIF not found: `{filt_cif_path}`"))
    return filt_cif_path, filt_cutoff


@app.cell
def _(count_water_oxygens, filt_cif_path, gemmi, pdbx):
    _cif = pdbx.CIFFile.read(str(filt_cif_path))
    filt_atoms = pdbx.get_structure(
        _cif,
        model=1,
        altloc="all",
        extra_fields=["b_factor", "occupancy"],
        use_author_fields=True,
    )

    _st = gemmi.read_structure(str(filt_cif_path))
    filt_cell = _st.cell
    filt_sg = _st.find_spacegroup() or gemmi.SpaceGroup("P 1")
    filt_n_before = count_water_oxygens(filt_atoms)
    return filt_atoms, filt_cell, filt_n_before, filt_sg


@app.cell
def _(
    count_water_oxygens,
    filt_atoms,
    filt_cell,
    filt_cutoff,
    filt_n_before,
    filt_sg,
    filter_waters,
    mo,
):
    filtered, _stats, _final_mask = filter_waters(
        filt_atoms.copy(), filt_cell, filt_sg, filt_cutoff
    )
    filt_n_moved = _stats["n_moved"]
    _n_removed = _stats["n_removed_distance"]
    _n_after = count_water_oxygens(filtered)
    assert _stats["n_water"] == filt_n_before

    mo.md(
        f"### filter result  (cutoff = {filt_cutoff} Å)\n\n"
        f"- Waters before: **{filt_n_before}**\n"
        f"- Waters moved: **{filt_n_moved}**\n"
        f"- Waters removed: **{_n_removed}**\n"
        f"- Waters kept: **{_n_after}**\n"
        f"- Cell / space group: `{filt_cell}` / `{filt_sg.hm if filt_sg else 'P 1'}`"
    )
    return (filt_n_moved,)


@app.cell
def _(
    best_sym_positions,
    cdist,
    filt_atoms,
    filt_cell,
    filt_sg,
    water_oxygen_mask,
):
    _protein_heavy_mask = (~filt_atoms.hetero) & (filt_atoms.element != "H")
    _water_coords = filt_atoms.coord[water_oxygen_mask(filt_atoms)]
    _protein_coords = filt_atoms.coord[_protein_heavy_mask]

    filt_naive_dists = cdist(_water_coords, _protein_coords).min(axis=1)
    _, filt_dists = best_sym_positions(_water_coords, _protein_coords, filt_cell, filt_sg)
    return filt_dists, filt_naive_dists


@app.cell
def _(filt_cutoff, filt_dists, filt_naive_dists, np, plt):
    _bins = np.linspace(0, max(filt_naive_dists.max(), filt_dists.max()), 60)
    filt_fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.hist(filt_naive_dists, bins=_bins, alpha=0.5, color="gray", label="naive (no symop)")
    _ax.hist(filt_dists, bins=_bins, alpha=0.7, color="steelblue", label="symmetry-aware")
    _ax.axvline(filt_cutoff, color="tomato", linewidth=1.5, label=f"cutoff = {filt_cutoff} Å")
    _ax.set_xlabel("Distance to nearest protein atom (Å)")
    _ax.set_ylabel("Waters")
    _ax.set_title("Water–protein distances: naive vs symmetry-aware")
    _ax.legend()
    filt_fig.tight_layout()
    filt_fig
    return


@app.cell(hide_code=True)
def _(filt_dists, filt_n_moved, filt_naive_dists, mo, np):
    _rows = [
        "### distance distribution",
        "",
        f"Waters moved to a symmetry image: **{filt_n_moved}** / {len(filt_dists)}",
        "",
        "|           | min | median | max |",
        "|-----------|-----|--------|-----|",
        f"| naive     | {filt_naive_dists.min():.2f} | {np.median(filt_naive_dists):.2f} | {filt_naive_dists.max():.2f} |",
        f"| sym-aware | {filt_dists.min():.2f} | {np.median(filt_dists):.2f} | {filt_dists.max():.2f} |",
    ]
    mo.md("\n".join(_rows))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 3 — align a pair (`cw.align`)

    Aligns one mobile structure (default **5FEK**) onto a reference
    (`config.REF_PDB_ID`), reports RMSD before/after, and verifies the output CIF
    retains every atom and water.
    """)
    return


@app.cell
def _(config, mo):
    aln_mobile_id = "5fek"
    aln_ref_input = mo.ui.text(
        value=f"{config.ALL_PDB_REDO_DIR}/{config.CIF_TEMPLATE.format(pdb_id=config.REF_PDB_ID)}",
        placeholder="/path/to/ref/<id>_final.cif",
        label="Reference CIF",
        full_width=True,
    )
    aln_mob_input = mo.ui.text(
        value=f"{config.ALL_PDB_REDO_DIR}/{config.CIF_TEMPLATE.format(pdb_id=aln_mobile_id)}",
        placeholder="/path/to/mobile/<id>_final.cif",
        label="Mobile CIF",
        full_width=True,
    )
    aln_out_input = mo.ui.text(
        value="data/test_aligned.cif", label="Output path", full_width=True
    )
    mo.vstack([aln_ref_input, aln_mob_input, aln_out_input])
    return aln_mob_input, aln_out_input, aln_ref_input


@app.cell
def _(Path, aln_mob_input, aln_out_input, aln_ref_input, mo):
    aln_ref_path = Path(aln_ref_input.value.strip()) if aln_ref_input.value.strip() else None
    aln_mob_path = Path(aln_mob_input.value.strip()) if aln_mob_input.value.strip() else None
    aln_out_path = Path(aln_out_input.value.strip() or "/tmp/test_aligned.cif")

    mo.stop(not (aln_ref_path and aln_mob_path), mo.md("**Set both CIF paths above to continue.**"))
    for _p, _label in [(aln_ref_path, "Reference"), (aln_mob_path, "Mobile")]:
        mo.stop(not _p.exists(), mo.md(f"**{_label} not found:** `{_p}`"))
    return aln_mob_path, aln_out_path, aln_ref_path


@app.cell
def _(
    align_to_reference,
    aln_mob_path,
    aln_out_path,
    aln_ref_path,
    load_protein,
    mo,
):
    aln_ref_protein, _ = load_protein(aln_ref_path)
    aln_report = align_to_reference(aln_mob_path, aln_ref_protein, out_path=aln_out_path)

    mo.stop(aln_report is None, mo.md("**Alignment skipped** — fewer than 10 common Cα atoms."))

    def _fmt(v):
        return " ".join(str(v).split())  # collapse the 3×3 R's newlines onto one line

    mo.md(
        "### alignment report\n\n| field | value |\n|---|---|\n"
        + "\n".join(f"| `{k}` | `{_fmt(v)}` |" for k, v in aln_report.items())
    )
    return (aln_report,)


@app.cell
def _(aln_mob_path, aln_out_path, aln_report, count_water_oxygens, mo, pdbx):
    def _counts(path):
        _f = pdbx.CIFFile.read(str(path))
        _a = pdbx.get_structure(_f, model=1, altloc="all", use_author_fields=True)
        return len(_a), count_water_oxygens(_a)

    _ = aln_report  # depend on the report so the aligned file is written before we read it
    _n_before, _w_before = _counts(aln_mob_path)
    _n_after, _w_after = _counts(aln_out_path)
    _status = (
        "✓ match"
        if (_n_before == _n_after and _w_before == _w_after)
        else "✗ mismatch — check altloc write"
    )
    mo.md(
        "### atom counts (mobile vs aligned output)\n\n"
        f"| | before | after |\n|---|---|---|\n"
        f"| total atoms (altloc=all) | {_n_before} | {_n_after} |\n"
        f"| water oxygens | {_w_before} | {_w_after} |\n\n"
        f"**{_status}**"
    )
    return


if __name__ == "__main__":
    app.run()
