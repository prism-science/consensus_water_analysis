"""Protein heavy-atom shift of one refine_ucsf_water.sh dataset: Dimple input vs the final
stage-2 model, matched by atom id with no superposition (both are in the same frame).

Prints one CSV row, rmsd_ca,rmsd_heavy,max_shift,n_flips, for collect_ucsf_water.sh.
Symmetry-equivalent atom names are matched to the closer assignment before measuring;
n_flips counts the Asn/Gln/His side chains whose swap that was.

usage: phenix.python protein_rmsd.py <out_root>/<dataset>
"""
import math
import os
import sys

import iotbx.pdb


def protein_coordinates(path):
    hierarchy = iotbx.pdb.input(file_name=path).construct_hierarchy()
    coordinates = {}
    for chain in hierarchy.only_model().chains():
        if not chain.is_protein():
            continue
        for residue_group in chain.residue_groups():
            for atom_group in residue_group.atom_groups():
                for atom in atom_group.atoms():
                    if atom.element.strip() in ("H", "D"):
                        continue
                    key = (chain.id, residue_group.resseq.strip(), residue_group.icode,
                           atom_group.altloc, atom_group.resname, atom.name.strip())
                    coordinates[key] = atom.xyz
    return coordinates


# Atom pairs whose names are interchangeable by symmetry (Phe/Tyr/Arg/Asp/Glu) or by an
# amide/imidazole flip (Asn/Gln/His); flips are counted separately.
SWAP_PAIRS = {
    "PHE": [("CD1", "CD2"), ("CE1", "CE2")],
    "TYR": [("CD1", "CD2"), ("CE1", "CE2")],
    "ARG": [("NH1", "NH2")],
    "ASP": [("OD1", "OD2")],
    "GLU": [("OE1", "OE2")],
    "ASN": [("OD1", "ND2")],
    "GLN": [("OE1", "NE2")],
    "HIS": [("ND1", "CD2"), ("CE1", "NE2")],
}
FLIP_RESIDUES = ("ASN", "GLN", "HIS")


def resolve_swaps(before, after):
    """Rename `after` atoms per residue when the swapped naming lies closer to `before`."""
    resolved = dict(after)
    flips = 0
    residues = {k[:5] for k in before if k[4] in SWAP_PAIRS}
    for residue in residues:
        pairs = [(residue + (a,), residue + (b,)) for a, b in SWAP_PAIRS[residue[4]]]
        if not all(k in before and k in after for pair in pairs for k in pair):
            continue
        direct = sum(squared_distance(before[k], after[k]) for pair in pairs for k in pair)
        swapped = sum(squared_distance(before[a], after[b]) + squared_distance(before[b], after[a])
                      for a, b in pairs)
        if swapped < direct:
            for a, b in pairs:
                resolved[a], resolved[b] = after[b], after[a]
            if residue[4] in FLIP_RESIDUES:
                flips += 1
    return resolved, flips


def squared_distance(first, second):
    return sum((f - s) ** 2 for f, s in zip(first, second))


def rmsd(before, after, keys):
    if not keys:
        return float("nan")
    total = sum(squared_distance(before[k], after[k]) for k in keys)
    return math.sqrt(total / len(keys))


dataset_dir = sys.argv[1].rstrip("/")
dataset = os.path.basename(dataset_dir)
input_pdb = os.path.join(dataset_dir, "inputs", dataset + "_dimple.pdb")
final_pdb = os.path.join(dataset_dir, "stage2_water", dataset + "_stage2_water_001.pdb")
before = protein_coordinates(input_pdb)
after, flips = resolve_swaps(before, protein_coordinates(final_pdb))
matched = [k for k in before if k in after]
if len(matched) != len(before):
    sys.stderr.write("%s: %d of %d input protein atoms unmatched\n"
                     % (dataset, len(before) - len(matched), len(before)))
alpha_carbons = [k for k in matched if k[5] == "CA"]
max_shift = max(math.sqrt(squared_distance(before[k], after[k])) for k in matched)
print("%.3f,%.3f,%.2f,%d" % (rmsd(before, after, alpha_carbons), rmsd(before, after, matched),
                             max_shift, flips))
