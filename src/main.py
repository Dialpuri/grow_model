"""
    main.py - Jordan Dialpuri 01/08/2023
"""

import os
from dataclasses import dataclass
from typing import List

import gemmi


@dataclass
class Params:
    """
    Dataclass containing script parameters
    """

    file_list: str = "data/files_list.txt"
    alignment_atoms: List[str] = ["C4'", "C5'", "O5'", "P"]
    pdb_dir: str = "/vault"
    


def get_pdb_list() -> List[str]:
    """
    Get PDB list from Params.file_list
    """
    if not os.path.isfile(Params.file_list):
        print(f"{Params.file_list} is not a file")
        return []

    pdb_list = None
    with open(Params.file_list, "r", encoding="UTF-8") as file_list:
        pdb_list = [x.strip("\n").lower() for x in file_list]

    if pdb_list:
        return pdb_list

    print("PDB List is empty, check input file")
    return []


def get_pdb_structure(pdb_code: str) -> gemmi.Structure: 
    """Get a gemmi.Structure from a pdb code using the standard rsynced PDB vault format

    Args:
        pdb_code (str): PDB 4 Letter Code

    Returns:
        gemmi.Structure: Result of gemmi.read_structure
    """
    pdb_path = os.path.join(Params.pdb_dir, pdb_code[1:3], 


def align_residue(
    current_residue: gemmi.Residue, reference_residue: gemmi.Residue
) -> gemmi.Residue:
    """Align current_residue to supplied reference residue and return transformed residue

    Args:
        current_residue (gemmi.Residue): Residue to transform
        reference_residue (gemmi.Residue): Residue to transform to

    Returns:
        gemmi.Residue: Transformed residue
    """

    superpose: gemmi.SupResult = gemmi.superpose_positions(
        [current_residue.sole_atom(atom) for atom in Params.alignment_atoms],
        [reference_residue.sole_atom(atom) for atom in Params.alignment_atoms],
    )

    transform: gemmi.Transform = superpose.transform

    for atom in current_residue:
        atom.pos = transform.apply(atom.pos)

    return current_residue

def analyse_pdb(pdb_code: str) -> None: 
    """Load and analyse PDB file

    Args:
        pdb_code (str): PDB 4 Letter Code
    """
    
    


def main():
    pdb_list: List[str] = get_pdb_list()

    

if __name__ == "__main__":
    main()
