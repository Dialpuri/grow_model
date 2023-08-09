"""
    main.py - Jordan Dialpuri 01/08/2023
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar

import gemmi


@dataclass
class Params:
    """
    Dataclass containing script parameters
    """

    file_list: str = "data/files_list.txt"
    alignment_atoms: ClassVar[list[str]] = ["C4'", "C5'", "O5'", "P"]
    pdb_dir: str = "/old_vault/pdb"
    pdb_dir_extracted: bool = True
    reference_residue_path: str = "data/reference_residue.pdb"
    


def get_pdb_list() -> Optional[List[str]]:
    """
    Get PDB list from Params.file_list
    """
    if not os.path.isfile(Params.file_list):
        print(f"{Params.file_list} is not a file")
        return 

    pdb_list = None
    with open(Params.file_list, "r", encoding="UTF-8") as file_list:
        pdb_list = [x.strip("\n").lower() for x in file_list]

    if pdb_list:
        return pdb_list

    print("PDB List is empty, check input file")
    return 


def get_pdb_structure(pdb_code: str, strict_checking: bool = False) -> Optional[gemmi.Structure]: 
    """Get a gemmi.Structure from a pdb code using the standard rsynced PDB vault format

    Args:
        pdb_code (str): PDB 4 Letter Code

    Returns:
        gemmi.Structure: Result of gemmi.read_structure
    """
    if Params.pdb_dir_extracted:
        pdb_path = os.path.join(Params.pdb_dir, f"pdb{pdb_code}.ent")
    else:
        pdb_path = os.path.join(Params.pdb_dir, pdb_code[1:3], f"pdb{pdb_code}.ent")
        
    if not os.path.isfile(pdb_path):
        print(f"{pdb_code} at path {pdb_path} cannot be found ...")
        return 
        
    structure: gemmi.Structure = gemmi.read_structure(pdb_path)
    
    if strict_checking:
        for chain in structure[0]:
            for residue in chain: 
                info = gemmi.find_tabulated_residue(residue.name)

                if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                    return structure
    else: 
        return structure


def align_residue(
    current_residue: gemmi.Residue, reference_residue: gemmi.Residue
) -> Optional[gemmi.Residue]:
    """Align current_residue to supplied reference residue and return transformed residue

    Args:
        current_residue (gemmi.Residue): Residue to transform
        reference_residue (gemmi.Residue): Residue to transform to

    Returns:
        gemmi.Residue: Transformed residue
    """

    for alignment_atom in Params.alignment_atoms: 
        if alignment_atom not in current_residue:
            return 

    superpose: gemmi.SupResult = gemmi.superpose_positions(
        [reference_residue.sole_atom(atom).pos for atom in Params.alignment_atoms],
        [current_residue.sole_atom(atom).pos for atom in Params.alignment_atoms],
    )

    transform: gemmi.Transform = superpose.transform

    for atom in current_residue:
        atom.pos = gemmi.Position(transform.apply(atom.pos))

    return current_residue

def analyse_pdb(pdb_code: str, reference_residue: gemmi.Residue) -> None: 
    """Load and analyse PDB file

    Args:
        pdb_code (str): PDB 4 Letter Code
        reference_residue (gemmi.Residue): Reference Residue loaded from file 
    """
    
    structure: gemmi.Structure = get_pdb_structure(pdb_code=pdb_code, strict_checking=True)
    
    output_structure = gemmi.Structure()
    output_model = gemmi.Model("1") 
    output_chain = gemmi.Chain("A")

    for chain in structure[0]:
        for residue in chain: 
            aligned_residue = align_residue(current_residue=residue, reference_residue=reference_residue)
            
            if not aligned_residue: 
                continue
            
            output_chain.add_residue(aligned_residue)
            
    output_model.add_chain(output_chain)
    output_structure.add_model(output_model)    
    output_structure.write_pdb("data/all_superimposed.pdb")
            
            

def create_reference_residue(pdb_code: str, output_path: str) -> None: 
    """Create reference residue from a PDB code and output PDB file to output_path

    Args:
        pdb_code (str): PDB 4 Letter Code
        output_path (str): Output Path
    """
    structure = get_pdb_structure(pdb_code=pdb_code)
    
    reference_residue = None
    
    for chain in structure[0]:
        if reference_residue is None: 
            for residue in chain: 
                info = gemmi.find_tabulated_residue(residue.name)

                if info.kind in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                    skip = False
                    for atom in Params.alignment_atoms:
                        if atom not in residue: 
                            skip = True
                    
                    if not skip:
                        reference_residue = residue
                        break
                    
    if not reference_residue:
        print("No suitable residue found in this PDB file - try another")
        return
    
    alignment_atom_positions: List[gemmi.Position] = [residue.sole_atom(atom).pos for atom in Params.alignment_atoms]

    sum_position: gemmi.Position = gemmi.Position(0,0,0)
    count: int = 0
    
    for position in alignment_atom_positions:
        sum_position += position
        count += 1
        
    average_position: gemmi.Position = sum_position / count
    
    transform: gemmi.Transform = gemmi.Transform(gemmi.Mat33([[1,0,0],[0,1,0],[0,0,1]]), -average_position)
    
    for atom in residue: 
        atom.pos = gemmi.Position(transform.apply(atom.pos))
        
    output_structure = gemmi.Structure()
    output_model = gemmi.Model("1") 
    output_chain = gemmi.Chain("A")
    output_chain.add_residue(residue)
    output_model.add_chain(output_chain)
    output_structure.add_model(output_model)
    output_structure.write_pdb(output_path)
    
    
    
def main():
    # create_reference_residue(pdb_list[0], "data/reference_residue.pdb")
    
    pdb_list: List[str] = get_pdb_list()
    
    reference_residue = gemmi.read_structure(Params.reference_residue_path)[0][0][0]
    
    analyse_pdb(pdb_code=pdb_list[0], reference_residue=reference_residue)

if __name__ == "__main__":
    main()
