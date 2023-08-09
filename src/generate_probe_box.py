"""
    main.py - Jordan Dialpuri 01/08/2023
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, ClassVar
import numpy as np
import gemmi


@dataclass
class Params:
    """
    Dataclass containing script parameters
    """

    file_list: str = "data/files_list.txt"
    alignment_atoms: ClassVar[list[str]] = ["C4'", "C3'", "O3'"]
    sugar_atoms: ClassVar[list[str]] = ["C1'", "C2'", "C3'", "O3'", "C4'", "C5'"]
    pdb_dir: str = "/old_vault/pdb"
    pdb_dir_extracted: bool = True
    reference_residue_path: str = "data/reference_residue_zeroed.pdb"
    mtz_dir: str = "/y/people/jsd523/dev/grow_model/data/mtz_files"
    shape: int = 16

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
    current_residue: gemmi.Residue, reference_residue: gemmi.Residue, mean_pos
) -> Optional[gemmi.Transform]:
    """Align current_residue to supplied reference residue and return transform

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
    return gemmi.Transform(transform.mat, transform.vec  + gemmi.Vec3(*mean_pos) - gemmi.Vec3(Params.shape/2, Params.shape/2, Params.shape/2))
    #   
    # return transform


def analyse_pdb(pdb_code: str, reference_residue: gemmi.Residue, mean_pos) -> None: 
    """Analyse PDB structure

    Args:
        pdb_code (str): PDB 4 Letter Code
        output_path (str): Output Path
    """
    print(pdb_code)
    structure: gemmi.Structure = get_pdb_structure(pdb_code=pdb_code, strict_checking=True)
    mtz_file_path = os.path.join(Params.mtz_dir, f"{pdb_code}.mtz")
    if not os.path.isfile(mtz_file_path):
        print("Cannot find MTZ")
        return
    
                
    mtz = gemmi.read_mtz_file(mtz_file_path)
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    
    output_structure = gemmi.Structure()
    output_model = gemmi.Model("1") 
    output_chain = gemmi.Chain("A")
    
    for chain in structure[0]:
        for residue in chain: 
            info = gemmi.find_tabulated_residue(residue.name)

            if info.kind not in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                continue
            print(mean_pos)
            transform = align_residue(residue, reference_residue=reference_residue, mean_pos=mean_pos)
            if not transform:
                continue

            print("Ref Before:", reference_residue.sole_atom("O3'").pos)

            print("Before:", residue.sole_atom("O3'").pos)
            
            for atom in residue:
                atom.pos = gemmi.Position(transform.apply(atom.pos))
            
            print("After:", residue.sole_atom("O3'").pos)

            print(transform.vec, transform.mat)
            print(transform.inverse().vec)

            box = np.zeros((Params.shape, Params.shape, Params.shape), dtype=np.float32)
            grid.interpolate_values(box, transform.inverse())
            
            ccp4 = gemmi.Ccp4Map()
            ccp4.grid = gemmi.FloatGrid(box)
            ccp4.grid.unit_cell.set(Params.shape, Params.shape, Params.shape, 90, 90, 90)
            ccp4.grid.spacegroup = gemmi.SpaceGroup('P1')
            ccp4.update_ccp4_header()
            ccp4.write_ccp4_map('debug/aligned_residue_grid.ccp4')
            

            residue.name = "W"
            output_chain.add_residue(residue)
            reference_residue.name = "R"
            output_chain.add_residue(reference_residue)
            output_model.add_chain(output_chain)
            output_structure.add_model(output_model)
            output_structure.write_pdb("debug/aligned_residue.pdb")
            
            exit()
            
    

def create_reference_residue_box(pdb_code: str) -> None: 
    """Create reference residue from a PDB code and center at 8,8,8 and output PDB file 

    Args:
        pdb_code (str): PDB 4 Letter Code
        output_path (str): Output Path
    """
    
    structure: gemmi.Structure = get_pdb_structure(pdb_code=pdb_code, strict_checking=True)
    mtz_file_path = os.path.join(Params.mtz_dir, f"{pdb_code}.mtz")
    if not os.path.isfile(mtz_file_path):
        return
                
    mtz = gemmi.read_mtz_file(mtz_file_path)

    output_structure = gemmi.Structure()
    output_model = gemmi.Model("1") 
    output_chain = gemmi.Chain("A")

    reference_residue = structure[0][0][2]
    
    if reference_residue is None: 
        return          
    
    box_offset = gemmi.Position(
        Params.shape / 2, Params.shape / 2, Params.shape / 2)

    positions = [reference_residue.sole_atom(x).pos.tolist() for x in Params.sugar_atoms]
    sugar_mean_pos = np.mean(positions, axis=0)

    # print(f"{sugar_mean_pos=}")
    
    # middle_box_pos = [0,0,0]
    # middle_to_mean = middle_box_pos - sugar_mean_pos    

    # middle_to_mean_vec = gemmi.Vec3(*middle_to_mean)

    # print(f"{middle_box_pos=}\n{mean_pos=}\n{middle_to_mean_vec=}")

    identity_matrix = gemmi.Mat33([[1,0,0], [0,1,0], [0,0,1]])
    transform = gemmi.Transform(identity_matrix, -gemmi.Position(*sugar_mean_pos)+box_offset)
    
    for atom in reference_residue:
        atom.pos = gemmi.Position(transform.apply(atom.pos))
    
    positions = [reference_residue.sole_atom(x).pos.tolist() for x in Params.alignment_atoms]
    mean_pos = np.mean(positions, axis=0)
    # print(f"{mean_pos=}")

    # print(np.mean([reference_residue.sole_atom(x).pos.tolist() for x in Params.alignment_atoms], axis=-1))
    # print(f"ref_align_atoms={[reference_residue.sole_atom(x).pos.tolist() for x in Params.alignment_atoms]}")

    # output_chain.add_residue(reference_residue)
    # output_model.add_chain(output_chain)
    # output_structure.add_model(output_model)
    # output_structure.write_pdb("data/reference_residue_zeroed.pdb")
     
    return reference_residue, mean_pos
            

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
    
    reference_residue, mean_pos = create_reference_residue_box("1hr2")
    
    # reference_residue = gemmi.read_structure(Params.reference_residue_path)[0][0][0]
    
    analyse_pdb(pdb_code="1hr2", reference_residue=reference_residue, mean_pos=mean_pos)
        
if __name__ == "__main__":
    main()
