"""
Align PDB to probe points 
Jordan Dialpuri 2023
"""
import json
import os
import random
from dataclasses import dataclass
from typing import ClassVar, Dict, Generator, List, Optional, Tuple

import gemmi
import numpy as np


@dataclass
class Params:
    """
    Dataclass containing script parameters
    """

    file_list: str = "data/files_list.txt"
    alignment_atoms: ClassVar[list[str]] = ["C4'", "C3'", "O3'"]
    pdb_dir: str = "/old_vault/pdb"
    pdb_dir_extracted: bool = True
    reference_residue_path: str = "data/reference_residue.pdb"
    mtz_dir: str = "/y/people/jsd523/dev/grow_model/data/mtz_files"


def get_pdb_list(file_list: str = Params.file_list) -> Optional[List[str]]:
    """
    Get PDB list from Params.file_list
    """
    if not os.path.isfile(file_list):
        # print(f"{file_list} is not a file")
        return

    pdb_list = None
    with open(file_list, "r", encoding="UTF-8") as file_:
        pdb_list = [x.strip("\n").lower() for x in file_]

    if pdb_list:
        return pdb_list

    # print("PDB List is empty, check input file")
    return


def get_pdb_structure(
    pdb_code: str, strict_checking: bool = False
) -> Optional[gemmi.Structure]:
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
        # print(f"{pdb_code} at path {pdb_path} cannot be found ...")
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
    current_residue: gemmi.Residue, reference_data: Dict[str, List[float]]
) -> gemmi.Transform:
    """Align current_residue to supplied reference residue and return transformed residue

    Args:
        current_residue (gemmi.Residue): Residue to transform
        reference_data (Dict[str, List[float]]): List of probe point reference location

    Returns:
        gemmi.Residue: Transformed residue
    """

    for alignment_atom in Params.alignment_atoms:
        if alignment_atom not in current_residue:
            return

    try:
        superpose: gemmi.SupResult = gemmi.superpose_positions(
            [current_residue.sole_atom(atom).pos for atom in Params.alignment_atoms],
            [gemmi.Position(*reference_data[atom]) for atom in Params.alignment_atoms],
        )
    except RuntimeError:
        return 
    
    return superpose.transform


def transform_probe_data(
    transform: gemmi.Transform, probe_data: Dict[str, List[float]]
) -> List[gemmi.Position]:
    """Transform probe data (list of points) using supplied transform

    Args:
        transform (gemmi.Transform): Transform calculated using supoerpose
        probe_data (Dict[int, List[float]]): Input data from JSON file

    Returns:
        List[gemmi.Position]: List of probe points to query
    """

    return_list: List[gemmi.Position] = []
    for value in probe_data.values():
        position = gemmi.Position(*value)

        return_list.append(gemmi.Position(transform.apply(position)))

    return return_list


def calculate_torsions(
    residue: gemmi.Residue, next_residue: gemmi.Residue
) -> Optional[Tuple[float, float, float]]:
    """Calculate torsion angles from residue
        epsilon - C3' O3' P O5'
        phi - O3' P O5' C5'
        eta - C4' C5' O5' P

    Args:
        residue (gemmi.Residue): _description_
        next_residue (gemmi.Residue): _description_

    Returns:
        Tuple[float, float, float]: epsilon, phi, eta
    """

    current_p = residue.find_atom("P", "\0")
    # current_c4 = residue.find_atom("C4'", "\0")
    current_c3 = residue.find_atom("C3'", "\0")
    current_o3 = residue.find_atom("O3'", "\0")

    next_c4 = next_residue.find_atom("C4'", "\0")
    next_c5 = next_residue.find_atom("C5'", "\0")
    next_o5 = next_residue.find_atom("O5'", "\0")
    next_p = next_residue.find_atom("P", "\0")

    # next_o3 = next_residue.find_atom("O3'", "\0")
    # next_c3 = next_residue.find_atom("C3'", "\0")
    next_c4 = next_residue.find_atom("C4'", "\0")

    eta = [next_p, next_o5, next_c5, next_c4]
    epsilon = [current_c3, current_o3, next_p, next_o5]
    phi = [current_o3, next_p, next_o5, next_c5]

    if all(eta) and all(phi) and all(epsilon):
        epsilon = gemmi.calculate_dihedral(*[x.pos for x in epsilon])
        phi = gemmi.calculate_dihedral(*[x.pos for x in phi])
        eta = gemmi.calculate_dihedral(*[x.pos for x in eta])

        return epsilon, phi, eta


def create_gemmi_residue(
    name: str, num: str, positions: List[gemmi.Position]
) -> gemmi.Residue:
    """Create and return a gemmi residue from a list of positions

    Taken from Paul Bond - 
    https://github.com/paulsbond/grow22/blob/0553d55046b850f54afeec1cfaa329cb49c01445/probes.py#L108C38-L108C38

    Args:
        name (_type_): _description_
        num (_type_): _description_
        atoms (_type_): _description_

    Returns:
        _type_: _description_
    """
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = gemmi.SeqId(num)
    for position in positions:
        atom = gemmi.Atom()
        atom.name = "X"
        # atom.element = gemmi.Element("X")
        atom.pos = position
        residue.add_atom(atom)
    return residue


def analyse_pdb(
    pdb_code: str, probe_data: Dict[str, Dict[str, List[float]]]
) -> Optional[Generator[Tuple[np.ndarray, float, float, float], None, None]]:
    """Load and analyse PDB file

    Args:
        pdb_code (str): PDB 4 Letter Code
        reference_residue (gemmi.Residue): Reference Residue loaded from file
    """
    # print("Analysing ", pdb_code)
    structure: gemmi.Structure = get_pdb_structure(
        pdb_code=pdb_code, strict_checking=True
    )
    
    if not structure: 
        return
    
    mtz_file_path = os.path.join(Params.mtz_dir, f"{pdb_code}.mtz")
    if not os.path.isfile(mtz_file_path):
        return
                
    mtz = gemmi.read_mtz_file(mtz_file_path)
    
    grid = mtz.transform_f_phi_to_map("FWT", "PHWT")
    grid.normalize()
    output_structure = gemmi.Structure()
    output_model = gemmi.Model("1")
    output_chain = gemmi.Chain("A")

    for chain in structure[0]:
        for n_res in range(len(chain) - 1):
            residue = chain[n_res]
            next_residue = chain[n_res + 1]
            current_O3 = residue.find_atom("O3'", "\0")
            next_P = next_residue.find_atom("P", "\0")

            info = gemmi.find_tabulated_residue(next_residue.name)

            if info.kind not in (gemmi.ResidueInfoKind.RNA, gemmi.ResidueInfoKind.DNA):
                continue

            if current_O3 and next_P:
                diff = current_O3.pos - next_P.pos

                if diff.length() > 2:
                    continue

            torsions = calculate_torsions(residue, next_residue)
            if not torsions:
                continue

            # epsilon, phi, eta = torsions
                
            transform = align_residue(
                current_residue=residue, reference_data=probe_data["base"]
            )
            if not transform:
                continue

            list_of_probes = transform_probe_data(
                transform=transform, probe_data=probe_data["probes"]
            )

            data_list = np.array([grid.interpolate_value(x) for x in list_of_probes])

            # probe_residue = create_gemmi_residue("A", "1", list_of_probes)
            # output_chain.add_residue(residue)
            # output_chain.add_residue(next_residue)
            # output_chain.add_residue(probe_residue)

            # output_model.add_chain(output_chain)
            # output_structure.add_model(output_model)
            # output_structure.write_pdb("debug/probe_points_3.pdb")

            # print(torsions)
            
            # exit()
            yield data_list, *torsions


def generate_test_train_split(output_dir: str):
    """Generate test train split

    Args:
        output_dir (str): Output directory for files train.txt and test.txt
    """
    pdb_list = None
    with open(Params.file_list, "r", encoding="UTF-8") as file_list:
        pdb_list = [x.strip("\n").lower() for x in file_list]

    random.shuffle(pdb_list)

    test_train_split = 0.8
    train_data = pdb_list[: int(test_train_split * len(pdb_list))]
    test_data = pdb_list[int(test_train_split * len(pdb_list)) :]

    with open(os.path.join(output_dir, "test.txt"), "w", encoding="UTF-8") as test_output:
        for test in test_data:
            test_output.write(f"{test}\n")

    with open(os.path.join(output_dir, "train.txt"), "w", encoding="UTF-8") as train_output:
        for train in train_data:
            train_output.write(f"{train}\n")


def generate_dataset(train_or_test: str):
    """Generate Dataset for training or testing

    Args:
        train_or_test (str): name test or train

    Yields:
        [np.ndarray, List[float], List[float]]: Data for ML
    """
    print("Generating dataset")
    probe_file = "/y/people/jsd523/dev/nautilus_library_gen/probes/probe_C3C4O3_3.json"

    data = None
    with open(probe_file, "r", encoding="UTF-8") as json_file:
        data = json.load(json_file)

    pdb_list = get_pdb_list(f"data/{train_or_test}.txt")

    if not pdb_list:
        return

    for pdb in pdb_list:
        pdb_gen = analyse_pdb(pdb, probe_data=data)
        if not pdb_gen: 
            continue
        
        for generated_data in pdb_gen:
            probe_array, epsilon, phi, eta = generated_data

            sin_ = [np.sin(epsilon), np.sin(phi), np.sin(eta)]
            cos_ = [np.cos(epsilon), np.cos(phi), np.cos(eta)]

            yield probe_array, (sin_, cos_)


if __name__ == "__main__":

    for x in generate_dataset("train"):
        ...
    # generate_test_train_split("data")
    ...
    