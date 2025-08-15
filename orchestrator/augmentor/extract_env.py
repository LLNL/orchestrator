import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList
from typing import Optional
from orchestrator.utils.exceptions import CellTooSmallError
from orchestrator.utils.data_standard import METADATA_KEY


def extract_env(
    original_atoms: Atoms,
    rc: float,
    atom_inds: list[int],
    new_cell: np.ndarray,
    extract_cube: Optional[bool] = False,
    keys_to_transfer: Optional[list[str]] = None,
) -> list[Atoms]:
    """
    function for extracting local environments

    Requires ase and numpy. Written by Jared Stimac (documentation
    reformatted for Orchestrator), additional checks added.

    :param original_atoms: ase atoms object of the config you wish to extract
        an atoms local env
    :param rc: cuttoff radius to extract and constrain positions in Angstroms
    :param atom_ind: list of indices (0-based) for which atom you want to
        extract local environments.
    :param new_cell: ase Cell object (3x3 array) you wish to embed the
        environment into, expected to be cube
    :param extract_cube: specifies if you want to extract all atoms
        within a cube of the same size of new_cell.
        * NOTE: the when extracting a cube shape, only atoms within a sphere
        defined by rc will be constrained for potential relaxation (not
        currently performed)
    :param keys_to_transfer: list of array keys which contain additional data
        that should be attached to the new configurations
    :returns: list of ase atoms objects with the local environment emedded
    """
    if new_cell.size == 3:
        cell_norms = new_cell
        max_cell_len = np.max(new_cell)
    else:
        if ~(new_cell[np.where(~np.eye(new_cell.shape[0], dtype=bool))] == 0):
            raise ValueError('New cell is non orthorhombic; this is an '
                             'unxpected case not accounted for')

        cell_norms = np.linalg.norm(new_cell, 2, 1)
        max_cell_len = np.max(cell_norms)
    if keys_to_transfer is None:
        keys_to_transfer = []

    # initial checks
    if ~(cell_norms[0] == cell_norms[1] == cell_norms[2]):
        raise ValueError('New cell is not a cube. This is an unxpected case '
                         'not accounted for')
    if (max_cell_len > original_atoms.cell.cellpar()[0]
            or max_cell_len > original_atoms.cell.cellpar()[1]
            or max_cell_len > original_atoms.cell.cellpar()[2]):
        raise CellTooSmallError(
            'Requested extracted cell size is larger than original structure')
    if max_cell_len < rc:
        raise CellTooSmallError('The specified value for rc is greater than '
                                'the maximum cell vector length for the new '
                                'supercell')
    if rc * 2 > max_cell_len:
        raise CellTooSmallError('2*rc is greater than the extracted cell side '
                                'length')
    if isinstance(atom_inds, int):
        atom_inds = [atom_inds]

    # get neighboring atom pos displacements
    n_atoms = original_atoms.get_positions().shape[0]
    # cuttoff for each atom, uses overlapping spheres of rc, so only need 1/2
    # length, see docs
    cutoffs = (0.5 * max_cell_len * np.ones((n_atoms))).tolist()
    nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
    nl.update(original_atoms)
    subcells = []
    key_arrays = {k: original_atoms.get_array(k) for k in keys_to_transfer}

    for atom_ind in atom_inds:
        indices, offsets = nl.get_neighbors(atom_ind)
        neigh_disp = np.zeros((offsets.shape[0], 3))
        neigh_symbols = []
        neigh_arrays = {k: [] for k in keys_to_transfer}
        neigh_ind = 0
        for i, offset in zip(indices, offsets):
            neigh_disp[neigh_ind, :] = (original_atoms.positions[i]
                                        + offset @ original_atoms.get_cell()
                                        ) - original_atoms.positions[atom_ind]
            neigh_symbols.append(original_atoms.symbols[i])
            for key in keys_to_transfer:
                neigh_arrays[key].append(key_arrays[key][i])
            neigh_ind += 1

        # find atoms in cube
        ind_in_cube = np.where(
            np.all(np.abs(neigh_disp) <= (max_cell_len / 2), 1))[0]
        cube_disp = neigh_disp[ind_in_cube, :]
        cube_symbols = []
        cube_arrays = {k: [] for k in keys_to_transfer}
        for i in ind_in_cube:
            cube_symbols.append(neigh_symbols[int(i)])
            for key in keys_to_transfer:
                cube_arrays[key].append(neigh_arrays[key][int(i)])

        # find atoms within rc
        ind_in_rc = np.where(np.linalg.norm(cube_disp, 2, 1) <= rc)[0]
        sphere_disp = cube_disp[ind_in_rc, :]
        sphere_symbols = []
        sphere_arrays = {k: [] for k in keys_to_transfer}
        for i in ind_in_rc:
            sphere_symbols.append(cube_symbols[int(i)])
            for key in keys_to_transfer:
                sphere_arrays[key].append(cube_arrays[key][int(i)])

        # make new atoms object
        if extract_cube:
            new_pos = cube_disp
            new_symbols = cube_symbols
            ind_fix = ind_in_rc
            new_arrays = cube_arrays
        else:
            new_pos = sphere_disp
            new_symbols = sphere_symbols
            ind_fix = np.arange(new_pos.shape[0], dtype=int)
            new_arrays = sphere_arrays

        box_center = cell_norms / 2
        new_pos = new_pos + box_center

        new_atoms = Atoms(symbols=new_symbols,
                          positions=new_pos,
                          cell=new_cell,
                          pbc=True)
        for key, arr in new_arrays.items():
            # data was saved as list of 1d arrays convert to 2D
            new_atoms.set_array(key, np.array(arr))
        # check info dict for any keys related to the keys_to_transfer
        new_info_dict = {}
        new_metadata_dict = {}
        for output_key in [x.rsplit('_', 1)[0] for x in keys_to_transfer]:
            for info_key in original_atoms.info:
                if output_key in info_key:
                    new_info_dict[info_key] = original_atoms.info[info_key]
            if output_key in original_atoms.info[METADATA_KEY]:
                new_metadata_dict[output_key] = original_atoms.info[
                    METADATA_KEY][output_key]
        new_atoms.info = new_info_dict
        new_atoms.info[METADATA_KEY] = new_metadata_dict
        # add constraint
        from ase.constraints import FixAtoms
        c = FixAtoms(indices=ind_fix)
        new_atoms.set_constraint(c)

        subcells.append(new_atoms)

    return subcells


def find_central_atom(config: Atoms, side_size: float) -> int:
    """
    Find the central atom index in an extracted environment

    The extract_env function does not specify the index of the atom which was
    extracted. However, it is guaranteed to be in the center of the cell. This
    method uses this to find the index of the central atom.

    :param config: subcell within which the central atom will be found
    :param side_size: length of the cubic cell, half of which will be the
        central atom's coordinates in all three direction
    :returns: index of the central atom in the Atoms object
    """
    half_length = side_size / 2
    atom_found = False
    for i, atom_pos in enumerate(config.positions):
        for coord in atom_pos:
            if coord == half_length:
                atom_found = True
            else:
                atom_found = False
                break
        if atom_found:
            return i
