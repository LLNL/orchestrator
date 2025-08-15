import os
import glob
import numpy as np
from ase.io import read, write

from orchestrator.utils.data_standard import ENERGY_KEY, FORCES_KEY, STRESS_KEY


def ase_glob_read(root_dir, file_ext='.xyz', file_format='extxyz'):
    """
    Reads all ASE atoms objects in `root_dir` with the matching` file_ext.
    """

    if file_ext[0] != '.':
        file_ext = '.' + file_ext

    images = []
    for f in sorted(glob.glob(os.path.join(root_dir, f'*{file_ext}'))):
        images += safe_read(f, format=file_format)

    return images


def try_loading_ase_keys(images):
    """
    Try to populate energy/forces/stress fields, in case they weren't
    loaded due to changes in ASE >= 3.23
    """
    if not isinstance(images, list):
        images = [images]

    for atoms in images:
        try:
            atoms.info[ENERGY_KEY] = atoms.get_potential_energy()
        except Exception:
            pass

        try:
            atoms.arrays[FORCES_KEY] = atoms.get_forces()
        except Exception:
            pass

        try:
            atoms.info[STRESS_KEY] = atoms.get_stress()
        except Exception:
            pass

    return images


def safe_read(path, **kwargs):
    """
    This is a wrapper to ASE.read that attempts to load energy/forces/stress
    from the SinglePointCalculator.
    """
    return try_loading_ase_keys(read(path, **kwargs))


def safe_write(path, images, **kwargs):
    """
    This is a wrapper to ASE.write that **removes the SinglePointCalculator**
    from all atoms objects, if the calculator is attached. This is to
    avoid issues caused by ase>=3.23 which uses a dummy SinglePointCalculator
    to store energy/forces/stress keys.
    """
    if not isinstance(images, list):
        images = [images]

    # Note: some Oracles may have valid calculators that should NOT be removed
    # e.g. "KIMModelCalculator"
    from ase.calculators.singlepoint import SinglePointCalculator
    for atoms in images:
        if isinstance(atoms.calc, SinglePointCalculator):
            atoms.calc = None
    write(path, images, **kwargs)


def sort_configs_and_tag_atoms(list_of_atoms, id_key='co-id'):
    """
    Sorts the configurations by their ID, and assigns unique tags to each atom.
    Intended to be used for error trajectory logging. The tags will be stored
    under the atoms.arrays['atom_id'] field.

    :param list_of_atoms: the atoms to be sorted
    :type list_of_atoms: list
    :param id_key: the key used for sorting the atoms. Must exist in atoms.info
        dict. Default is 'co_id'.
    :type id_key: str
    """

    sorted_atoms = sorted(list_of_atoms, key=lambda atoms: atoms.info[id_key])

    counter = 0
    for atoms in sorted_atoms:
        n = len(atoms)
        atoms.arrays['atom_id'] = np.arange(counter, counter + n)
        counter += n

    return sorted_atoms
