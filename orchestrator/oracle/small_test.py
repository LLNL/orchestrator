import numpy as np
from os.path import abspath
from ase.io import read
from .oracle_base import Oracle
from ..utils.templates import Templates


class EspressoOracle(Oracle):
    """
    Class for creating, running, and parsing quantum espresso calculations

    Resposible for making PWscf input file, run the code and extract energy,
    forces, and stress tensor from PWscf output

    :param code_path: path of the code executable
    :type code_path: str
    :param input_template: path to input template made by the user, this input
        is required for EspressoOracle |default| ``None``
    :type input_template: str
    :param potential: name of the potential (KIM potential, currently) to use
        in a calculation, not used by this module. |default| ``None``
    :type potential: str
    """
    def __init__(self, code_path, input_template=None, potential=None):
        """
        Class for creating, running, and parsing quantum espresso calculations

        :param code_path: path of the code executable
        :type code_path: str
        :param input_template: path to input template made by the user, this
            input is required for EspressoOracle |default| ``None``
        :type input_template: str
        :param potential: name of the potential (KIM potential, currently) to
            use in a calculation, not used by this module. |default| ``None``
        :type potential: str
        """
        super().__init__(code_path, input_template, potential)
        if self.input_template is None:
            raise Exception(('For building an EspressoOracle, input_template '
                             'must be provided!'))
        self.convert_units = self.ry_to_metal_units()
        self.output_filename = 'espresso.out'

    def write_atomic_positions(self, pos, atype):
        """
        Write atomic coordinates in PWscf input file format

        Helper function to write atomic coordinates in the format of
        quantum espresso DFT calculations. Returns the combined string.

        :param pos: atomic positions in cartesian coords in units of angstroms,
            array will be size [Nx3]
        :type pos: np.ndarray
        :param atype: atomic symbols cooresponding to each position entry in
            pos, array of size [N]
        :type atype: np.ndarray
        :returns: formatted atomic coordinates for QE input
        :rtype: str
        """
        pos_ = ''
        for iat in range(pos.shape[0]):
            pos_ += f'{atype[iat]} {pos[iat][0]} {pos[iat][1]} {pos[iat][2]}\n'
        return pos_

    def write_cell(self, cell):
        """
        Write cell lattice in PWscf input file format

        Helper function to write unit cell information in the format of
        quantum espresso DFT calculations. Returns the formatted string

        :param cell: calculation unit cell in Angstrom ([3x3] array)
        :type cell: np.ndarray
        :returns: formatted cell for QE input
        :rtype: str
        """
        cell_ = ''
        for i in range(3):
            cell_ += f'{cell[i][0]} {cell[i][1]} {cell[i][2]}\n'
        return cell_

    def write_input(self, run_path, input_args, config):
        """
        Write PWscf input file

        This is the main method that utilizes the helper functions and the
        input template to write the full QE input file.

        :param run_path: directory path where the file is written
        :type run_path: str
        :param input_args: dictionary of input arguments for the espresso
            input file template
        :type input_args: dict
        :param config: data specifying an atomic configuration. Each config is
            an ordered list of: atomic coordinates [natoms,3], cell lettice
            [3,3], and atomic labels [natoms]
        :type config: list
        :returns: name of written input file
        :rtype: str
        """
        # currently Templates don't modify other calculation parameters such as
        # kgrig, energy cutoffs, pseudos, etc
        input_file = Templates(self.input_template, run_path)
        patterns = ['natoms', 'ntype', 'cell_parameters', 'atomic_positions']
        pos, cell, atype = config
        replacements = [
            pos.shape[0],
            np.unique(atype).size,
            self.write_cell(cell),
            self.write_atomic_positions(pos, atype)
        ]
        if input_args is not None:
            input_patterns = list(input_args.keys())
            patterns += input_patterns
            input_replacements = list(input_args.values())
            replacements += input_replacements

        file_name = input_file.replace(patterns, replacements)
        self.logger.info(f'Espresso input written to {run_path}/{file_name}')
        return file_name

    def get_run_command(self, args={}):
        """
        return the command to run a quantum espresso calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Oracle, which the
        :class:`~orchestrator.workflow.workflow_base.Workflow` will execute in
        the proper ``run_path``. ``args`` include parallelization schemes for
        espresso, including: 'nimage', 'npool', 'nband', 'ntg', 'ndiag'. Each
        will be set to 1 if not specified.

        :param args: dictionary for parameters to decorate or enable the run
            command |default| ``{}``
        :type args: dict
        :returns: single line string with code execution statement
        :rtype: str
        """

        nimage = args.get('nimage', 1)
        npools = args.get('npool', 1)
        nband = args.get('nband', 1)
        ntg = args.get('ntg', 1)
        ndiag = args.get('ndiag', 1)
        input_file = args.get('input_file', 'espresso.in')

        command = (f'{self.code_path} -nimage {nimage} -npools {npools} '
                   f'-nband {nband} -ntg {ntg} -ndiag {ndiag} < {input_file} >'
                   f' {self.output_filename}')
        return command

    def parse_for_storage(self, run_path):
        """
        process calculation output to extract data in a consistent format

        Parse the output from the Espresso calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations, cell info,
        and possibly energies, forces, and stresses. Units are: total system
        energy in eV, forces on each atom in eV/A, and stress on the system in
        eV/A^3

        :param run_path: directory where the oracle output file resides
        :type run_path: str
        :returns: tuple of Atoms of the configurations and attached properties
            as well as a dictionary of metadata that should be stored with the
            configuration.
        :rtype: tuple (Atoms, dict)
        """
        data_file = f'{run_path}/{self.output_filename}'
        atoms = read(data_file, format='espresso-out')
        atoms.info['energy'] = atoms.get_potential_energy()
        atoms.set_array('forces', atoms.get_forces())
        metadata = {'data_source': abspath(data_file)}
        return (atoms, metadata)

    def parse_output(self, run_path):
        """
        Extract energy, forces and stress tensor from QE output

        Parse through the QE output to extract the ground truth data from the
        converged calculation. Units are: total system energy in eV, forces on
        each atom in eV/A, and stress on the system in eV/A^3. This method is
        largely deprecated after the storage module (use parse_for_storage
        instead).

        :param run_path: directory where the espresso output file resides
        :type run_path: str
        :returns: data structures including the energy [float], forces [Nx3],
            and stress [3x3] of the configuration
        :rtype: tuple
        """
        pattern_atoms = 'number of atoms/cell'
        pattern_energy = '!    total energy'
        pattern_forces = 'Forces acting'
        pattern_stress = 'total   stress'
        stress = np.zeros((3, 3))

        with open(run_path + '/' + self.output_filename) as qe_output:
            lines = qe_output.readlines()

        for i in range(len(lines)):
            if (lines[i].find(pattern_atoms) != -1):
                natoms = int(lines[i].split()[4])
                forces = np.zeros((natoms, 3))
            elif (lines[i].find(pattern_energy) != -1):
                energy = float(lines[i].split()[4])
            elif (lines[i].find(pattern_forces) != -1):
                for j in range(natoms):
                    forces[j] = [
                        float(ff) for ff in lines[i + 2 + j].split()[6:]
                    ]
            elif (lines[i].find(pattern_stress) != -1):
                for j in range(3):
                    stress[j] = [
                        float(st) for st in lines[i + 1 + j].split()[:3]
                    ]

        energy *= self.convert_units['energy']
        forces = np.array(forces) * self.convert_units['force']
        stress = np.array(stress) * self.convert_units['stress']

        return energy, forces, stress

    # Espresso specific helper for unit conversions
    def ry_to_metal_units(self):
        """
        Constants to convert Rydberg units to metal units

        Espresso specific helper for unit conversions as QE outputs in Ry and
        bohr instead of eV and angstrom
        """

        distance = 0.529177  # Bohr to angstrom
        energy = 13.6057039763  # Ry to eV
        force = energy / distance  # Ry/bohr to eV/angstrom
        stress = force / distance**2  # Ry/bohr^3 to eV/angstrom^3

        return {
            'distance': distance,
            'energy': energy,
            'force': force,
            'stress': stress
        }
