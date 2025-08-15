from os import system
from os.path import abspath, join, isdir
from ase.io import read
from .simulator_base import Simulator
from ..utils import templates
from ..utils.data_standard import METADATA_KEY
from ..utils.input_output import safe_write


class LAMMPSSimulator(Simulator):
    """
    Class for preparing input, running, and processing LAMMPS calculations

    Responsible for creating lammps and configuration input files, providing
    commands to run LAMMPS, linking KIM potentials to the LAMMPS run, and
    parsing the output to extract atomic configurations from trajectories

    :param simulator_args: dictionary of parameters to instantiate the
        Simulator, such as code_path (executable to use), elements (list of
        elements present in the simulation), and input_template (the path to an
        input template to build from)
    :type simulator_args: dict
    """

    def __init__(self, simulator_args):
        """
        Class for preparing input, running, and processing LAMMPS calculations

        :param simulator_args: dictionary of parameters to instantiate the
            Simulator, such as code_path (executable to use), elements (list of
            elements present in the simulation), and input_template (the path
            to an input template to build from)
        :type simulator_args: dict
        """
        self.code_path = simulator_args.get('code_path')
        if self.code_path is None:
            raise KeyError(('A path to the LAMMPS executable (code_path) must '
                            'be provided in simulator_args to instantiate a '
                            'LAMMPSSimulator'))

        self.elements = simulator_args.get('elements')
        if self.elements is None:
            raise KeyError(('A list of elements (elements) present in the '
                            'simulations must be provided in simulator_args to'
                            'instantiate a LAMMPSSimulator'))

        self.input_template = simulator_args.get('input_template')
        if self.input_template is None:
            raise KeyError(('A path to an input template (input_template) must'
                            ' be present in simulator_args to instantiate a '
                            'LAMMPSSimulator'))

        super().__init__(simulator_args)

    def write_input(self, run_path, input_args, input_file_name):
        """
        generate an input file for running a simulator calculation

        generate an input file using the ``input_template`` and ``input_args``
        for the given structural configuration, written as an external file by
        :meth:`write_initial_config`

        :param run_path: root path where simulations will run
        :type run_path: str
        :param input_args: additional arguments for the template, model
            specific
        :type input_args: dict
        :param input_file_name: name for the input file
        :type input_file_name: str
        """
        if input_file_name is None:
            input_file_name = 'lammps.in'
        input_file = templates.Templates(self.input_template, run_path,
                                         input_file_name)
        patterns = list(input_args.keys())
        replacements = list(input_args.values())
        file_name = input_file.replace(patterns, replacements)
        self.logger.info(f'LAMMPS input written to {run_path}/{file_name}')

    def write_initial_config(self, run_path, atoms):
        """
        Write LAMMPS conf file - initial condintion for simulation

        In addition to the lammps input file, the inital configuration is
        specified in the conf.lmp file. Conf.lmp defines the atomic positions
        cell parameters, and atomic types, and is saved in the run_path
        Assuming the cell matrix follows Lammps convention:
        https://docs.lammps.org/Howto_triclinic.html

        :param run_path: path where the configuration file will be written
        :type run_path: str
        :param atoms: the configuration to write
        :type atoms: Atoms
        """
        safe_write(join(run_path, 'conf.lmp'), atoms, format='lammps-data')
        self.logger.info((f'Completed writing of the initial configuration '
                          f'file to {run_path}/conf.lmp'))

    def get_run_command(self, args=None):
        """
        return the command to run a LAMMPS calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Simulator, which the
        :class:`~orchestrator.workflow.workflow_base.Workflow` will execute in
        the proper ``run_path``. The args dictionary can be used to pass the
        GPU flag, ``gpu_use``, to format the run command for GPU execution.

        :param args: dictionary for parameters to decorate or enable the run
            command. GPU command is selected with ``gpu_use`` set to True in
            ``args``. |default| ``None``
        :type args: dict
        :returns: command to run the simulator
        :rtype: str
        """
        if args is None:
            args = {}
        gpu_use = args.get('gpu_use', False)
        input_file = args.get('input_file_name', 'lammps.in')

        if gpu_use:
            num_gpu = args.get('num_gpu', 1)
            command = (f'{self.code_path} -sf kk -k on g {num_gpu} t {num_gpu}'
                       f' -pk kokkos newton on neigh half '
                       f'-in {input_file} -log lammps.out')
        else:
            command = f'{self.code_path} -in {input_file} -log lammps.out'
        return command

    def parse_for_storage(self, run_path):
        """
        process LAMMPS output to extract data in the format for Storage

        Typically, the output of interest from simulators are the calculation
        cell and atomic coordinates and type. However, additional information
        could also be extracted as properties in the ASE Atoms object.

        :param run_path: directory where the lammps output file resides
        :type run_path: str
        :returns: list of ASE Atoms of the configurations and any attached
            properties. Metadata with the configuration source information is
            attached to the METADATA_KEY in the info dict.
        :rtype: Atoms
        """
        output_file = 'dump.lammpstrj'
        full_path = run_path + '/' + output_file

        # index = ':' will return all trajectories, and always a list
        trajectory = read(full_path, index=':', format='lammps-dump-text')
        for i, config in enumerate(trajectory):
            atom_ids = config.get_atomic_numbers()
            atom_labels = self._convert_integer_to_label(atom_ids)
            config.set_chemical_symbols(atom_labels)
            # each configuration should have it's source recorded
            config.info[METADATA_KEY] = {
                'data_source': abspath(full_path),
                'config_index': i
            }
        return trajectory

    def load_potential(self, run_path, model_path):
        """
        set up the potential to be used at run_path

        Make the trained model accessible for simulations, i.e. through loading
        a KIM potential or ensuring the potential files are present in the
        requisite folder. If model path is not provided, then the code will
        assume that the model has been loaded in the user enviroment of KIM and
        is accessible from outside the current directory.

        :param run_path: root path where simulations will run and potential
            should be loaded/linked
        :type run_path: str
        :param model_path: path where the model to load is stored
        :type model_path: str
        """
        if model_path is None:
            self.logger.info(('Model path not provided, model should be '
                              'installed at the user level'))
        else:
            if isdir(model_path):
                # if the potential is a directory with files (i.e. KIM) then
                # link the file to run_path with the same name
                potential_name = model_path.strip('/').split('/')[-1]
                system(f'ln -s `realpath {model_path}` ./{run_path}/'
                       f'{potential_name}')
            else:
                # if the potential is not a directory, it is a (set of)
                # file(s) which need to be linked
                system(f'ln -s `realpath {model_path}`* ./{run_path}/')

    # lammps specific helper function
    def _convert_label_to_integer(self, atomic_labels):
        """
        Converts atomic label (string) to integer

        LAMMPS identifies atoms by integer indexes, but atoms are typically
        identified by their chemical symbol strings. This helper function
        converts from the string labels to integers in a repeatable and
        consistent manner.

        :param atomic_labels: chemical identity of each atom (str)
        :type atomic_labels: list of str
        :returns: list of int mapped by ``elements``
        :rtype: list
        """
        return [self.elements.index(k) + 1 for k in atomic_labels]

    def _convert_integer_to_label(self, atomic_ids):
        """
        Converts atomic ids (integer) to labels (string)

        LAMMPS identifies atoms by integer indexes, but atoms are typically
        identified by their chemical symbol strings. This helper function
        converts from integers to the string labels in a repeatable and
        consistent manner.

        :param atomic_ids: chemical ID each atom (int)
        :type atomic_ids: list of int
        :returns: list of str mapped by ``elements``
        :rtype: list
        """
        return [self.elements[k - 1] for k in atomic_ids]
