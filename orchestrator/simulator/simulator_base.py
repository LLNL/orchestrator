from abc import ABC, abstractmethod
from datetime import datetime
from random import randrange, seed
from ..workflow.factory import workflow_builder
from ..utils.recorder import Recorder
from ..utils.exceptions import UnidentifiedPathError, UnidentifiedStorageError
from ..utils.input_output import ase_glob_read
from ..utils.isinstance import isinstance_no_import


class Simulator(Recorder, ABC):
    """
    Abstract base class to manage and run simulations (exploration)

    The simulator class manages the construction and parsing of molecular
    dynamics calculations using interatomic potentials. The input will
    typically consist of an initial atomic configuration and calculation
    parameters (including the potential to use), while the output will include
    frames or configurations from the simulation as well as information such as
    the energy of the system, forces on each atom, and/or the stress on the
    cell, amongst others.

    :param simulator_args: dictionary of parameters to instantiate the
        Simulator, such as code_path (executable to use), elements (list of
        elements present in the simulation), and input_template (the path to an
        input template to build from)
    :type simulator_args: dict
    """

    def __init__(self, simulator_args):
        """
        Abstract base class to manage and run simulations (exploration)

        :param simulator_args: dictionary of parameters to instantiate the
            Simulator, such as code_path (executable to use), elements (list of
            elements present in the simulation), and input_template (the path
            to an input template to build from)
        :type simulator_args: dict
        """
        super().__init__()
        self.simulator_args = simulator_args
        #: default workflow to use within the Simulator class
        self.default_wf = workflow_builder.build(
            'LOCAL',
            {'root_directory': './simulator'},
        )
        # this flag should be set to True externally
        self.external_setup = False
        # if external_setup is set to True, then external_func needs to be set
        # as a function
        self.external_func = None

    def get_init_configs_from_path(self,
                                   config_path,
                                   file_ext='.xyz',
                                   file_format='extxyz'):
        """
        get the initial configuration for the simulator input from path

        This function loads the configurations present in the ``config_path``
        and all of its sub-directories into a list of ASE Atoms, which is
        returned. Assumes files are stored in the extended xyz format. This
        function should only be used if configurations cannot be added to
        Storage.

        :param config_path: path of the root directory where configuration
            files are stored (extended xyz format)
        :type config_path: str
        :param file_ext: the file extension. Default is '.xyz'
        :type file_ext: str
        :param file_ext: the file format. Default is 'extxyz'
        :type file_ext: str
        :returns: dataset as list of Atoms objects
        :rtype: list
        """
        return ase_glob_read(config_path, file_ext, file_format)

    def run(
        self,
        path_type,
        model_path,
        input_args,
        init_config_args,
        workflow=None,
        job_details=None,
    ):
        """
        setup and execute a Simulator calculation

        Prepare input file and initial configuration. Execute the code (run
        simulation), returning the ``job_id`` for tracking purposes

        :param path_type: specifier for the workflow path, to differentiate
            calculation types
        :type path_type: str
        :param model_path: path where the potential file(s) is stored
        :type model_path: str
        :param input_args: input arguments to fill out the input template file
        :type input_args: dict
        :param init_config_args: dictionary containing information to specify
            how the configuration should be setup for the run. Key:value pairs
            are 'make_config': boolean if run() should create the initial
            configuration [if false, the other keys are not needed],
            'config_handle': identifier to retrieve the configuration,
            'storage': storage module for configuration options to be retrieved
            from. Alternatively, set to 'path' if config_handle is a path where
            configs should be read from,
            'random_seed': if selecting the configuration from a set, an int
            random seed can be specified to enable reproducability
        :type init_config_args: dict
        :param workflow: the workflow for managing job submission, if none are
            supplied, will use the default workflow defined in this class
            |default| ``None``
        :type workflow: Workflow
        :param job_details: dict that includes any additional parameters for
            running the job (passed to
            :meth:`~orchestrator.workflow.workflow_base.Workflow.submit_job`)
            |default| ``None``
        :type job_details: dict
        :returns: calculation ID
        :rtype: int
        """
        module_name = self.__class__.__name__
        if workflow is None:
            workflow = self.default_wf
        if job_details is None:
            job_details = {}
        run_path = workflow.make_path(module_name, path_type)

        self.load_potential(run_path, model_path)

        make_config = init_config_args.get('make_config', True)
        if make_config:
            self.logger.info(f'{module_name} is creating the configuration')
            config_handle = init_config_args.get('config_handle')
            storage = init_config_args.get('storage')
            random_seed = init_config_args.get('random_seed')
            if storage == 'path':
                self.logger.info('Reading configurations from path, consider '
                                 'using Storage instead')
                init_configs = self.get_init_configs_from_path(config_handle)
            elif isinstance_no_import(storage, 'Storage'):
                self.logger.info(
                    f'Reading configurations from dataset {config_handle} in '
                    f'database {storage.database_name} from '
                    f'{storage.__class__.__name__}')
                init_configs = storage.get_data(config_handle)
            else:
                raise UnidentifiedStorageError(
                    f'Simulator cannot use {storage}')

            if random_seed is not None:
                self.logger.info(
                    f'Initializing random seed with seed: {random_seed}')
                seed(random_seed)
            ind = randrange(0, len(init_configs))
            self.logger.info(f'Using random index: {ind}')
            self.write_initial_config(run_path, init_configs[ind])

        input_file_name = job_details.get('input_file_name')
        self.write_input(run_path, input_args, input_file_name)

        if self.external_setup:
            self._external_calculation_setup(run_path)

        simulator_command = self.get_run_command(job_details)
        calc_id = workflow.submit_job(simulator_command, run_path, job_details)

        return calc_id

    def save_configurations(
        self,
        path_ids,
        storage,
        dataset_handle=None,
        workflow=None,
    ):
        """
        save the configurations associated with path_ids to storage

        :param path_ids: list of ``calc_ids`` or explicit paths associated with
            simulator jobs. If ``calc_ids`` are supplied, the path is extracted
            from the :class:`~orchestrator.workflow.workflow_base.JobStatus`.
            Otherwise it is taken verbatim as the input.
        :type path_ids: list of int or str
        :param storage: the storage module where the configurations will be
            saved.
        :type storage: Storage
        :param dataset_handle: the handle to identify where in Storage the
            configurations should be saved. If ``None``, then the class default
            (date stamped) is used. |default| ``None``
        :type dataset_handle: str
        :param workflow: the workflow for managing job submission, if none are
            supplied, will use the default workflow defined in this class.
            Should be consistent with the workflow supplied for any run calls.
            |default| ``None``
        :type workflow: Workflow
        :returns: handle of the dataset which includes the new configurations
        :rtype: str
        """
        if not isinstance(path_ids, list):
            path_ids = [path_ids]
        data_paths = []
        for path_id in path_ids:
            if isinstance(path_id, int):
                self.logger.info(
                    'Supplied path ID is calc ID, extracting paths')
                if workflow is None:
                    workflow = self.default_wf
                calc_path = workflow.get_job_status(path_id).path
                data_paths.append(calc_path)
            elif '/' in path_id or '.' in path_id:
                self.logger.info('Reading explicit paths for parsing output')
                calc_path = path_id
                data_paths.append(calc_path)
            else:
                raise UnidentifiedPathError((f'Supplied path_id: "{path_id}" '
                                             f'is not in a recognized format'))

        self.logger.info((f'Saving {len(data_paths)} '
                          f'{self.__class__.__name__} trajectories'))

        data = []
        for run_path in data_paths:
            # parsed data is always a single atoms object from the Oracle
            data.extend(self.parse_for_storage(run_path))

        current_date = datetime.today().strftime('%Y-%m-%d')
        dataset_metadata = {
            'description': (f'data generated by {self.__class__.__name__} on '
                            f'{current_date}')
        }

        if dataset_handle is None:
            dataset_handle = storage.generate_dataset_name(
                f'{self.__class__.__name__}_dataset',
                f'{current_date}',
                check_uniqueness=True,
            )

        # this logic assumes colabfit style naming conventions for dataset IDs
        # if other storage is implemented, should switch to a more generic
        # "if dataset exists" logic check
        if dataset_handle[:3] == 'DS_':
            # handle is a colabfit ID, dataset exists
            new_handle = storage.add_data(dataset_handle, data,
                                          dataset_metadata)
        else:
            # handle is a name, create new dataset
            new_handle = storage.new_dataset(dataset_handle, data,
                                             dataset_metadata)

        return new_handle

    def _external_calculation_setup(self, path):
        """
        utility function to call an attached external function for input setup

        If self.external_setup is set to ``True``, then this method will be
        called. The external code which set the setup flag to True should also
        set the external_func to the desired function. It should take in the
        path to write output as its only parameter.

        :param path: location where input files should be written, passed to
            the attached external_func
        :type path: str
        """
        if callable(self.external_func):
            self.external_func(path)
        else:
            raise AttributeError('Set external_func to a callable function!')

    @abstractmethod
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
        pass

    @abstractmethod
    def write_initial_config(self, run_path, atoms):
        """
        generate an input file for the initial structural configuration

        Codes such as LAMMPS have an input file specifying the calculation and
        a separate input file specifying the structural configuration. This
        method generates the latter file.

        :param run_path: path where the configuration file will be written
        :type run_path: str
        :param atoms: the ASE Atoms object
        :type pos: Atoms
        """
        pass

    @abstractmethod
    def get_run_command(self, args=None):
        """
        return the command to run a simulator calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Simulator, which the
        :class:`~orchestrator.workflow.workflow_base.Workflow` will execute in
        the proper ``run_path``. The args dictionary can be used to pass any
        necessary extra parameters to the specific implementations.

        :param args: dictionary for parameters to decorate or enable the run
            command |default| ``None``
        :type args: dict
        :returns: command to run the simulator
        :rtype: str
        """
        pass

    @abstractmethod
    def parse_for_storage(self, run_path):
        """
        process calculation output to extract data in a consistent format

        Typically, the output of interest from simulators are the calculation
        cell and atomic coordinates and type. However, additional information
        could also be extracted as properties in the ASE Atoms object.

        :param run_path: directory where the simulator output file resides
        :type run_path: str
        :returns: list of ASE Atoms of the configurations and any attached
            properties. Metadata with the configuration source information is
            attached to the METADATA_KEY in the info dict.
        :rtype: Atoms list
        """
        pass

    @abstractmethod
    def load_potential(self, run_path, model_path):
        """
        set up the potential to be used at run_path

        Make the trained model accessible for simulations, i.e. through loading
        a KIM potential or ensuring the potential files are present in the
        requisite folder

        :param run_path: root path where simulations will run and potential
            should be loaded/linked
        :type run_path: str
        :param model_path: path where the model to load is stored
        :type model_path: str
        """
        pass
