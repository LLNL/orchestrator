from abc import ABC, abstractmethod
from ase import Atoms
from datetime import datetime
from copy import deepcopy
from typing import Optional, Union
from ..workflow import Workflow, workflow_builder
from ..storage import Storage
from ..utils.recorder import Recorder
from ..utils.exceptions import UnidentifiedPathError, DatasetDoesNotExistError
from ..utils.data_standard import METADATA_KEY


class Oracle(Recorder, ABC):
    """
    Abstract base class for ground truth calculations

    The oracle class manages the construction and parsing of ground truth
    calculations to provide training or reference data. The input will
    typically consist of an atomic configuration and calculation parameters,
    the output will include the energy of the system, forces on each atom,
    and/or the stress on the cell.
    """

    def __init__(self, **kwargs):
        """
        set variables and initialize the recorder and default workflow

        :param kwargs: arguments for instantiating Oracle. These arguments
            are defined by the concrete classes, and include items such as the
            path to an input file template (input_template), potential name
            (potential), and executable (code_path)
        :type oracle_args: dict
        """
        super().__init__()
        self.remaining_args = kwargs
        #: default workflow to use within the Oracle class
        self.default_wf = workflow_builder.build(
            'LOCAL',
            {
                'root_directory': './oracle',
                'checkpoint_name': 'default_oracle_workflow',
                'job_record_file': './default_oracle_job_record.pkl',
            },
        )

    def run(
        self,
        path_type: str,
        input_args: dict[str, Union[int, float, str]],
        configs: list[Atoms],
        workflow: Optional[Workflow] = None,
        job_details: Optional[dict[str, Union[int, float, str]]] = None,
    ) -> list[int]:
        """
        Main function to call ground truth calculation

        The run method includes half of the main functionality of the oracle,
        taking atomic configurations as input and handling the submission of
        calculations to obtain the ground truth data. Configs is a dataset of 1
        or more structures. run() will create independent jobs for
        each structure using the supplied workflow, with job_details
        parameterizing the job submission.

        :param path_type: specifier for the workflow path, to differentiate
            calculation types
        :type path_type: str
        :param input_args: input arguments to fill out the input file
        :type input_args: dict
        :param configs: list of configurations as ASE atoms to run ground truth
            calculations for
        :type configs: list
        :param workflow: the workflow for managing job submission, if none are
            supplied, will use the default workflow defined in this class
            |default| ``None``
        :type workflow: Workflow
        :param job_details: dict that includes any additional parameters for
            running the job (passed to
            :meth:`~orchestrator.workflow.workflow_base.Workflow.submit_job`)
            |default| ``None``
        :type job_details: dict
        :returns: a list of calculation IDs from the workflow.
        :rtype: list
        """
        module_name = self.__class__.__name__
        calc_ids = []
        num_calcs = len(configs)
        self.logger.info(f'Spinning up {num_calcs} {module_name} calculations')

        if workflow is None:
            workflow = self.default_wf
        if job_details is None:
            job_details = {}

        path_base = workflow.make_path_base(module_name, path_type)
        self.path_base = path_base
        for frame in range(0, num_calcs):
            run_path = workflow.make_path(module_name, path_type)
            file_name = self.write_input(run_path, input_args, configs[frame])
            modified_job_details = deepcopy(job_details)
            modified_job_details['run_path'] = run_path
            modified_job_details['input_file'] = file_name
            # if metadata is attached to the config, link it to the job
            if configs[frame].info.get(METADATA_KEY) is not None:
                try:
                    modified_job_details['extra_args'][METADATA_KEY] = configs[
                        frame].info[METADATA_KEY]
                except KeyError:
                    modified_job_details['extra_args'] = {
                        METADATA_KEY: configs[frame].info[METADATA_KEY]
                    }
            oracle_command = self.get_run_command(**modified_job_details)
            calc_id = workflow.submit_job(
                oracle_command,
                run_path,
                job_details=modified_job_details,
            )
            calc_ids.append(calc_id)
        self.logger.info(f'Done generating {module_name} calculations')

        return calc_ids

    def save_labeled_configs(
        self,
        paths: list[Union[int, str]],
        storage: Storage,
        dataset_name: Optional[str] = None,
        dataset_handle: Optional[str] = None,
        workflow: Optional[Workflow] = None,
    ) -> str:
        """
        extract and save computed data to storage

        The save_labeled_configs method includes the other half of the main
        functionality of the Oracle. Once the calculations are complete, the
        data they generate must be integrated with the strucutral configuration
        in a consistent framework to be used for training. This is done by
        parsing and ingesting the configuration and attached data (energies,
        forces, stresses) into a dataset handled by the
        :class:`~orchestrator.storage.storage_base.Storage` module.

        :param paths: calc_ids or explicit paths associated with each config.
            If calc_ids are supplied, the path is extracted from the
            :class:`~orchestrator.workflow.workflow_base.JobStatus`. Calc IDs
            are generally prefered as they can also carry metadata with them.
        :param storage: specific module that handles the staroge of data.
        :param dataset_name: The name of the datset in the Storage where the
            configurations should be saved. If a dataset with this name already
            exists, it will be added to. Otherwise a new dataset with this name
            will be created. If ``None`` and dataset_handle is not provided,
            then the class default (date stamped) is used. |default| ``None``.
        :param dataset_handle: the handle to identify the dataset in Storage.
            If provided, will add data to this dataset. If ``None`` and
            dataset_name is not provided the class default name (date stamped)
            is used. |default| ``None``
        :param workflow: the workflow for managing job submission, if none are
            supplied, will use the default workflow defined in this class.
            Should be consistent with the workflow supplied for the run calls.
            |default| ``None``
        :returns: dataset handle
        """
        if workflow is None:
            workflow = self.default_wf

        if not isinstance(paths, list):
            raise TypeError('paths should be a list, instead received '
                            f'{type(paths).__name__}')
        if isinstance(paths[0], int):
            self.logger.info('Supplied paths are calc IDs, extracting paths')
            data_paths = []
            existing_metadata = []
            for id in paths:
                data_paths.append(workflow.get_job_path(id))
                existing_metadata.append(workflow.get_attached_metadata(id))
        elif '/' in paths[0] or '.' in paths[0]:
            self.logger.info('Reading explicit paths for labeling configs')
            data_paths = paths
            existing_metadata = [{} for _ in range(len(paths))]
        else:
            raise UnidentifiedPathError(
                'Supplied paths are not in a recognized format')

        self.logger.info((f'Labelling {len(data_paths)} '
                          f'{self.__class__.__name__} calculations'))

        data = []
        for run_path, old_metadata in zip(data_paths, existing_metadata):
            # parsed data is always a single atoms object from the Oracle
            config = self.parse_for_storage(run_path)
            # merge data source metadata to any existing metadata
            combined_metadata = old_metadata | config.info[METADATA_KEY]
            config.info[METADATA_KEY] = combined_metadata
            data.append(config)

        current_date = datetime.today().strftime('%Y-%m-%d')
        dataset_metadata = {
            'description': (f'data generated by {self.__class__.__name__} on '
                            f'{current_date}')
        }

        if dataset_name is None and dataset_handle is None:
            # no names or IDs provided, make new dataset name
            dataset_name = storage.generate_dataset_name(
                f'{self.__class__.__name__}_dataset',
                f'{current_date}',
                check_uniqueness=True,
            )
        elif dataset_name:
            # only dataset name provided
            try:
                # since extracted, we do not need to validate form
                dataset_handle = storage._get_id_from_name(dataset_name)
            except DatasetDoesNotExistError:
                pass
        else:
            # dataset handle provided, ensure it is of correct form
            if dataset_handle[:3] != 'DS_':
                raise ValueError(
                    'dataset handles should be in format DS_************_#')

        if dataset_handle:
            # handle is a colabfit ID, dataset exists
            new_handle = storage.add_data(dataset_handle, data,
                                          dataset_metadata)
        else:
            # handle is a name, create new dataset
            new_handle = storage.new_dataset(dataset_name, data,
                                             dataset_metadata)

        return new_handle

    def data_from_calc_ids(
            self,
            calc_ids: list[int] = None,
            workflow: Workflow = None) -> tuple[list[Atoms], dict]:
        """
        Given a list of calc_ids, will iterate over the list and make relevant
        checks to ensure the returned values have the same inputs.

        :param calc_ids: List of calculation ids, in this case pks from AiiDA,
            that are used to parse relevant data.
        :returns: List of Atoms object and a nested dictionary containing the
            input parameters of the code and the universal values.
        """

        if workflow is None:
            workflow = self.default_wf

        configs = []
        code_parameters = {}

        for calc_id in calc_ids:
            atoms = self.parse_for_storage(calc_id=calc_id, workflow=workflow)

            metadata = atoms.info[METADATA_KEY]
            parameters = metadata.pop('code_parameters', None)

            # Compare the input dictionaries of each calculation.
            if not code_parameters.get('code', None):
                code_parameters['code'] = parameters['code']
                code_parameters['universal'] = parameters['universal']
            else:
                if code_parameters['universal'] != parameters['universal']:
                    raise ValueError(
                        f'The provided pk values, {calc_ids}, have differing '
                        'universal parameters which is not currently '
                        'supported.')

        return configs, code_parameters

    @abstractmethod
    def write_input(
        self,
        run_path: str,
        input_args: dict,
        config: Atoms,
    ) -> str:
        """
        generate an input file for running the ground truth calculation

        This method will write the requisite input file in the run_path using
        the input_args of a given configuration. Specific implementations may
        leverage additional helper functions to construct the input.

        :param run_path: directory path where the file is written
        :type run_path: str
        :param input_args: input arguments for the oracle input file, typically
            controlled using a template
        :type input_args: dict
        :param config: the configuration as an Atoms object
        :type config: Atoms
        :returns: name of written input file
        :rtype: str
        """
        pass

    @abstractmethod
    def get_run_command(self, **job_details):
        """
        return the command to run an oracle calculation

        this method formats the run command based on the ``code_path`` internal
        variable set at instantiation of the Oracle, while the args dictionary
        can be used to pass any necessary extra parameters to the specific
        implementations.

        :param job_details: dictionary for parameters to decorate or enable
            the run command. Keys are defined in concrete classes. |default|
            ``None``
        :type args: dict
        :returns: implementation dependent
        :rtype: implementation dependent
        """
        pass

    @abstractmethod
    def parse_for_storage(self,
                          run_path: str = '',
                          calc_id: int = None,
                          workflow: Workflow = None) -> Atoms:
        """
        process calculation output to extract data in a consistent format

        Parse the output from the Oracle calculation into ASE Atoms objects.
        The resulting Atoms will include the atomic configurations, cell info,
        and possibly energies, forces, and stresses. Units are: total system
        energy in eV, forces on each atom in eV/A, and stress on the system in
        eV/A^3

        :param run_path: directory where the oracle output file resides.
        :param calc_id: Job ID of the calculation to parse.
        :param workflow: Workflow object of Orchestrator.
        :returns: Atoms of the configurations with attached properties and
            metadata.
        """
