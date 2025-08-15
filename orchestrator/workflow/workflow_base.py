from abc import ABC, abstractmethod
from copy import deepcopy
from ..utils.recorder import Recorder
from ..utils.restart import restarter
from ..utils.templates import Templates
from os import system, path, PathLike
import pickle
from time import sleep
from typing import Any, Optional, Union


class JobStatus:
    """
    data class for collecting meta-data about a job

    Minimum attribute set for a JobStatus object is ``path``, ``state``, and
    ``exit_code``, where ``path`` defines the location where inputs and outputs
    are stored, ``state`` defines the jobs status (i.e. done, pending, running)
    , and ``exit_code`` shows the result of the job once completed, with 0
    indicating success. Note that ``state`` can change over time and
    ``exit_code`` may not be known when the object is first created.
    """

    def __init__(self, path, state=None, exit_code=None):
        """
        :param path: directory path where inputs and outputs are stored for
            the job
        :type path: str
        :param state: jobs status (i.e. done, pending, running)
        :type state: str
        :param exit_code: result of the job
        :type exit_code: int or str
        """
        self.path = path
        self.state = state
        self.exit_code = exit_code


class Workflow(Recorder, ABC):
    """
    Abstract base class to manage workflows

    Responsibilities include directory creation, job creation, job status
    checking. A given workflow class will use the ``root_directory`` as the
    base for all calculation inputs and outputs in a directory hierarchy
    managed by :meth:`make_path_base` and :meth:`make_path`. ``workflow_args``
    provide a vehicle for modulating the workflow behavior, though are not
    strictly required. The ``counters`` and ``jobs`` dictionaries are also
    initialized at instantiation, which are used to internally track workflow
    components.
    """

    def __init__(
        self,
        root_directory: Optional[str] = './orchestrator_workflow',
        checkpoint_file: Optional[str] = './orchestrator_checkpoint.json',
        checkpoint_name: Optional[Union[str, None]] = None,
        job_record_file: Optional[Union[str, None]] = None,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

        :param root_directory: name of the directory under which all files and
            subdirectories will sit |default| './orchestrator_workflow'
        :type root_directory: str
        :param checkpoint_file: name of the checkpoint file to write restart
            information to |default| './orchestrator_checkpoint.json'
        :type checkpoint_file: str
        :param checkpoint_name: name of the restart block for this module in
            the checkpoint file |default| 'workflow'
        :type checkpoint_name: str
        :param job_record_file: name of the file to save the pickled jobs dict
            |default| './job_record.pkl'
        :type job_record_file: str
        """
        super().__init__()
        self.root_directory = root_directory
        self.checkpoint_file = checkpoint_file
        if checkpoint_name is None:
            checkpoint_name = self.__class__.__name__
        self.checkpoint_name = checkpoint_name
        if job_record_file is None:
            job_record_file = "./" + self.__class__.__name__ +\
                "_job_record.pkl"
        self.job_record_file = job_record_file
        self.counters = {}
        self.jobs = {}
        self.problematic_states = [
            'unknown',
            'error',
            'done_timeout',
            'done_cancelled',
            'done_other',
            'done_unknown',
        ]
        self.new_counters = False
        self.restart_workflow()

    def make_path_base(
        self,
        module: str,
        path_type: str,
    ) -> Union[str, PathLike]:
        """
        Create the base directory for data and run outputs, excluding counter

        make_path_base generates a hierarchical directory structure within the
        root directory. The path hierarchy is root/module/path_type, where the
        ``module`` is the orchestrator module responsible for the call,
        ``path_type`` is the type of calculation or job within that module.
        This is a shortened version of :meth:`make_path`, omitting the count at
        the end of the directory path. The method returns the path as a string

        :param module: module requesting the path
        :type module: str
        :param path_type: "type" or purpose of the path, i.e. training_data,
            trajectories, ground_truth, etc. This is typically supplied in an
            input file
        :type path_type: str
        :returns: created path name
        :rtype: str
        """
        dir_name = f'{self.root_directory}/{module}/{path_type}'
        system(f'mkdir -p {dir_name} 2> /dev/null')

        return dir_name

    def make_path(self, module: str, path_type: str) -> Union[str, PathLike]:
        """
        Create the directory for data and run outputs to be located

        make_path generates a hierarchical directory structure within the root
        directory. The path hierarchy is root/module/path_type/count, where the
        ``module`` is the orchestrator module responsible for the call,
        ``path_type`` is the type of calculation or job within that module, and
        count is the increment of that specific calculation type. The method
        returns the generated path as a string

        :param module: the module requesting the path
        :type module: str
        :param path_type: "type" of the path, i.e. the purpose
        :type path_type: str
        :returns: created path name
        :rtype: str
        """
        counter_key = module + '.' + path_type
        if self.counters.get(counter_key) is None:
            self.counters[counter_key] = 0

        dir_name = (f'{self.root_directory}/{module}/{path_type}/'
                    f'{self.counters[counter_key]:05d}')
        system(f'mkdir -p {dir_name} 2> /dev/null')

        self.counters[counter_key] += 1
        self.new_counters = True
        self.checkpoint_workflow()  # necessary for local
        return dir_name

    def get_job_status(self, job_handle: int) -> JobStatus:
        """
        Queries the status of a job handle.

        Returns :class:`~JobStatus`, which has (minimally) attributes: ``path``
        , ``state``, and ``exit_code``, which correspond to the location of the
        job, the job state (i.e. completed), and an exit code that is 0 if
        successful and a flag with information if not.

        :param job_handle: job ID originally returned from :meth:`~submit_job`
        :type job_handle: int
        :returns: job's :class:`~JobStatus`
        :rtype: JobStatus
        """
        job_status = self.jobs.get(job_handle)
        if job_status is None:
            self.logger.info(f'Queried ID {job_handle} does not exist')
        return job_status

    def get_job_path(self, job_handle: int) -> Union[str, PathLike]:
        """
        returns the path where a specific job was run

        :param job_handle: job ID
        :type job_handle: int
        :returns: path where the job inputs/outputs are stored
        :rtype: str or PathLike
        """
        try:
            path = self.get_job_status(job_handle).path
        except AttributeError:
            path = None
            self.logger.info(
                f'Could not find path for job {job_handle}, return None')
        return path

    def get_attached_metadata(self, job_handle: int) -> dict[str, Any]:
        """
        returns the metadata associated with a specific job

        :param job_handle: job ID
        :type job_handle: int
        :returns: dict of metadata associated with the job
        :rtype: dict
        """
        try:
            metadata = self.get_job_status(job_handle).metadata
        except AttributeError:
            metadata = {}
        return metadata

    def get_all_statuses(self) -> dict[int, JobStatus]:
        """
        Returns information about all jobs from this Workflow.

        Returns a dictionary with ``job_handle``: ``status``, where
        ``job_handle`` is returned by :meth:`~submit_job` and ``status`` is a
        :class:`~JobStatus` instance

        :returns: a dictionary of :class:`~JobStatus` objects
        :rtype: dict
        """
        return self.jobs

    def job_done_file_present(self, job_id: int) -> bool:
        """
        check if the job directory contains the "job_done" file

        Use an empty file to give persistent indication if the job completed
        to avoid problems where job statuses are purged by the scheduler after
        a certain amount of time. The job templates will "touch job_done" at
        the end of the script to provide this persistent indicator.

        :param job_id: job ID to check
        :type job_id: int
        :returns: True if file exists, false otherwise
        :rtype: boolean
        """
        job_path = self.get_job_path(job_id)
        return path.isfile(f'{job_path}/job_done')

    def save_job_dict(self):
        """
        Serialize the job dictionary for persistant storage

        Write the jobs dict, containing all of the workflow's JobStatus objects
        to the job_record_file. The record file is overwritten each time.
        """
        copied_dict = False
        # if the file has already been written, we want to save the old version
        # in case something goes wrong with the dump
        if path.isfile(self.job_record_file):
            copied_dict = True
            system(f'mv {self.job_record_file} old_job_dict.pkl')

        with open(self.job_record_file, 'wb') as fout:
            pickle.dump(self.jobs, fout)

        # if nothing went wrong with dump, can delete the old job dict
        if copied_dict:
            system('rm old_job_dict.pkl')

    def read_job_dict(self):
        """
        Read a serialized job dictionary and set to the internal jobs dict
        """
        try:
            with open(self.job_record_file, 'rb') as fin:
                self.jobs = pickle.load(fin)
            self.logger.info(f'Read jobs dict from {self.job_record_file}')
        except FileNotFoundError:
            self.logger.info(f'{self.job_record_file} does not exist')

    @abstractmethod
    def checkpoint_workflow(self):
        """
        checkpoint the workflow module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    @abstractmethod
    def restart_workflow(self):
        """
        restart the workflow module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    @abstractmethod
    def block_until_completed(self, calc_ids: list[int]):
        """
        Function for enforcing synchronous execution

        Helper function to run when blocking behavior is desired. Will
        consistently check the queue until the specified job has completed.
        Implementation for synchronous workflows is just to pass

        :param calc_ids: list of job IDs of the calculations to check for
            completion. Can also pass a single ID.
        :type calc_ids: int or list
        """
        pass

    @abstractmethod
    def submit_job(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
    ) -> int:
        """
        Submits a job for running

        submit_job handles job submission for the modules and is the main
        interface for the workflows to be used. Inputs define the command to be
        executed for the job, location for the run, and details of the job
        resources. ``job_details`` inlcude ``dependencies`` of the job in the
        form of a list of job_ids, if the job is blocking (``synchronus``) or
        not, as a boolean, and an extra dictionary, ``extra_args``, to add
        flexibility for concrete implementations for further parameterizing the
        job. The ``dependencies`` are a list of job IDs which must have a
        successfully completed :class:`~JobStatus` for the present job to run.
        If one of the dependencies returns an error, this job will not run and
        the status will return an error. If synchronous is True, submit_job
        will not return until the job is completed. This may be the only
        behavior of some implementations. Returns a job handle that can be used
        to query status and to retrieve the present job's :class:`~JobStatus`,
        which is typically created by this method.

        :param command: command that defines the job to be executed
        :type command: implementation dependent
        :param run_path: location in the file system where inputs and outputs
            are to be accessed and stored
        :type run_path: str
        :param job_details: optional parameters for running the job, such as
            number of nodes, queue, etc. |default| ``None``
        :type job_details: dict
        :returns: return job ID to query this job status and location
        :rtype: int
        """
        pass


class HPCWorkflow(Workflow, ABC):
    """
    Generic (and abstract class) for workflows leveraging HPC schedulers

    HPCWorkflow defines the shared init args and restart functionality that
    LSF, Slurm, and other HPC schedulers require. It is not instantitated
    directly, but inherited.
    """

    def __init__(
        self,
        queue: str,
        account: str,
        walltime: Optional[Union[int, float]] = 60,
        nodes: Optional[int] = 1,
        tasks: Optional[int] = 1,
        tasks_per_node: Optional[int] = 1,
        wait_freq: Optional[int] = 60,
        **kwargs,
    ):
        """
        set variables and initialize the recorder

        The provided input arguments set the default parameters for the
        workflow, but can be overridden by values passed into ``job_details``
        dict provided to :meth:`submit_job`.

        :param queue: default name of the queue to submit to
        :type queue: str
        :param account: default name of the account for the job
        :type account: str
        :param walltime: default walltime for the job in minutes |default| 60
        :type walltime: int or float
        :param nodes: default number of nodes to request |default| 1
        :type nodes: int
        :param tasks: default number of tasks for a job |default| 1
        :type tasks: int
        :param tasks_per_node: default number of tasks per node for a job.
            Will not be used if tasks is explicitly set. |default| 1
        :type tasks_per_node: int
        :param wait_freq: the frequency with which squeue is called to get job
            status updates, in seconds |default| 60
        :type wait_freq: int
        :param kwargs: remaining keywords passed to parent: root_directory,
            checkpoint_file, checkpoint_name, and job_record_file
        :type kwargs: dict
        """
        self.default_queue = queue
        self.default_account = account
        # value in minutes
        self.default_walltime = walltime
        self.default_nodes = nodes
        self.default_tasks = tasks
        self.default_tasks_per_node = tasks_per_node
        self.unknown_job_id = 1000
        self.new_unknown_id = False

        # value in seconds
        self.synch_check_frequency = wait_freq
        super().__init__(**kwargs)

    @staticmethod
    def format_walltime(
        minutes: Union[float, int],
        include_seconds: bool,
    ) -> str:
        """
        utility function to create a time string based on input minutes

        different schedulers require time specifications with or without
        seconds, so this utility allows an integer input to be properly
        converted into a HH:MM or HH:MM:ss string

        :param minutes: number of minutes (can be fractional) to convert
        :type minutes: float or int
        :param include_seconds: whether to print out the seconds or not
        :type include_seconds: bool
        :returns: the formatted time string
        :rtype: str
        """
        if not include_seconds:
            minutes = round(minutes)
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        seconds = round((minutes - int(minutes)) * 60)
        if seconds == 60:
            seconds = 0
            remaining_minutes += 1
            if remaining_minutes == 60:
                remaining_minutes = 0
                hours += 1
        if include_seconds:
            return f'{hours:02}:{remaining_minutes:02}:{seconds:02}'
        else:
            return f'{hours:02}:{remaining_minutes:02}'

    def checkpoint_workflow(self):
        """
        checkpoint the workflow module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        self.save_job_dict()
        save_dict = {
            self.checkpoint_name: {
                'unknown_job': self.unknown_job_id,
                'counters': self.counters,
            }
        }
        if self.new_counters or self.new_unknown_id:
            restarter.write_checkpoint_file(self.checkpoint_file, save_dict)
            self.new_counters = False
            self.new_unknown_id = False

    def restart_workflow(self):
        """
        restart the workflow module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        # set the jobs dict from the pickle file
        self.read_job_dict()
        # see if any internal variables were checkpointed
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.unknown_job_id = restart_dict.get('unknown_job',
                                               self.unknown_job_id)
        self.counters = restart_dict.get('counters', self.counters)

    def block_until_completed(self, calc_ids: list[int]):
        """
        Function for enforcing synchronous execution

        Helper function to run when blocking behavior is desired. Will
        consistently check the queue until the specified job has completed.

        :param calc_ids: list of job IDs of the calculations to check for
            completion. Can also pass a single ID.
        :type calc_ids: int or list
        """
        if type(calc_ids) is list:
            remaining_jobs = deepcopy(calc_ids)
        elif type(calc_ids) is int:
            remaining_jobs = [calc_ids]
        else:
            raise TypeError('Job IDs must be a single int or a list!')

        wait_cycle_counter = 0
        while len(remaining_jobs) > 0:
            job_states = self.update_job_status(remaining_jobs)
            jobs_to_remove = []
            for job_id, job_state in zip(remaining_jobs, job_states):
                if job_state[:4] == 'done':
                    self.logger.info((f'Job {job_id} completed, removing '
                                      f'from waiting list'))
                    jobs_to_remove.append(job_id)
            # remove after iterating the whole list so no IDs are skipped
            for job_id in jobs_to_remove:
                remaining_jobs.remove(job_id)
            if len(remaining_jobs) == 0:
                break
            if wait_cycle_counter % 5 == 0:
                self.logger.info((f'Wait iteration {wait_cycle_counter}, ['
                                  f'cycle time = {self.synch_check_frequency}'
                                  f' s] with {len(remaining_jobs)} jobs left '
                                  f'to complete'))
            wait_cycle_counter += 1
            sleep(self.synch_check_frequency)
        self.logger.info(f'Jobs {calc_ids} have completed, continuing...')

    def generate_batch_file(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
        extra_args: Optional[dict[str, str]] = None,
    ) -> str:
        """
        Construct a batch file for job submission

        This is a helper funciton for submit_job to call to construct a batch
        file for the proper scheduler.

        :param command: command that defines the job to be executed
        :type command: str
        :param run_path: directory for the job to be executed in
        :type run_path: str
        :param job_details: optional parameters for running the job.
            Parameters specified in this dict are: 'nodes', 'tasks',
            'tasks_per_node', 'queue', 'account', 'walltime', and
            'custom_preamble' |default| ``None``
        :type job_details: dict
        :param extra_args: dictionary of extra args, can include lines to pre-
            or postpend the job command in the batch script (given by the
            'preamble' and 'postamble' keys), additional #SCHEDULER commands
            (given by the 'extra_header' key where the value is a formatted
            string including the #SBATCH/#BSUB keyword) as well as alternative
            batch template location, specified by the 'template' key |default|
            ``None``
        :type extra_args: dict
        :returns: name of the batch file
        :rtype: str
        """
        if job_details is None:
            job_details = {}
        if extra_args is None:
            extra_args = {}
        # default template will be set by the concrete child classes
        file_template = extra_args.get('template', self.default_template)
        batch_file = Templates(file_template, run_path)

        patterns = [
            'nodes',
            'queue',
            'account',
            'walltime',
            'extra_header',
            'preamble',
            'command',
            'postamble',
        ]
        # use the child defined generate_job_preamble
        scheduler_preamble = self.generate_job_preamble(job_details)
        custom_preamble = job_details.get('custom_preamble', None)
        if custom_preamble is not None:
            srun_launch = f'{scheduler_preamble} {command}'
        else:
            # run string defined for specific workflows
            srun_launch = f'{self.run_string} {scheduler_preamble} {command}'
        replacements = [
            job_details.get('nodes', self.default_nodes),
            job_details.get('queue', self.default_queue),
            job_details.get('account', self.default_account),
            self.format_walltime(
                job_details.get('walltime', self.default_walltime),
                self.USE_SEC,
            ),
            extra_args.get('extra_header', ''),
            extra_args.get('preamble', ''),
            srun_launch,
            extra_args.get('postamble', ''),
        ]
        file_name = batch_file.replace(patterns, replacements)
        self.logger.info(f'Batch file written to {run_path}/{file_name}')
        return file_name
