from abc import ABC
from os import path, PathLike
import subprocess as sp
from typing import Optional, Union
from .workflow_base import HPCWorkflow, JobStatus
from ..utils.exceptions import (
    ProblematicJobStateError,
    JobSubmissionError,
    UnfullfillableDependenciesError,
)
from ..utils.data_standard import METADATA_KEY


class SlurmWF(HPCWorkflow, ABC):
    """
    Workflow manager for full execution using a batch file and slurm scheduler

    SlurmWF is a fully featured workflow module for submitting jobs to the
    slurm scheduler using batch files. It can handle asynchronus job submission
    but still provides the option for blocking (synchronous) behavior.
    Responsibilities include directory creation, job creation, job status
    checking.
    """

    def __init__(self, default_template: Optional[str] = None, **kwargs):
        """
        set variables and initialize the recorder

        :param default_template: path to the template file to use for
            submission scripts. If none provided, uses the default template
            present in ./default_templates |default| ``None``
        :type default_template: str
        :param kwargs: remaining parameters passed to parent for init. Keys
            include: queue, account, walltime, nodes, tasks, tasks_per_node,
            wait_freq, root_directory, checkpoint_file, checkpoint_name, and
            job_record_file
        :type kwargs: dict
        """
        super().__init__(**kwargs)
        if default_template is None:
            source_file_location = path.dirname(path.abspath(__file__))
            self.default_template = (f'{source_file_location}/'
                                     'default_templates/slurm.sh')
        else:
            self.default_template = default_template
        # determines print format of walltime strings
        self.USE_SEC = True
        self.run_string = 'srun'

    def extract_slurm_id(self, str_output: str) -> int:
        """
        From the command line output, extract the slurm job id

        :param str_output: full output string to extract ID from
        :type str_output: str
        :returns: slurm ID
        :rtype: int
        """
        successful_extraction = False
        try:
            split_output = str_output.split()
            # check if expected output format is present

            if split_output[0] == 'Submitted' and split_output[2] == 'job':
                # output from sbatch
                calc_id = int(split_output[3])
                self.logger.info(f'Found slurm ID: {calc_id}')
                successful_extraction = True
            elif split_output[0] == 'srun:' and split_output[1] == 'job':
                # output from srun
                calc_id = int(split_output[2])
                self.logger.info(f'Found slurm ID: {calc_id}')
                successful_extraction = True
            else:
                self.logger.info((f'Output string is unexpected format:\n'
                                  f'\t{str_output.strip()}'))
                self.logger.info(f'Setting ID to: {self.unknown_job_id}')
                calc_id = self.unknown_job_id
                self.unknown_job_id += 1
                self.new_unknown_id = True
        except AttributeError:
            self.logger.info((f'Passed output not a splitable string, setting '
                              f'ID to {self.unknown_job_id}'))
            calc_id = self.unknown_job_id
            self.unknown_job_id += 1
            self.new_unknown_id = True
        except IndexError:
            self.logger.info((f'Passed output likely an empty string, setting '
                              f'ID to {self.unknown_job_id}'))
            calc_id = self.unknown_job_id
            self.unknown_job_id += 1
            self.new_unknown_id = True
        if not successful_extraction:
            raise JobSubmissionError(('could not obtain slurm ID from job '
                                      'submission - cannot continue at this '
                                      'time'))
        return calc_id

    def generate_job_preamble(
        self,
        job_details: dict[str, Union[float, str]],
    ) -> str:
        """
        Set slurm arguments from job_details or from defaults defined by the WF

        This is a helper function for constructing the preamble of the srun
        command. Values set are nodes, tasks (optional)

        :param job_details: dict passed through :meth:`~submit_job` including
            any desired alterations from the workflow defaults
        :type job_details: dict
        :returns: populated preable string
        :rtype: str
        """
        node_val = job_details.get('nodes', self.default_nodes)
        task_val = job_details.get('tasks', self.default_tasks)
        custom_preamble = job_details.get('custom_preamble', None)
        tasks_per_node_val = job_details.get('tasks_per_node',
                                             self.default_tasks_per_node)

        if custom_preamble is not None:
            job_arg_string = custom_preamble
        elif task_val > 1:
            if tasks_per_node_val > 1:
                self.logger.info((f'Warning: tasks and tasks-per-node are '
                                  f'both specified. Using tasks = {task_val}'))
            job_arg_string = (f'-N {node_val} -n {task_val}')
        else:
            # either tasks_per_node_val > 1, or both tasks-per-node and tasks
            # = 1. In both cases, desireable to use tasks-per-node so if nodes
            # > 1, can benefit from the distributed memory setup
            job_arg_string = (
                f'-N {node_val} --ntasks-per-node {tasks_per_node_val}')
        return job_arg_string

    def check_completed_job_status(self, slurm_id: int) -> str:
        """
        Use scontrol to extract the status of a completed job

        :param slurm_id: job ID of the job to query
        :type slurm_id: int
        :returns: job status
        :rtype: str
        """
        control_output = sp.run(f'scontrol show job {slurm_id}',
                                capture_output=True,
                                shell=True,
                                encoding='UTF-8')
        if control_output.returncode != 0:
            # can't find job
            if self.job_done_file_present(slurm_id):
                self.logger.info('scontrol query return code nonzero, but '
                                 'job_done file present')
                state = 'done'
            else:
                state = 'done_unknown'
        else:
            # will get full report, JobState at 10 after =
            job_state = control_output.stdout.split()[10].split('=')[1]
            if job_state == 'COMPLETED':
                state = 'done'
            elif job_state == 'TIMEOUT':
                state = 'done_timeout'
            elif job_state == 'CANCELLED':
                state = 'done_cancelled'
            else:
                state = 'done_other'
        return state

    def update_job_status(self, slurm_ids: list[int]) -> list[str]:
        """
        Query the scheduler and extract the job_status

        This helper function uses squeue to check the slurm queue and extracts
        updates about a job's progress, modifying the corresponding job_status
        object. Status options are: 'done', 'pending', 'running', 'dependency',
        and 'completing'. The current status is returned for convenience.

        :param slurm_ids: list of slurm IDs of the jobs to check for completion
        :type slurm_ids: list
        :returns: list of job states
        :rtype: list (of str)
        """
        status_changed = False
        job_str = f'{slurm_ids[0]}'
        if len(slurm_ids) > 1:
            for remaining_ids in slurm_ids[1:]:
                job_str += f',{remaining_ids}'
        queue_output = sp.run(f'squeue -j {job_str} -o "%i %t %R"',
                              capture_output=True,
                              shell=True,
                              encoding='UTF-8')
        updated_states = []
        if queue_output.returncode != 0 or queue_output.stderr:
            self.logger.info((f'Problem checking for jobs {slurm_ids}, '
                              f'exit code: {queue_output.returncode} '
                              f'stderr: {queue_output.stderr.strip()}'))
            # possible that the job is just not in database anymore
            if queue_output.stderr[23:37] == 'Invalid job id':
                for slurm_id in slurm_ids:
                    known_status = self.get_job_status(slurm_id)
                    if self.job_done_file_present(slurm_id):
                        new_state = 'done'
                        self.logger.info((f'Updating job {slurm_id} state '
                                          f'from {known_status.state} to '
                                          f'{new_state} based on job_done '
                                          'file'))
                        known_status.state = new_state
                    else:
                        new_state = 'error'
                        raise ProblematicJobStateError(
                            ('Orchestrator does not currently have set '
                             f'behavior for "{new_state}" job state. Check '
                             'your queue for any remaining pending jobs.'))
                    updated_states.append(known_status.state)
        else:
            split_output = queue_output.stdout.split()
            for slurm_id in slurm_ids:
                known_status = self.get_job_status(slurm_id)
                try:
                    slurm_str_index = split_output.index(str(slurm_id))
                    slurm_state = split_output[slurm_str_index + 1]
                    if slurm_state == 'CG':
                        new_state = 'completing'
                    elif slurm_state == 'R':
                        new_state = 'running'
                    elif slurm_state == 'PD':
                        reason = split_output[slurm_str_index + 2][1:-1]
                        if reason == 'Dependency':
                            new_state = 'dependency'
                        else:
                            new_state = 'pending'
                    else:
                        new_state = 'unknown'
                except ValueError:
                    # slurm id not in list, so query job status with scontrol
                    if known_status.state[:4] != 'done':
                        # completed job state has not been assigned
                        new_state = self.check_completed_job_status(slurm_id)
                    else:
                        # scontrol already called, no changes
                        new_state = known_status.state
                except Exception:
                    new_state = 'error'
                    self.logger.info((f'Job {slurm_id} state parsing had an '
                                      f' unknown error, set state to "error"'))
                if new_state != known_status.state:
                    self.logger.info((f'Updating job {slurm_id} state '
                                      f'from {known_status.state} to '
                                      f'{new_state}'))
                    known_status.state = new_state
                    status_changed = True

                if new_state in self.problematic_states:
                    raise ProblematicJobStateError(
                        (f'Orchestrator does not currently have set behavior '
                         f'for "{new_state}" job state. Check your queue for '
                         f'any remaining pending jobs.'))

                updated_states.append(known_status.state)

            if status_changed:
                # we only update the job dict if any statuses have changed
                self.checkpoint_workflow()

        return updated_states

    def submit_job(
        self,
        command: str,
        run_path: Union[str, PathLike],
        job_details: Optional[dict[str, Union[float, str]]] = None,
    ) -> int:
        """
        Submits a job for running using a submission script and sbatch.

        submit_job handles job submission for the modules and is the main
        interface for the workflows to be used. For the :class:`SlurmWF`
        implementation, fully articulated batch scripts are generated each job
        and submitted to the slurm scheduler via sbatch. Method inputs define
        the ``command`` to be executed for the job, location for the run, and
        details about the job's resources (``job_details``). ``job_details``
        inlcudes ``dependencies`` of the job in the form of a list of job_ids,
        if the job is blocking (``synchronus``) or not, and an extra dictionary
        , ``extra_args``, to add flexibility for parameterizing the job (such
        as pre- or postambles to include in the batch file). ``dependencies``
        are a list of job IDs which must have a successfully completed
        :class:`~JobStatus` for the present job to run. However, jobs can still
        be submitted to the queue with outstanding dependencies if run
        asynchronously. The ``after`` key in ``extra_args`` can be specified to
        define the slurm dependency behavior (i.e. 'afterany' or 'afterok').
        Note that while default job resources (nodes, account, walltime, etc.)
        are present, they can be overridden by providing these keywords in the
        ``job_details`` dict for any specific calculation. Creates the
        :class:`~.workflow_base.JobStatus` for this job, where the job state is
        initially 'submitted' and can be updated to 'pending', 'dependency',
        'running', 'completing', 'done', or 'unknown'. The 'done' state means
        the calculation has completed, but can be decorated with suffixes that
        add more information if the job didn't successfully complete (i.e.
        'done_timeout'). Status checks are preformed by
        :meth:`~update_job_status`. Returns the slurm ID, which can be used to
        retrieve the present job's :class:`~.workflow_base.JobStatus`. If the
        slurm ID cannot be identified, an internal tracking number
        (starting at 1000) is used instead.

        :param command: command that defines the job to be executed
        :type command: str
        :param run_path: directory for the job to be executed in
        :type run_path: str
        :param job_details: specifics for running the job, such as
            number of nodes, queue, etc., as well as optional dependency list,
            if the job should be synchronous or asychronous, and any other
            optional arguments, such as pre- or postambles |default| ``None``
        :type job_details: dict
        :returns: return job ID to query this job status and location
        :rtype: int
        """
        # check inputs and provide information to user

        if job_details is None:
            job_details = {}
            self.logger.info((f'No job details specified, will use defaults:\n'
                              f'  N = {self.default_nodes}, A = '
                              f'{self.default_account}, t = '
                              f'{self.default_walltime}, p = '
                              f'{self.default_queue}'))

        synchronous = job_details.get('synchronous', False)
        dependencies = job_details.get('dependencies', [])
        extra_args = job_details.get('extra_args', {})

        job_can_run = True
        calc_id = -1
        exit_code = 'undefined error'

        if command is None or command == '':
            job_can_run = False
            exit_code = 'empty command'
            self.logger.info('Job will not run: no command')

        if job_can_run:
            # generate the batch script
            batch_file = self.generate_batch_file(
                command,
                run_path,
                job_details,
                extra_args,
            )
            # submit the batch script, with dependencies
            if dependencies:
                self.logger.info(
                    f'Including dependencies: {str(dependencies)[1:-1]}')
                after_type = extra_args.get('after', 'afterany')
                # format the list to remove [] and spaces between commas
                no_space_list = ''.join(str(dependencies)[1:-1].split())
                # swap commas for : to separate job ids
                no_space_list = no_space_list.replace(',', ':')
                depend_str = f'-d {after_type}:{no_space_list}'
            else:
                depend_str = ''

            self.logger.info('Spawning job, ID to be defined')
            submit_command = (
                f'cd {run_path}; sbatch {depend_str} {batch_file}')
            process_output = sp.run(submit_command,
                                    capture_output=True,
                                    shell=True,
                                    encoding='UTF-8')
            exit_code = process_output.returncode
            # get the slurm ID
            if exit_code != 0:
                self.logger.info((f'Something wrong with submission [exit '
                                  f'code = {exit_code}]'))
                calc_id = self.extract_slurm_id(process_output.stderr)
            else:
                calc_id = self.extract_slurm_id(process_output.stdout)
            # create job_status and add to self.jobs
            job_status = JobStatus(run_path, 'submitted', exit_code)
            # if synch, wait and check for completion
            if synchronous:
                self.block_until_completed(calc_id)
        else:
            calc_id = self.unknown_job_id
            self.unknown_job_id += 1
            self.new_unknown_id = True
            job_status = JobStatus(run_path, 'done_cancelled', exit_code)
            if dependencies:
                raise UnfullfillableDependenciesError()

        job_status.metadata = extra_args.get(METADATA_KEY, {})
        self.jobs[calc_id] = job_status
        self.checkpoint_workflow()
        return calc_id
