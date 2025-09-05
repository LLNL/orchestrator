from ..storage import Storage
from ..computer.score.score_base import ScoreBase
from ..computer.score.quests import QUESTSEfficiencyScore
from ..utils.data_standard import METADATA_KEY, SELECTION_MASK_KEY
from ..utils.exceptions import CellTooSmallError
from ..utils.isinstance import isinstance_no_import
from ..utils.recorder import Recorder
from ..utils.restart import restarter
from ..workflow import Workflow
from .extract_env import extract_env, find_central_atom
from ase import Atoms
from copy import deepcopy
from functools import partial
import itertools
from multiprocessing import Pool, cpu_count
import numpy as np
from os import getpid
from typing import Union, Optional, Any

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:


class Augmentor(Recorder):
    """
    Augmentor class containing methods for dataset augmentation operations

    :param default_iteration_limit: iteration limit for iterative FPS
        algorithms used within the Augmentor, can be overridden by method
        arguments at runtime. Optional, |default| 50
    :type default_iteration_limit: int
    :param checkpoint_file: name of the checkpoint file to write restart
            information to |default| './orchestrator_checkpoint.json'
    :type checkpoint_file: str
    :param checkpoint_name: name of the restart block for this module in
        the checkpoint file |default| 'augmentor'
    :type checkpoint_name: str
    """

    def __init__(
        self,
        default_iteration_limit: Optional[int] = 50,
        checkpoint_file: Optional[str] = './orchestrator_checkpoint.json',
        checkpoint_name: Optional[str] = 'augmentor',
        **kwargs: dict,
    ):
        """
        set variables and initialize the recorder and default workflow

        :param default_iteration_limit: iteration limit for iterative FPS
            algorithms used within the Augmentor, can be overridden by method
            arguments at runtime. Optional, |default| 50
        :type default_iteration_limit: int
        :param checkpoint_file: name of the checkpoint file to write restart
                information to |default| './orchestrator_checkpoint.json'
        :type checkpoint_file: str
        :param checkpoint_name: name of the restart block for this module in
            the checkpoint file |default| 'augmentor'
        :type checkpoint_name: str
        """
        super().__init__()
        self.default_iteration_limit = default_iteration_limit
        self.checkpoint_file = checkpoint_file
        self.checkpoint_name = checkpoint_name
        self.current_method = 'init'
        self.outstanding_jobs = []
        self.restart_augmentor()

    def checkpoint_augmentor(self):
        """
        checkpoint the augmentor module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        save_dict = {
            self.checkpoint_name: {
                'current_method': self.current_method,
                'outstanding_jobs': self.outstanding_jobs,
            }
        }
        restarter.write_checkpoint_file(self.checkpoint_file, save_dict)
        # once a checkpoint is reached we should not operate as if restarting
        self.restart = False

    def restart_augmentor(self):
        """
        restart the augmentor module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        restart_dict = restarter.read_checkpoint_file(
            self.checkpoint_file,
            self.checkpoint_name,
        )
        self.current_method = restart_dict.get(
            'current_method',
            self.current_method,
        )
        self.outstanding_jobs = restart_dict.get(
            'outstanding_jobs',
            self.outstanding_jobs,
        )

    ###########################################################################
    # Environment extraction and analysis routines                            #
    ###########################################################################

    def identify_novel_environments(
        self,
        configs_to_evaluate: Union[Atoms, list[Atoms]],
        reference_dataset: Union[Atoms, list[Atoms]],
        score_module: ScoreBase,
        score_compute_args: dict[str, Any],
        workflow: Workflow,
        job_details: Optional[dict[str, Union[str, int]]] = None,
        batch_size: Optional[int] = 1,
        test_criteria: Optional[float] = 0.0,
    ) -> tuple[list[Atoms], list[np.ndarray], int]:
        """
        Use a novelty score metric to find environments outside a dataset

        Evaluate the atomic environments of an input list of configurations
        with respect to an existing dataset, denoting the environments which
        are novel to the dataset based on the evaluated score metric.

        :param configs_to_evaluate: ASE Atoms list (or single Atoms object) of
            configurations to analyze compared to a reference dataset.
            Descriptors must be precomputed on the Atoms.
        :type configs_to_evaluate: list[Atoms]
        :param reference_dataset: ASE Atoms which encapsulate the reference set
            to compare to. Descriptors must be precomputed on the Atoms.
        :type reference_dataset: list[Atoms]
        :param score_module: instantiated Score module which provides compute
            functions for obtaining a score value for a given dataset.
            Currently this method only supports QUESTSDeltaEntropyScore.
        :type score_module: ScoreBase
        :param score_compute_args: arguments that define the computation of
            the score module. Must include the 'descriptors_key' key for the
            precomputed descriptors.
        :type score_compute_args: str
        :param workflow: Workflow module to use for computing scores
        :type workflow: Workflow
        :param job_details: dict that includes any additional parameters for
            running the job (passed to :meth:`~.Workflow.submit_job`)
            |default| ``None``
        :type job_details: dict
        :param batch_size: number of configurations to pass to
            :meth:`~.ScoreBase.compute_batch` at once. |default| 1
        :type batch_size: int
        :param test_criteria: value of the score which indicates an atomic
            environment should be considered "novel" |default| 0
        :type test_criteria: float
        :returns: tuple of list of configs with attached scores, list of
            length num configs of boolean masks matching the size of the
            different configs to evaluate, where True indicates that the
            environment should be considered as novel compared to the
            reference dataset and the max index in the combined array to seed
            a FPS if not all points will be used
        :rtype: tuple(list[np.ndarray], int)
        """
        # we currently only support the usage of QUESTSDeltaEntropyScore for
        # this method, but this is an augmentor method to enable flexible
        # future implementation for other score types
        if isinstance_no_import(score_module, 'QUESTSDeltaEntropyScore'):
            # input score_compute_args should contain descriptors_key, and
            # bandwidth, num_nearest_neighbors, and graph_neighbors, if any
            # should be overridden.
            if isinstance(reference_dataset, Atoms):
                reference_dataset = [reference_dataset]
            ref_descriptors = []
            for ref_data in reference_dataset:
                descriptors = ref_data.get_array(
                    score_compute_args['descriptors_key'])
                if SELECTION_MASK_KEY in ref_data.arrays.keys():
                    selection_mask = ref_data.get_array(SELECTION_MASK_KEY)
                    ref_descriptors.append(descriptors[selection_mask])
                else:
                    ref_descriptors.append(descriptors)
            ref_descriptors = np.concatenate(ref_descriptors)
            # augment the compute args with these keys, which will always be
            # the values listed below
            full_score_compute_args = deepcopy(score_compute_args)
            full_score_compute_args |= {
                'score_quantity': 'DELTA_ENTROPY',
                'reference_set': ref_descriptors,
            }
        else:
            raise TypeError(
                'Only QUESTSDeltaEntropyScore supported at this time.')

        if self.current_method == 'id_novel-waiting':
            calc_ids = self.outstanding_jobs
        else:
            if isinstance(configs_to_evaluate, Atoms):
                configs_to_evaluate = [configs_to_evaluate]
            calc_ids = score_module.run(
                'identify_novel_environments',
                configs_to_evaluate,
                full_score_compute_args,
                workflow,
                job_details,
                batch_size,
            )  # verbose not used here

            # checkpoint for restart
            self.current_method = 'id_novel-waiting'
            self.outstanding_jobs = calc_ids
            self.checkpoint_augmentor()

        # parse the output
        # wait for job to complete is done in data_from_calc_ids
        scored_configs = score_module.data_from_calc_ids(calc_ids, workflow)
        novelty_scores = [
            config.get_array(f'{score_module.OUTPUT_KEY}_score')
            for config in scored_configs
        ]
        combined_scores = np.concatenate(novelty_scores)
        self.logger.info(f'{np.sum(combined_scores > test_criteria)} '
                         f'environments out of {len(combined_scores)} are '
                         'found to be novel')
        max_novelty_index = np.argmax(combined_scores)
        self.logger.info('Maximum novelty score is: '
                         f'{combined_scores[max_novelty_index]}')
        selection_masks = [
            score_array > test_criteria for score_array in novelty_scores
        ]

        # update the checkpoint file so subsequent calls to this method don't
        # skip submission
        self.outstanding_jobs = []
        self.current_method = 'id_novel-done'
        self.checkpoint_augmentor()

        return scored_configs, selection_masks, max_novelty_index

    def extract_and_tag_subcells(
        self,
        configs: list[Atoms],
        selection_masks: Union[list[np.ndarray], str],
        extract_rc: float,
        extract_box_size: float,
        keys_to_transfer: Optional[list[str]] = None,
    ) -> list[Atoms]:
        """
        use the extract_env function to isolate and relax subcells from a cell

        this method will generally be used as part of an active learning loop
        in conjunction with :meth:`~.identify_novel_environments`. If the
        extracted cell that is desired is larger than the inital
        configuration, the inital configuration is returned instead.

        :param configs: ASE Atoms list (or single Atoms object) containing the
            configurations from which subcells will be extracted. Should be the
            same length as the selection_masks list
        :type configs: list[Atoms]
        :param selection_masks: list of masks of the same shape as configs
            which note the atomic environments to extract or the string
            'attached' in which case masks will be taken from the arrays with
            the key SELECTION_MASK_KEY from the configs
        :type selection_masks: np.ndarray or str
        :param extract_rc: cutoff radius to extract and constrain positions in
            Angstroms
        :type extract_rc: float
        :param extract_box_size: side length of the box to embed the extracted
            environment into
        :type extract_box_size: float
        :param keys_to_transfer: any keys of data attached to the configs which
            should be preserved through the extraction process.
        :type keys_to_transfer: list of str
        :returns: a list of length sum(selection_masks) which contain ASE Atoms
            with a ``central_atom_index`` key in their metadata dictionary
        :rtype: list[Atoms]
        """
        if selection_masks == 'attached':
            selection_masks = [
                c.get_array(SELECTION_MASK_KEY) for c in configs
            ]
        num_envs = np.sum(np.concatenate(selection_masks))
        num_configs = len(selection_masks)
        atom_ids = [np.argwhere(mask).ravel() for mask in selection_masks]
        # define the box size we will cut out
        extract_box = np.array([
            extract_box_size,
            extract_box_size,
            extract_box_size,
        ])
        subcell_list = []
        # create the subcells with fixed central spheres embedded in cubes cut
        # from the larger configuration
        self.logger.info(f'Extracting a total of {num_envs} environments from '
                         f'{num_configs} configurations')
        progress = 0
        for config, atom_id_list in zip(configs, atom_ids):
            if len(atom_id_list) == 0:
                # no atoms to extract from this configuration
                continue
            # pull out the fixed sphere in an enclosing cube for each index
            try:
                subcells = extract_env(
                    config,
                    extract_rc,
                    atom_id_list,
                    extract_box,
                    extract_cube=True,
                    keys_to_transfer=keys_to_transfer,
                )
                for subcell in subcells:
                    central_atom_index = find_central_atom(
                        subcell,
                        extract_box_size,
                    )
                    extraction_dict = {
                        'central_atom_index': central_atom_index,
                        'extracted_cell': True,
                    }
                    # create metadata dict if doesn't exist already
                    subcell.info.setdefault(METADATA_KEY, {})
                    # add the extraction info to metadata dict
                    subcell.info[METADATA_KEY].update(extraction_dict)
                    # add the mask info for the central atom
                    mask = np.zeros(len(subcell), dtype=bool)
                    mask[central_atom_index] = 1
                    subcell.set_array(SELECTION_MASK_KEY, mask)
                    # we don't want to attach the relaxation forces/energy
                    subcell.calc = None
                    subcell_list.append(subcell)
            # if the cell is too small to extract from, just return the
            # full cell and set the central atom to be the one that was
            # selected
            except CellTooSmallError as e:
                self.logger.info(f'Could not extract subcell due to: {e}')
                # NOTE: there may be an issue with this if the config has
                # other data attached which we don't want to save?
                subcell = deepcopy(config)
                extraction_dict = {
                    'central_atom_index': atom_id_list,
                    'extracted_cell': False,
                }
                # create metadata dict if doesn't exist already
                subcell.info.setdefault(METADATA_KEY, {})
                # add the extraction info to metadata dict
                subcell.info[METADATA_KEY].update(extraction_dict)
                # add the mask info for the selected atoms
                mask = np.zeros(len(subcell), dtype=bool)
                mask[atom_id_list] = 1
                subcell.set_array(SELECTION_MASK_KEY, mask)
                subcell_list.append(subcell)
            progress += len(atom_id_list)
            self.logger.info(f'{progress}/{num_envs} environments extracted')
        return subcell_list

    def score_and_extract_subcells(
        self,
        configs: Union[Atoms, list[Atoms]],
        reference_dataset: Union[Atoms, list[Atoms]],
        score_module: ScoreBase,
        score_compute_args: dict[str, Any],
        workflow: Workflow,
        extract_rc: float,
        extract_box_size: float,
        job_details: Optional[dict[str, Union[str, int]]] = None,
        batch_size: Optional[int] = 1,
        max_num_to_extract: Optional[int] = None,
        extraction_pruning_score: Optional[ScoreBase] = None,
        extraction_pruning_args: Optional[dict] = None,
    ) -> list[Atoms]:
        """
        Integrated method for identifying and extracting novel atomic envs

        This method uses :meth:`~.identify_novel_environments` and
        :meth:`~.extract_and_tag_subcells` to obtain a list of subcells with
        central atoms which are considered novel with respect to the provided
        reference set by the provided score module.

        :param configs: ASE Atoms list (or single Atoms object) of
            configurations to analyze compared to a reference dataset.
            Descriptors must be precomputed on the Atoms.
        :type configs: list[Atoms]
        :param reference_dataset: ASE Atoms which encapsulate the reference set
            to compare to. Descriptors must be precomputed on the Atoms.
        :type reference_dataset: list[Atoms]
        :param score_module: instantiated Score module which provides compute
            functions for obtaining a score value for a given dataset.
            Currently this method only supports QUESTSDeltaEntropyScore.
        :type score_module: ScoreBase
        :param score_compute_args: arguments that define the computation of
            the score module. Must include the 'descriptors_key' key for the
            precomputed descriptors.
        :type score_compute_args: str
        :param workflow: Workflow module to use for computing scores
        :type workflow: Workflow
        :param extract_rc: cutoff radius to extract and constrain positions in
            Angstroms
        :type extract_rc: float
        :param extract_box_size: side length of the box to embed the extracted
            environment into
        :type extract_box_size: float
        :param job_details: dict that includes any additional parameters for
            running the initial novel environment job |default| ``None``
        :type job_details: dict
        :param batch_size: number of configurations to batch for the novel
            environments job |default| 1
        :type batch_size: int
        :param max_num_to_extract: limit of number of subcells to extract. If
            this parameter is provided, FPS will be performed directly on the
            subset of novel environments to return the desired number of envs.
            The first point selected for sampling will be the most novel
            determined by the score module.
            |default| ``None``
        :type max_num_to_extract: int
        :param extraction_pruning_score: Score module to use for
            down-selecting which of the novel environments to extract. If
            max_num_to_extract is provided, this argument will be ignored.
            Chunked iterative pruning will be applied using this score module
            to remove redundancy from the novel environments prior to subcell
            extraction. Compute arguments can be provided via the
            `extraction_pruning_args` argument. |default| ``None``
        :type extraction_pruning_score: Score
        :param extraction_pruning_args: Arguments to specify the score
            calculation employed in iterative pruning if
            `extraction_pruning_score` is provided. If
            `extraction_pruning_score` is present, this argument remains
            optional, with the score using default values for the compute args
            in this circumstance. |default| ``None``.
        :type extraction_pruning_args: dict
        :returns: a list of length min(extraction, number of novel
            environments) which contain ASE Atoms of subcells extracted from
            the provided configurations with a ``central_atom_index`` key in
            their metadata dictionary to mark which atom in the subcell is
            calculated to be novel
        :rtype: list[Atoms]
        """
        # first find all the atomic environments which are novel (including
        # the MOST novel). Returned configs will have the scores attached
        (
            scored_configs,
            selection_masks,
            most_novel_index,
        ) = self.identify_novel_environments(
            deepcopy(configs),
            reference_dataset,
            score_module,
            score_compute_args,
            workflow,
            job_details,
            batch_size,
        )

        descriptors_key = score_compute_args['descriptors_key']
        combined_masks = np.concatenate(selection_masks)
        num_novel = np.sum(combined_masks)
        if num_novel == 0:
            self.logger.info('No novel environments found!')
            return []

        # if only a certain number of environments are desired, start with the
        # most novel and then fill out with well spaced additional options
        if max_num_to_extract is not None:
            if extraction_pruning_score is not None:
                self.logger.info('WARNING: max_num_to_extract and '
                                 'extraction_pruning_input are both set, only '
                                 'using max_num_to_extract')
            # see if there are more novel environments than requested,
            # otherwise selection_masks can be used as-is
            if num_novel > max_num_to_extract:
                # sampling only from novel environments
                combined_descriptors = np.concatenate([
                    c.get_array(descriptors_key)[m]
                    for c, m in zip(scored_configs, selection_masks)
                ])
                # convert most_novel_index to the selected subset
                sampling_start = np.sum(combined_masks[:most_novel_index])
                # select the requested number of environments from the full set
                keep_indices = self.__class__._fps(
                    combined_descriptors,
                    max_num_to_extract,
                    first_index=sampling_start,
                )
                dataset_shapes = [len(config) for config in scored_configs]
                combined_keep_mask = np.zeros(np.sum(dataset_shapes),
                                              dtype=bool)
                # propagate the keep_indices to the indices of the full array
                # of the configuration shapes considering the existing mask
                full_data_indices = np.flatnonzero(
                    combined_masks)[keep_indices]
                combined_keep_mask[full_data_indices] = True
                # split mask to match shapes of the dataset elements,
                # overwriting the selection_mask
                selection_masks = np.array_split(
                    combined_keep_mask,
                    np.cumsum(dataset_shapes)[:-1],
                )
        elif extraction_pruning_score is not None:
            for config, mask in zip(scored_configs, selection_masks):
                config.set_array(SELECTION_MASK_KEY, mask)
            # override this variable for the later call to extact_and_tag...
            selection_masks = 'attached'
            scored_configs = self.chunked_iterative_fps_prune(
                scored_configs,
                descriptors_key,
                extraction_pruning_score,
                # set a large number to use cpu_count() - max parallelism
                num_chunks=500,
                prune_ratio_args=extraction_pruning_args,
                pruning_convergence=0.01,
                first_index=most_novel_index,
                hierarchical_parallelism=True,
            )

        # extract the atomic environments notated by selection_masks
        extracted_cells = self.extract_and_tag_subcells(
            scored_configs,
            selection_masks,
            extract_rc,
            extract_box_size,
            # descriptors and scores are the only values we care to preserve
            [descriptors_key, f'{score_module.OUTPUT_KEY}_score'],
        )
        return extracted_cells

    ###########################################################################
    # Pruning methods and submethods                                          #
    ###########################################################################

    def simple_prune_dataset(
        self,
        dataset: Union[list[Atoms], str],
        prune_method: str,
        prune_value: float,
        prune_large_value: bool,
        score_args: dict,
        score_module: ScoreBase,
        workflow: Workflow,
        storage: Optional[Storage] = None,
    ) -> list[Atoms]:
        """
        prune a dataset based on analysis from one or more UQ methods

        Take an existing dataset and reduce its size. This can be done based
        on a percentage or a value cutoff (``prune_method``) of the UQ metric
        output. The ``dataset`` can be an explicit list of ASE atoms or a
        storage handle. If the latter, then the storage module must also be
        provided. For more intelligent pruning, consider using the
        :meth:`~.chunked_iterative_fps_prune` method instead.

        :param dataset: ASE list or dataset handle
        :type dataset: list[Atoms] or str
        :param prune_method: option of either 'percentage' or 'cutoff'
        :type prune_method: str
        :param prune_value: metric to apply to prune_method. If
            ``prune_method`` = 'percentage', value can be between 0 and 1.0,
            representing the percentage that will be pruned from the dataset.
            If ``prune_method`` = 'cutoff', value represents the absolute
            quantity above or below (see ``prune_large_value``) which data is
            pruned.
        :type prune_value: float
        :param prune_large_value: a sorting variable to indicate if larger
            values (True) or smaller values (False) should be pruned.
        :type prune_large_value: bool
        :param score_args: arguments that define the computation of the score
            module. These may be set by an input file or from the caller.
        :type score_args: dict
        :param score_module: instantiated Score module which provides compute
            functions for obtaining a score value for a given dataset.
        :type score_module: ScoreBase
        :param workflow: Workflow module to use for computing scores
        :type workflow: Workflow
        :param storage: Storage module where the dataset is stored if the
            ``dataset`` argument is a dataset_handle. Otherwise this argument
            is not necessary.
        :type storage: Storage
        :returns: list of ASE atoms with a pruning mask applied as 0 or 1
            weights
        :rtype: list[Atoms]
        """
        # validate input
        if isinstance(dataset, str):
            if not isinstance_no_import(storage, 'Storage'):
                raise TypeError('When providing a dataset handle, a Storage '
                                'module must be provided as the storage '
                                'parameter')
            else:
                dataset = storage.get_data(dataset)
        if prune_method == 'percentage':
            if prune_value <= 0 or prune_value >= 1:
                raise ValueError(
                    'For percentage method, value must be between 0 and 1')
        elif prune_method != 'cutoff':
            raise ValueError(
                'Prune method must be either "percentage" or "cutoff"!')
        score_type = score_args.get('score_type')
        # TODO: update for supported types
        if score_type not in ['importance', 'uncertainty']:
            raise ValueError(
                'Score type must be one of "importance" or "uncertainty"')
        if score_type not in score_module.supported_score_types:
            raise ValueError(f'{score_module.__class__.__name__} does not '
                             f'support {score_type}')

        if score_module.OUTPUT_KEY + '_score' in dataset[0].arrays.keys():
            # use existing attached scores
            self.logger.info(
                'Skipping score generation, using attached scores instead.')
            scored_configs = dataset
        else:
            # obtain scores from score module
            self.logger.info(
                f'Obtaining scores using {score_module.__class__.__name__}')
            calc_ids = score_module.run(
                'augmentor_score_for_pruning',
                score_args,
                dataset,
                workflow,
                job_details=None,
                batch_size=1,
                verbose=False,
            )
            workflow.block_until_completed(calc_ids)
            self.logger.info('Scores done computing')
            scored_configs = score_module.data_from_calc_ids(calc_ids)

        # combined the atomic environment scores across configs
        environment_scores = [
            config.get_array(score_module.OUTPUT_KEY + '_score')
            for config in scored_configs
        ]
        unified_score_array = np.concatenate(environment_scores)

        config_lens = [len(config) for config in dataset]
        num_atomic_envs = np.cumsum(config_lens)[-1]
        config_index_splits = np.cumsum(config_lens)[:-1]

        if prune_method == 'cutoff':
            # find envs which satisfy the cutoff
            if prune_large_value:
                keep_mask = unified_score_array < prune_value
                compare_str = 'above'
            else:
                keep_mask = unified_score_array > prune_value
                compare_str = 'below'
            self.logger.info(f'Prune with the cutoff method ({prune_value}), '
                             f'pruning {np.sum(~keep_mask)}/{num_atomic_envs} '
                             f'atomic environments which are {compare_str} the'
                             ' cutoff.')
        else:
            # use keep a percentage
            num_envs_to_prune = int(prune_value * num_atomic_envs)
            # these indices correspond to the index of atomic environments
            # and are sorted in ascending order
            sorted_score_indices = np.argsort(unified_score_array)
            if prune_large_value:
                # remove num_atomic_envs_to_prune with the largest scores
                indices_to_keep = sorted_score_indices[:-num_envs_to_prune]
                compare_str = 'largest'
            else:
                # remove num_atomic_envs_to_prune with the smallest scores
                indices_to_keep = sorted_score_indices[num_envs_to_prune:]
                compare_str = 'smallest'
            keep_mask = np.zeros(num_atomic_envs, dtype=bool)
            keep_mask[indices_to_keep] = True
            self.logger.info(f'Prune a percentage ({prune_value}), pruning '
                             f'{num_envs_to_prune}/{num_atomic_envs} atomic '
                             f'environments which have the {compare_str} '
                             'scores.')

        # attach the weights to the scored configs
        for config, prune_weights in zip(
                scored_configs, np.array_split(keep_mask,
                                               config_index_splits)):
            config.set_array(SELECTION_MASK_KEY, prune_weights)
        return scored_configs

    def estimate_pruning_ratio(
        self,
        dataset: list[Atoms],
        score_module: 'QUESTSEfficiencyScore',
        score_compute_args: dict[str, Any],
    ) -> float:
        """
        Use an appropriate score metric to estimate the ideal degree of pruning

        Apply a score metric that estimates the redundancy of the provided
        dataset. From this quantity, provide the estimated pruning ratio as a
        value between 0.0 and 1.0

        :param dataset: dataset to evaluate for pruning
        :type dataset: list[Atoms]
        :param score_module: score module to obtain the (redundancy) metric
            from
        :type score_module: QUESTSEfficiencyScore
        :param score_compute_args: compute arguments to drive the score
            calculation
        :type score_compute_args: dict
        :returns: the pruning ratio as a value between 0 and 1, representing
            the % to be pruned from a dataset
        :rtype: float
        """
        # we currently only support the usage of QUESTSEfficiencyScore for this
        # method, but this is an augmentor method to enable flexible future
        # implementation for other score types
        if not isinstance_no_import(score_module, 'QUESTSEfficiencyScore'):
            raise TypeError(
                'Only QUESTSEfficiencyScore supported at this time.')
        # if computing in memory
        efficiency_metric = score_module.compute(dataset, **score_compute_args)
        # if running online - generally fast enough to just do in memory
        # calc_id = score_module.run('estimate_pruning_ratio', dataset,
        #                            score_compute_args, workflow)
        # workflow.block_until_completed(calc_id)
        # efficiency_metric = score_module.parse_for_storage(
        #     workflow.get_job_path(calc_id))

        # currently compute returns an array
        efficiency_metric = efficiency_metric[0]
        prune_ratio = 1.0 - efficiency_metric
        if prune_ratio < 0:
            self.logger.info(f'WARNING: prune_ratio = {prune_ratio} '
                             f'(efficiency={efficiency_metric}), setting to '
                             '0.0 instead')
            prune_ratio = 0
        return prune_ratio

    @classmethod
    def _fps(
        cls,
        points: np.ndarray,
        num_samples: int,
        index_shift: Optional[int] = 0,
        first_index: Optional[int] = 0,
    ) -> np.ndarray:
        """
        simple numpy-based FPS algorithm for internal augmentor use

        this function could be a placeholder for a more performant FPS
        implementation, but is used for now for testing with no additional
        dependencies.

        :param points: the data array of descriptors, with size NxD where N is
            the number of atomic environments and D is the descriptor length
        :type points: np.ndarray
        :param num_samples: number of points to select using the FPS algorithm
        :type num_samples: int
        :param index_shift: value to shift the indices to, applied after they
            are selected. This is useful if operating on a subsection of a
            larger set which will be recombined. |default| 0
        :type index_shift: int
        :param first_index: index of the first point from which farthest
            distances are calculated to begin the sampling. |default| 0
        :type first_index: int
        :returns: an array of length ``num_samples`` which contains the indices
            (ranging from 0 to len(points)) of which points constitute a
            farthest point set started from the ``first_index``
        :rtype: np.ndarray
        """
        if num_samples > points.shape[0]:
            raise RuntimeError(f"{num_samples=}, but {points.shape[0]=}. "
                               "Request fewer points.")

        farthest_indices = np.zeros(num_samples, dtype=int)
        distances = np.full(points.shape[0], np.inf)
        selected_mask = np.zeros(points.shape[0], dtype=bool)

        # First point
        farthest_indices[0] = first_index
        selected_mask[first_index] = True

        for i in range(1, num_samples):
            # Update distances from latest selected point
            distances = np.minimum(
                distances,
                np.linalg.norm(points - points[farthest_indices[i - 1]],
                               axis=1),
            )

            unselected_indices = np.flatnonzero(~selected_mask)
            unselected_distances = distances[unselected_indices]

            # early stopping criteria if all remaining points are the same
            if np.all(unselected_distances < 1e-12):
                farthest_indices[i:] = unselected_indices[:num_samples - i]
                break
            else:
                next_idx = unselected_indices[np.argmax(unselected_distances)]
                farthest_indices[i] = next_idx
                selected_mask[next_idx] = True

        return farthest_indices + index_shift

    def _multiprocess_prune(
        self,
        points: np.ndarray,
        num_chunks: int,
        num_samples: int,
        first_index: Optional[int] = 0,
    ) -> np.ndarray:
        """
        Internal function for applying FPS sampling in parallel over split data

        This method uses the multiprocessing module to perform FPS pruning in
        parallel (degree of parallelism is set by ``num_chunks``). It is only
        relevant for pruning ratios > 0.5, as each chunk must contain at least
        ``num_samples``. After the parallel sampling is completed, a final
        sampling across the combined results is performed. Note that results of
        this split sampling will not necessarily be equivalent to a single-shot
        approach.

        :param points: single array of points to sample over, with size NxD
            where N is the number of atomic environments and D is the dimension
            of the descriptor to analyze for distance metric
        :type points: np.ndarray
        :param num_chunks: desired parallelism, this may be modified
            internally based on the available hardware and ``num_samples``
            relative to ``len(points)``. If it is ever 1, perform single-shot
            :meth:`~._fps()` sampling instead.
        :type num_chunks: int
        :param num_samples: number of samples to select from the data
        :type num_samples: int
        :param first_index: index of the first point from which farthest
            distances are calculated to begin the sampling. |default| 0
        :type first_index: int
        :returns: an array of length ``num_samples`` which contains the indices
            (ranging from 0 to len(points)) corresponding to rows of the full
            points matrix for well-spaced points
        :rtype: np.ndarray
        """
        num_envs = len(points)
        # determine maximum parallelism available on the compute resource
        num_procs = cpu_count()
        # set our max pool size
        pool_size = np.minimum(num_procs, num_chunks)
        # ensure that each chunk is large enough - round down here to catch min
        envs_per_chunk = num_envs // pool_size
        # we want each chunk to have at least num_samples + 1
        while envs_per_chunk <= num_samples and pool_size > 1:
            pool_size = np.maximum(pool_size // 2, 1)
            envs_per_chunk = num_envs // pool_size
        self.logger.info(f'Performing multiprocess prune of {num_envs} points '
                         f'across {pool_size} subgroups to {num_samples} '
                         'points')
        # make sure we actually need parallelism first
        if pool_size > 1:
            # chunk the descriptors
            chunked_points = np.array_split(points, pool_size)
            # create the index shifts to map the chunks to the full list
            # indices
            index_shifts = [0] + list(
                np.cumsum([len(a) for a in chunked_points[:-1]]))
            # do the sampling over all the pools in parallel
            with Pool(pool_size) as p:
                # fix the first_index uniformally
                split_fps = partial(
                    self.__class__._fps,
                    first_index=first_index,
                )
                # map the chunked values to the FPS algorithm
                pool_results = p.starmap(
                    split_fps,
                    zip(chunked_points, num_samples, index_shifts),
                )
            # combine the results into a single array
            combined_indices = np.concatenate(pool_results)
            # now do a second sample iteration on the downselected points
            downselected_points = points[combined_indices]
            self.logger.info(f'Final sampling of {num_samples} from '
                             f'downselected set of {len(downselected_points)}')
            downselected_indices = self.__class__._fps(
                downselected_points,
                num_samples,
                first_index=first_index,
            )
            # propagate selection back to the full list
            final_indices = combined_indices[downselected_indices]
        else:
            # based on data and num_samples, only have a single pool - do 1
            # shot FPS sampling instead to avoid parallel and 2-step overhead
            final_indices = self.__class__._fps(
                points,
                num_samples,
                first_index=first_index,
            )
        return final_indices

    def _approximate_multiprocess_prune(
        self,
        points: np.ndarray,
        num_chunks: int,
        num_samples: int,
        first_index: Optional[int] = 0,
    ) -> np.ndarray:
        """
        an approximate FPS algorithm that splits the sampling over num_chunks

        This method uses the multiprocessing module to perform FPS pruning in
        parallel (degree of parallelism is set by ``num_chunks``). It can be
        used for any pruning ratio. It is approximate because ``num_samples/
        num_chunks`` points are selected from each chunk and combined. This may
        result in sub-optimal sampling, but will generally run much faster than
        the single-shot (:meth:`~._fps`) or full multiprocess
        (:meth:`~._multiprocess_prune`) methods.

        :param points: single array of points to sample over, with size NxD
            where N is the number of atomic environments and D is the dimension
            of the descriptor to analyze for distance metric
        :type points: np.ndarray
        :param num_chunks: desired parallelism, this will also split the
            number of samples chosen from each subset evenly
        :type num_chunks: int
        :param num_samples: number of samples to select from the data
        :type num_samples: int
        :param first_index: index of the first point from which farthest
            distances are calculated to begin the sampling. |default| 0
        :type first_index: int
        :returns: an array of length ``num_samples`` which contains the indices
            (ranging from 0 to len(points)) corresponding to rows of the full
            points matrix for well-spaced points
        :rtype: np.ndarray
        """
        # determine maximum parallelism available on the compute resource
        num_procs = cpu_count()
        # set our max pool size
        pool_size = np.minimum(num_procs, num_chunks)
        # divide the num_samples across pool_size
        per_chunk_samples = [
            len(a) for a in np.array_split(np.arange(num_samples), pool_size)
        ]
        self.logger.info(f'Performing approximate prune of {len(points)} '
                         f'points across {pool_size} subgroups to '
                         f'{num_samples} points')
        # chunk the descriptors
        chunked_points = np.array_split(points, pool_size)
        # create the index shifts to map the chunks to the full list indices
        index_shifts = [0] + list(
            np.cumsum([len(a) for a in chunked_points[:-1]]))
        # do the sampling over all the pools in parallel
        with Pool(pool_size) as p:
            # fix the first_index uniformally
            split_fps = partial(
                self.__class__._fps,
                first_index=first_index,
            )
            # map the chunked values to the FPS algorithm
            pool_results = p.starmap(
                split_fps,
                zip(chunked_points, per_chunk_samples, index_shifts),
            )
        # combine the results into a single array
        combined_indices = np.concatenate(pool_results)
        return combined_indices

    def iterative_fps_prune(
        self,
        dataset: list[Atoms],
        descriptors_key: str,
        prune_approach: Union['QUESTSEfficiencyScore'],
        num_chunks: Optional[int] = 1,
        prune_ratio_args: Optional[dict[str, Any]] = None,
        pruning_convergence: Optional[float] = 0.01,
        iteration_limit: Optional[int] = None,
        fps_approach: Optional[str] = 'full',
        first_index: Optional[int] = 0,
        _print_pid: Optional[bool] = False,
    ) -> list[Atoms]:
        """
        Iteratively apply an FPS algorithm to select an optimally diverse set

        Use an iterative estimation of the dataset's information content to set
        a pruning ratio, removing that fraction of the dataset until the
        pruning ratio is below the convergence threshold. Different sampling
        approaches are supported for executing the pruning of the dataset,
        including the ability to split pruning over multiple processors.
        Pruning is performed on precomputed descriptors which are provided with
        the dataset. For large or production level runs, consider using the
        :meth:`~.chunked_iterative_fps_prune` method instead, as it will scale
        much more favorably.

        :param dataset: list of Atoms which includes precomputed descriptors.
            Can optionally include existing ``SELECTION_MASK_KEY`` data.
        :type dataset: list[Atoms]
        :param descriptors_key: string key for the computed descriptors over
            which we will compute distance for sampling
        :type descriptors_key: str
        :param prune_approach: a score module that can be used to provide a
            pruning quantity based on the information of the dataset. Currently
            only QUESTSEfficiencyScore is supported
        :type prune_approach: Score
        :param num_chunks: degree of parallelism to apply to sampling, if
            desired. |default| 1
        :type num_chunks: int
        :param prune_ratio_args: arguments needed to run the score computation
            |default| ``None``
        :type prune_ratio_args: dict
        :param pruning_convergence: value below which pruning will stop
            |default| 0.01
        :type pruning_convergence: float
        :param iteration_limit: variable to limit the number of iterations that
            can be performed in the pruning loop. Uses the values set at
            Augmentor initialization (default of 30) if None |default| ``None``
        :type iteration_limit: int
        :param fps_approach: selector for the FPS method to use. Supported
            options are 'full', 'multi', and 'approximate'. The 'full'
            approach will carry out serial FPS pruning on the full dataset.
            This option is the most accurate and most expensive. The 'multi'
            option divides the dataset into `num_chunks` and extracting the
            full number of samples in each chunk, combining these pruned
            divisions and pruning a second time on the full set. Because the
            full number of samples is pruned from each chunk, the minimum
            pruning ratio is 50% (supporting 2 chunks). See
            :meth:`~._multiprocess_prune` for more detail. The 'approximate'
            option will also divide the full dataset into `num_chunks`, but
            extract only an equal proportion of the total number of samples
            from each chunk. This method will be the fastest, but also least
            accurate, as there are no guarantees for the distribution of
            points within or between different chunks. See
            :meth:`~._approximate_multiprocess_prune` for more details.
            |default| `'full'`
        :type fps_approach: str
        :param first_index: first index to use in the FPS algorithm. If not
            supplied, 0 will always be used. The index is updated through
            the pruning iterations to track a given sample if non-zero. This
            assumes that the index corresponds to the atom index in the full
            dataset. |default| 0
        :type first_index: int
        :param _print_pid: internal variable to print the associated PID in
            logger data. Used when this method is called from within a
            multiprocessing call |default| ``False``
        :type _print_pid: bool
        :returns: a list of atoms matching the input dataset with a new (or)
            updated array with key ``SELECTION_MASK_KEY`` which is True for
            atomic environments which should be considered/included in
            subsequent operations.
        :rtype: list[Atoms]
        """
        converged = False
        if iteration_limit is None:
            iteration_limit = self.default_iteration_limit
        if _print_pid:
            pid = getpid()
            pid_string = f'[{pid}] '
        else:
            pid_string = ''
        if isinstance_no_import(prune_approach, 'QUESTSEfficiencyScore'):
            if prune_ratio_args is None:
                full_prune_ratio_args = {}
            else:
                full_prune_ratio_args = deepcopy(prune_ratio_args)
            full_prune_ratio_args |= {
                'score_quantity': 'EFFICIENCY',
                'apply_mask': True,
                'descriptors_key': descriptors_key,
            }
        else:
            raise TypeError('Unsupported type for prune_approach')
        # the first pruned version is the full dataset
        pruned_dataset = deepcopy(dataset)
        # save the dataset shape (# atoms per config)
        dataset_shapes = [len(config) for config in pruned_dataset]
        for config in pruned_dataset:
            # set the inital mask to include all envs if no mask already
            if SELECTION_MASK_KEY not in config.arrays.keys():
                config.set_array(SELECTION_MASK_KEY,
                                 np.ones(len(config), dtype=bool))
        # get the estimated pruning ratio, pruned_dataset may have
        # selection mask from previous iteration
        prune_amount = self.estimate_pruning_ratio(
            pruned_dataset,
            prune_approach,
            full_prune_ratio_args,
        )
        # if we're pruning less than the convergences amount i.e. 1%, skip
        if prune_amount <= pruning_convergence:
            converged = True
            self.logger.info(f'{pid_string}Dataset is already pruned beyond '
                             'convergence criteria, skipping pruning step!')

        # use iterations to make sure we don't get stuck in the loop
        iterations = 0
        while not converged and iterations < iteration_limit:
            iterations += 1
            init_selection_masks = [
                config.get_array(SELECTION_MASK_KEY)
                for config in pruned_dataset
            ]
            # adjust the first_index to masked envs
            if first_index != 0:
                selected_first_index = np.sum(
                    np.concatenate(init_selection_masks)[:first_index])
            else:
                selected_first_index = 0
            # only count envs which can be selected
            num_envs = np.sum([np.sum(mask) for mask in init_selection_masks])
            # this is the number of environments we want to prune to (selected
            # by the FPS algorithm)
            sample_target = int(np.round(num_envs * (1 - prune_amount)))
            # ensure we can actually prune down the dataset
            if sample_target == num_envs or sample_target == 0:
                # no more pruning is possible due to dataset size
                self.logger.info(
                    f"{pid_string}With prune amount of {prune_amount}, can't "
                    f'reduce dataset past {num_envs} atomic environments')
                break
            self.logger.info(f'{pid_string}Starting with {num_envs} envs from '
                             f'{len(pruned_dataset)} configurations, pruning '
                             f'to {sample_target} ({(1 - prune_amount) * 100}'
                             '%)')
            # prune over distance in descriptor space, with a single array of
            # all atomic envs
            selected_descriptors = []
            for config, mask in zip(pruned_dataset, init_selection_masks):
                all_descriptors = config.get_array(descriptors_key)
                selected_descriptors.append(all_descriptors[mask])
            combined_descriptors = np.concatenate(selected_descriptors)
            # sanity check for shapes
            if num_envs != combined_descriptors.shape[0]:
                raise RuntimeError(f'The number of environments ({num_envs}) '
                                   'does not match the number of descriptors ('
                                   f'{combined_descriptors.shape[0]})')
            # determine how to apply the FPS algo based on the parameters and
            # provided approach. len(keep_indices) = sample_target
            if num_chunks == 1 or selected_first_index != 0:
                # if num_chunks is 1, then just use single shot
                keep_indices = self.__class__._fps(
                    combined_descriptors,
                    sample_target,
                    first_index=selected_first_index,
                )
            elif fps_approach == 'approximate':
                keep_indices = self._approximate_multiprocess_prune(
                    combined_descriptors,
                    num_chunks,
                    sample_target,
                )
            elif fps_approach == 'multi' and prune_amount > 0.5:
                keep_indices = self._multiprocess_prune(
                    combined_descriptors,
                    num_chunks,
                    sample_target,
                )
            else:
                if fps_approach == 'multi':
                    self.logger.info('fps_approach requested is multi, but '
                                     f'prune_amount > 0.5 ({prune_amount})')
                keep_indices = self.__class__._fps(
                    combined_descriptors,
                    sample_target,
                    first_index=selected_first_index,
                )
            # create mask to propagate the pruning
            combined_keep_mask = np.zeros(np.sum(dataset_shapes), dtype=bool)
            # propagate the new keep_indices to the indices of the full array
            # of the configuration shapes considering the existing mask
            full_data_indices = np.flatnonzero(
                np.concatenate(init_selection_masks))[keep_indices]
            combined_keep_mask[full_data_indices] = True
            # split mask to match shapes of the dataset elements
            dataset_shaped_keep_masks = np.array_split(
                combined_keep_mask,
                np.cumsum(dataset_shapes)[:-1],
            )
            # re-write the SELECTION_MASK_KEY in each config with the updated
            # one
            for config, mask in zip(pruned_dataset, dataset_shaped_keep_masks):
                config.set_array(SELECTION_MASK_KEY, mask)

            # get the estimated pruning ratio, pruned_dataset may have
            # selection mask from previous iteration
            prune_amount = self.estimate_pruning_ratio(
                pruned_dataset,
                prune_approach,
                full_prune_ratio_args,
            )
            # if we're pruning less than the convergences amount i.e. 1%, stop
            if prune_amount <= pruning_convergence:
                # we are done
                converged = True
                self.logger.info(
                    f'{pid_string}Iterative prune completed in {iterations} '
                    'iterations, stopping with an estimated pruning ratio of '
                    f'{prune_amount}.')

        if not converged:
            self.logger.info(
                f'{pid_string}Pruning did not converge in {iteration_limit} '
                f'steps. Dataset pruned to {np.sum(combined_keep_mask)} / '
                f'{np.sum(dataset_shapes)} atomic environments')

        return pruned_dataset

    def chunked_iterative_fps_prune(
        self,
        dataset: list[Atoms],
        descriptors_key: str,
        prune_approach: Union['QUESTSEfficiencyScore'],
        num_chunks: Optional[int] = 1,
        prune_ratio_args: Optional[dict[str, Any]] = None,
        pruning_convergence: Optional[float] = 0.01,
        iteration_limit: Optional[int] = None,
        first_index: Optional[int] = 0,
        hierarchical_parallelism: Optional[bool] = False,
    ) -> list[Atoms]:
        """
        Iteratively apply an FPS algorithm to select an optimally diverse set

        Use an iterative estimation of the dataset's information content to set
        a pruning ratio, removing that fraction of the dataset until the
        pruning ratio is below the convergence threshold. This function
        performs complete pruning on each subset of the data independently
        prior to recombining the full dataset for pruning, allowing for better
        scaling of the pruning to large datasets and a more nuanced treatment
        of the pruning of the subsets compared to the
        :meth:`~.iterative_fps_prune` method. If the `hierarchical_parallelism`
        option is turned on, multiple rounds of increasingly coarse-grained
        chunking can be employed to refine the dataset. Pruning is performed on
        precomputed descriptors which are provided with the dataset.

        :param dataset: list of Atoms which includes precomputed descriptors.
            Can optionally include existing ``SELECTION_MASK_KEY`` data.
        :type dataset: list[Atoms]
        :param descriptors_key: string key for the computed descriptors over
            which we will compute distance for sampling
        :type descriptors_key: str
        :param prune_approach: a score module that can be used to provide a
            pruning quantity based on the information of the dataset. Currently
            only QUESTSEfficiencyScore is supported
        :type prune_approach: Score
        :param num_chunks: degree of parallelism to apply to sampling, if
            desired. |default| 1
        :type num_chunks: int
        :param prune_ratio_args: arguments needed to run the score computation
            |default| ``None``
        :type prune_ratio_args: dict
        :param pruning_convergence: value below which pruning will stop
            |default| 0.01
        :type pruning_convergence: float
        :param iteration_limit: variable to limit the number of iterations that
            can be performed in the pruning loop. Uses the values set at
            Augmentor initialization (default of 30) if None |default| ``None``
        :type iteration_limit: int
        :param first_index: first index to use in the FPS algorithm. If not
            supplied, 0 will always be used. The index is updated through
            the pruning iterations to track a given sample if non-zero. This
            assumes that the index corresponds to the atom index in the full
            dataset. |default| 0
        :type first_index: int
        :param hierarchical_parallelism: flag for running the chunking
            iteratively with number of chunks halved at each iteration. If
            False, just do a single step of chunking |default| ``False``
        :type hierarchical_parallelism: bool
        :returns: a list of atoms matching the input dataset with a new (or)
            updated array with key ``SELECTION_MASK_KEY`` which is True for
            atomic environments which should be considered/included in
            subsequent operations.
        :rtype: list[Atoms]
        """
        if iteration_limit is None:
            iteration_limit = self.default_iteration_limit
        if isinstance_no_import(prune_approach, 'QUESTSEfficiencyScore'):
            if prune_ratio_args is None:
                full_prune_ratio_args = {}
            else:
                full_prune_ratio_args = deepcopy(prune_ratio_args)
            full_prune_ratio_args |= {
                'score_quantity': 'EFFICIENCY',
                'apply_mask': True,
                'descriptors_key': descriptors_key,
            }
        else:
            raise TypeError('Unsupported type for prune_approach')
        # the first pruned version is the full dataset
        pruned_dataset = deepcopy(dataset)
        for config in pruned_dataset:
            # set the inital mask to include all envs if no mask already
            if SELECTION_MASK_KEY not in config.arrays.keys():
                config.set_array(SELECTION_MASK_KEY,
                                 np.ones(len(config), dtype=bool))
        # determine maximum parallelism available on the compute resource
        num_procs = cpu_count()
        # set our max pool size
        pool_size = np.min([len(pruned_dataset), num_procs, num_chunks])

        dataset_size = np.sum([len(c) for c in pruned_dataset])
        self.logger.info('Starting pruning on the dataset containing '
                         f'{dataset_size} atomic envs across '
                         f'{len(pruned_dataset)} configs')
        exit_early = False
        while pool_size > 1 and not exit_early:
            num_pruned_envs = np.sum([
                np.sum(c.get_array(SELECTION_MASK_KEY)) for c in pruned_dataset
            ])
            self.logger.info(f'Now pruning on {pool_size} splits of '
                             f'{num_pruned_envs} selected atomic envs.')
            # if first index is not provided, start_indices will be all 0
            # otherwise it will be all 0 with exception of the split which
            # contains the true "first_index"
            (
                config_splits,
                split_index_maps,
                start_indices,
            ) = self.__class__._approximate_even_configuration_split(
                pruned_dataset,
                pool_size,
                first_index,
            )
            full_indexes = list(
                itertools.chain.from_iterable(split_index_maps))
            # do iterative pruning on the chunks in parallel
            with Pool(pool_size) as p:
                # construct full list instead of using partial since we are
                # setting some required and some optional arguments
                full_args_list = zip(
                    config_splits,
                    itertools.repeat(descriptors_key),
                    itertools.repeat(prune_approach),
                    itertools.repeat(1),
                    itertools.repeat(full_prune_ratio_args),
                    itertools.repeat(pruning_convergence),
                    itertools.repeat(iteration_limit),
                    itertools.repeat('full'),
                    start_indices,
                    itertools.repeat(True),
                )
                # pool results are a list of lists of ASE Atoms
                pool_results = p.starmap(
                    self.iterative_fps_prune,
                    full_args_list,
                )
            # recombine and reorder the lists
            full_configs = list(itertools.chain.from_iterable(pool_results))
            # pruned_dataset now is in the same order as originally, but with
            # SELECTION_MASK_KEY arrays set based on independent pruning of the
            # chunks
            pruned_dataset = [
                c for _, c in sorted(zip(full_indexes, full_configs),
                                     key=lambda pair: pair[0])
            ]
            num_pruned_envs = np.sum([
                np.sum(c.get_array(SELECTION_MASK_KEY)) for c in pruned_dataset
            ])
            self.logger.info(f'Pruning with {pool_size} splits completed, '
                             f'selecting {num_pruned_envs} atomic envs.')
            pool_size = np.maximum(pool_size // 2, 1)
            if not hierarchical_parallelism:
                # user does not want to iterate on decreasing split sizes
                exit_early = True
        # do a final prune on the full dataset
        self.logger.info('Split pruning done, final prune on full set now')
        final_pruned_dataset = self.iterative_fps_prune(
            pruned_dataset,
            descriptors_key,
            prune_approach,
            num_chunks=1,
            prune_ratio_args=full_prune_ratio_args,
            pruning_convergence=pruning_convergence,
            iteration_limit=iteration_limit,
            fps_approach='full',
            # global position of the first index should remain constant
            # following recombination from the splits
            first_index=first_index,
        )
        return final_pruned_dataset

    @classmethod
    def _approximate_even_configuration_split(
        cls,
        configs: list[Atoms],
        num_splits: int,
        start_index: Optional[int] = None,
    ) -> tuple[list[list[Atoms]], list[list[int]], np.ndarray[int]]:
        """
        internal method for dividing a dataset based on number of atoms

        Use a greedy approach to split the dataset into num_splits divisions
        while keeping track of the original indices so that the original order
        can be recovered

        :param configs: list of Atoms to split approximately evenly
        :type configs: list[Atoms]
        :param num_splits: number of divisions to split the dataset into
        :type num_splits: int
        :param start_index: optional index to track for starting pruning on
        :type start_index: int
        :returns: a list of Atoms lists, a list of index lists where each
            of the Atoms lists is roughly equal size in terms of the number of
            atomic environments, and a list of start indices. The index lists
            denote the original index of each atom and can be used to
            reconstruct the original ordering. The list of start indices will
            always be all zeros, unless a ``start_index`` is provided. If one
            is, then one of the elements will correspond to that element from
            the full list. This can be used to ensure a desired start_index is
            included after splitting
        :rtype: tuple(list(list[Atoms]), list(list(int)), np.ndarray)
        """
        if start_index is not None:
            # we want to keep track of a specific index over the splits
            cumulative_lengths = np.cumsum([len(_) for _ in configs])
            for i, lens in enumerate(cumulative_lengths):
                # desired index is in this config
                if lens > start_index:
                    config_index = i
                    break
            # index within the config
            atom_index = start_index - cumulative_lengths[config_index - 1]
        else:
            # no configs with special atoms to track
            config_index = -1
        # Create a list of (size, original_index) tuples
        indexed_configs = [(i, c) for i, c in enumerate(configs)]
        # Initialize the chunks for the greedy sort
        config_splits = [[] for _ in range(num_splits)]
        split_index_maps = [[] for _ in range(num_splits)]
        start_indices = np.zeros(num_splits, dtype=int)
        split_sizes = np.zeros(num_splits, dtype=int)
        # Sort groups in descending order based on size
        sorted_sizes = sorted(
            indexed_configs,
            # sort by the length of selected envs in the configs
            key=lambda x: np.sum(x[1].get_array(SELECTION_MASK_KEY)),
            reverse=True,
        )
        for original_index, config in sorted_sizes:
            # Find the chunk with the smallest current size
            smallest_split_index = np.argmin(split_sizes)
            # check if this config contains the noted start atom
            if original_index == config_index:
                current_length = np.sum(
                    [len(_) for _ in config_splits[smallest_split_index]])
                adjusted_atom_index = atom_index + current_length
                # set the start index for this split to find the noted atom
                start_indices[smallest_split_index] += adjusted_atom_index
            # Add the config to that split
            config_splits[smallest_split_index].append(config)
            split_index_maps[smallest_split_index].append(original_index)
            # update the split sizes for next iteration
            # always increase by at least 1 to ensure even distribution, even
            # with multiple "empty" configs
            split_sizes[smallest_split_index] += np.maximum(
                1, np.sum(config.get_array(SELECTION_MASK_KEY)))

        return config_splits, split_index_maps, start_indices
