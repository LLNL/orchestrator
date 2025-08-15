from abc import ABC, abstractmethod
from ..utils.recorder import Recorder
from typing import Optional, Dict, Any, Union
from ..workflow.factory import workflow_builder
from ..workflow import Workflow
from ..storage import Storage
from ..potential import Potential


class TargetProperty(Recorder, ABC):
    """
    General class to manage target property calculations

    :param target_property_args: general argument structure which is specified
        by individual implementations
    :type args: dict
    """

    def __init__(self, **target_property_args):
        """
        :param target_property_args: general argument structure which is
            specified by individual implementations
        :type target_property_args: dict
        """

        super().__init__()
        self.args = target_property_args
        self.checkpoint_file = target_property_args.get(
            'checkpoint_file',
            './orchestrator_checkpoint.json',
        )
        self.checkpoint_name = target_property_args.get(
            'checkpoint_name', 'property')

        self.default_wf = workflow_builder.build(
            'LOCAL',
            {'root_directory': './target_property'},
        )
        self.restart_property()

    @abstractmethod
    def checkpoint_property(self):
        """
        checkpoint the property module into the checkpoint file

        save necessary internal variables into a dict with key checkpoint_name
        and write to the (json) checkpoint file for restart capabilities
        """
        pass

    @abstractmethod
    def restart_property(self):
        """
        restart the property module from the checkpoint file

        check if the checkpoint_file has an entry matching the checkpoint_name
        and set internal variables accordingly if so
        """
        pass

    @abstractmethod
    def calculate_property(
        self,
        iter_num: int = 0,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        workflow: Optional[Workflow] = None,
        storage: Optional[Storage] = None,
        **kwargs,
    ):
        """
        Perform analysis to calculate a property of interest.

        Derived classes should list explicit arguments required
        to calculate their properties. This module can utilize
        other modules within the orchestrator to carry out the
        target calculations.

        :param potential: interatomic potential to be used in LAMMPS
        :type potential: str
        :param workflow: the workflow for managing job submission, if none are
            supplied, will use the default workflow defined in this class
            |default| ``None``
        :type workflow: Workflow
        :returns: a dictionary with property output, errors, and calc ids as
            a tuple (different indices can correspond to different calc types)
        :rtype: dict
        """
        pass

    @abstractmethod
    def conduct_sim(
        self,
        sim_params: Dict[str, Any],
        workflow: Workflow,
        sim_path: str,
    ) -> int:
        """
        Perform the simulation for the target property calculations

        sim_params is a dictionary of key-value pairs that can
        be used to define various parameters related to conducting
        simulations (e.g. temperature, pressure, random seed, etc..).
        The dictionary is described in the input json file.

        :param sim_params: simulation specific parameters
        :type sim_params: dict
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :param sim_path: path to perform simulations for
            target property calculations
        :type sim_path: str
        """
        pass

    @abstractmethod
    def calculate_with_error(
        self,
        n_calc: int,
        modified_params: Optional[Dict[str, Any]] = None,
        potential: Optional[Union[str, Potential]] = None,
        workflow: Optional[Workflow] = None,
    ):
        """
        Calculate a target property with mean and standard deviation
        Derived classes should list explicit arguments required
        to calculate their properties.

        Mean and standard deviation will be obtained from multiple
        number of calculations (n_calc)

        :param n_calc: total number of calculations to perform
        :type n_calc: int
        :param potential: interatomic potential to be used in LAMMPS
        :type potential: str
        :param workflow: the workflow for managing job submission
        :type workflow: Workflow
        :returns: mean and standard deviation of the calculated property
        """
        pass
