from orchestrator.utils.module_factory import ModuleFactory, ModuleBuilder
from orchestrator.utils.exceptions import ModuleAlreadyInFactoryError
from .score_base import ScoreBase

from typing import Any

#: default factory for scoring methods
score_factory = ModuleFactory(ScoreBase)


class ScoreBuilder(ModuleBuilder):
    """
    Constructor for Score methods added in the factory.

    Set the factory to be used for the builder. The default is to use the
    score_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a score factory |default| :data:`score_factory`
    :type factory: ModuleFactory
    """

    def __init__(
        self,
        factory: ModuleFactory = score_factory,
    ):
        """
        Constructor for the ScoreBuilder, sets the factory to build from.

        :param factory: a score factory |default| :data:`score_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == ScoreBase.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Score!')

    def build(
        self,
        score_type: str,
        score_args: dict[str, Any],
    ) -> ScoreBase:
        """
        Return an instance of the specified Score method.

        The build method takes the specifier and input arguments to construct
        a concrete Score module instance.

        :param score_type: token of a score method which has been added to
            the factory
        :type score_type: str
        :param score_args: dictionary of arguments needed to instantiate a
            given Score class
        :type score_args: dict
        :returns: instantiated concrete Score
        :rtype: Score
        """
        if score_args is None:  # for modules with no init_args
            score_args = {}

        match score_type:
            case 'LTAUForcesUQScore':
                from .ltau import LTAUForcesUQScore
                try:
                    score_factory.add_new_module('LTAUForcesUQScore',
                                                 LTAUForcesUQScore)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'QUESTSEfficiencyScore':
                from .quests import QUESTSEfficiencyScore
                try:
                    score_factory.add_new_module('QUESTSEfficiencyScore',
                                                 QUESTSEfficiencyScore)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'QUESTSDiversityScore':
                from .quests import QUESTSDiversityScore
                try:
                    score_factory.add_new_module('QUESTSDiversityScore',
                                                 QUESTSDiversityScore)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'QUESTSDeltaEntropyScore':
                from .quests import QUESTSDeltaEntropyScore
                try:
                    score_factory.add_new_module('QUESTSDeltaEntropyScore',
                                                 QUESTSDeltaEntropyScore)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FIMTrainingSetScore':
                from .fim import FIMTrainingSetScore
                try:
                    score_factory.add_new_module('FIMTrainingSetScore',
                                                 FIMTrainingSetScore)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FIMPropertyScore':
                from .fim import FIMPropertyScore
                try:
                    score_factory.add_new_module('FIMPropertyScore',
                                                 FIMPropertyScore)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FIMMatchingScore':
                from .fim import FIMMatchingScore
                try:
                    score_factory.add_new_module('FIMMatchingScore',
                                                 FIMMatchingScore)
                except ModuleAlreadyInFactoryError:
                    pass

        score_constructor = self.factory.select_module(score_type)
        return score_constructor(**score_args)


#: score method builder object which can be imported for use in other modules
score_builder = ScoreBuilder()
