from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .potential_base import Potential

#: default factory for potentials, includes DNN (Behler Parrinello) and KIM
potential_factory = ModuleFactory(Potential)


class PotentialBuilder(ModuleBuilder):
    """
    Constructor for potentials added in the factory

    set the factory to be used for the builder. The default is to use the
    potential_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a potential factory |default| :data:`potential_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=potential_factory):
        """
        constructor for the PotentialBuilder, sets the factory to build from

        :param factory: a potential factory |default| :data:`potential_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Potential.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Potentials!')

    def build(self, potential_type, potential_args=None) -> Potential:
        """
        Return an instance of the specified potential

        The build method takes the specifier and input arguments to construct
        a concrete potential instance.

        :param potential_type: token of a potential which has been added to the
            factory
        :type potential_type: str
        :param potential_args: input arguments to instantiate the requested
            potential class
        :type args: dict
        :returns: instantiated concrete Potential
        :rtype: Potential
        """
        if potential_args is None:
            potential_args = {}

        match potential_type:
            case 'DNN':
                from .dnn import KliffBPPotential
                try:
                    potential_factory.add_new_module('DNN', KliffBPPotential)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'KIM':
                from .kim import KIMPotential
                try:
                    potential_factory.add_new_module('KIM', KIMPotential)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'FitSnap':
                from .fitsnap import FitSnapPotential
                try:
                    potential_factory.add_new_module('FitSnap',
                                                     FitSnapPotential)
                except ModuleAlreadyInFactoryError:
                    pass

        potential_constructor = self.factory.select_module(potential_type)
        built_class = potential_constructor(**potential_args)
        built_class.factory_token = potential_type
        return built_class


#: potential builder object which can be imported for use in other modules
potential_builder = PotentialBuilder()
