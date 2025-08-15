from ..utils.module_factory import ModuleFactory, ModuleBuilder
from .augmentor_base import Augmentor
#: default factory for augmentors, includes QE, LAMMPS, and KIM
augmentor_factory = ModuleFactory(Augmentor)
augmentor_factory.add_new_module('BASE', Augmentor)


class AugmentorBuilder(ModuleBuilder):
    """
    Constructor for augmentors added in the factory

    set the factory to be used for the builder. The default is to use the
    augmentor_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a augmentor factory |default| :data:`augmentor_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=augmentor_factory):
        """
        constructor for the AugmentorBuilder, sets the factory to build from

        :param factory: a augmentor factory |default| :data:`augmentor_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Augmentor.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Augmentors!')

    def build(self, augmentor_type, augmentor_args=None) -> Augmentor:
        """
        Return an instance of the specified augmentor

        The build method takes the specifier and input arguments to construct
        a concrete augmentor instance.

        :param augmentor_type: token of a augmentor which has been added to the
            factory
        :type augmentor_type: str
        :param augmentor_args: dictionary of arguments needed to instantiate a
            given augmentor class
        :type augmentor_args: dict
        :returns: instantiated concrete Augmentor
        :rtype: Augmentor
        """
        if augmentor_args is None:
            augmentor_args = {}
        augmentor_constructor = self.factory.select_module(augmentor_type)
        return augmentor_constructor(**augmentor_args)


#: augmentor builder object which can be imported for use in other modules
augmentor_builder = AugmentorBuilder()
