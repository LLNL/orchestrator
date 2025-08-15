from orchestrator.utils.module_factory import ModuleFactory, ModuleBuilder
from .descriptor_base import DescriptorBase, AtomCenteredDescriptor
from orchestrator.utils.exceptions import ModuleAlreadyInFactoryError

from typing import Any

#: default factory for descriptors
descriptor_factory = ModuleFactory(AtomCenteredDescriptor)


class DescriptorBuilder(ModuleBuilder):
    """
    Constructor for descriptors added in the factory

    Set the factory to be used for the builder. The default is to use the
    descriptor_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a descriptor factory |default| :data:`descriptor_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory: ModuleFactory = descriptor_factory):
        """
        Constructor for the DescriptorBuilder, sets the factory to build from

        :param factory: a descriptor factory |default|
            :data:`descriptor_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == AtomCenteredDescriptor.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for descriptors!')

    def build(self, descriptor_type: str,
              descriptor_args: dict[str, Any]) -> DescriptorBase:
        """
        Return an instance of the specified descriptor

        The build method takes the specifier and input arguments to construct
        a concrete descriptor instance.

        :param descriptor_type: token of a descriptor which has been added to
            the factory
        :type descriptor_type: str
        :param descriptor_args: dictionary of arguments needed to instantiate a
            given descriptor class
        :type descriptor_args: dict
        :returns: instantiated concrete descriptor
        :rtype: DescriptorBase, or child class
        """
        match descriptor_type:
            case 'KLIFFDescriptor':
                from .kliff import KLIFFDescriptor
                try:
                    descriptor_factory.add_new_module('KLIFFDescriptor',
                                                      KLIFFDescriptor)
                except ModuleAlreadyInFactoryError:
                    # Do nothing if module already added previously.
                    # Note that this no longer catches ImportError or
                    # ModuleNotFoundError, since these SHOULD be raised here.
                    pass
            case 'QUESTSDescriptor':
                from .quests import QUESTSDescriptor
                try:
                    descriptor_factory.add_new_module('QUESTSDescriptor',
                                                      QUESTSDescriptor)
                except ModuleAlreadyInFactoryError:
                    pass

        descriptor_constructor = self.factory.select_module(descriptor_type)
        return descriptor_constructor(**descriptor_args)


#: descriptor builder object which can be imported for use in other modules
descriptor_builder = DescriptorBuilder()
