from ..utils.module_factory import ModuleFactory, ModuleBuilder
from ..utils.exceptions import ModuleAlreadyInFactoryError
from .storage_base import Storage

#: default factory for oracles, includes QE, LAMMPS, and KIM
storage_factory = ModuleFactory(Storage)


class StorageBuilder(ModuleBuilder):
    """
    Constructor for storage modules added in the factory

    set the factory to be used for the builder. The default is to use the
    storage_factory generated at the end of this module. A user defined
    StorageFactory can optionally be supplied instead.

    :param factory: a storage factory |default| :data:`storage_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory=storage_factory):
        """
        constructor for the StorageBuilder, sets the factory to build from

        :param factory: a storage factory |default| :data:`storage_factory`
        :type factory: ModuleFactory
        """
        if factory.base_class.__name__ == Storage.__name__:
            super().__init__(factory)
        else:
            raise Exception('Supplied factory is not for Storage!')

    def build(self, storage_type, storage_args=None) -> Storage:
        """
        Return an instance of the specified data storage

        The build method takes the specifier and input arguments to construct
        a concrete storage instance.

        :param storage_type: token of a storage which has been added to the
            factory
        :type storage_type: str
        :param storage_args: dictionary with initialization parameters,
            including database_name and database_path. See module documentation
            for greater detail
        :type storage_args: dict
        :returns: instantiated concrete Storage
        :rtype: Storage
        """
        if storage_args is None:
            storage_args = {}

        match storage_type:
            case 'LOCAL':
                from .local import LocalStorage
                try:
                    storage_factory.add_new_module('LOCAL', LocalStorage)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'COLABFIT':
                from .colabfit import ColabfitStorage
                try:
                    storage_factory.add_new_module('COLABFIT', ColabfitStorage)
                except ModuleAlreadyInFactoryError:
                    pass

        storage_constructor = self.factory.select_module(storage_type)
        built_class = storage_constructor(**storage_args)
        built_class.factory_token = storage_type
        return built_class


#: storage builder object which can be imported for use in other modules
storage_builder = StorageBuilder()
