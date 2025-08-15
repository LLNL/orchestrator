from typing import Optional
from ..utils.exceptions import ModuleAlreadyInFactoryError
from ..utils.module_factory import ModuleFactory, ModuleBuilder
from .oracle_base import Oracle

#: default factory for oracles, includes QE, LAMMPS, and KIM
oracle_factory = ModuleFactory(Oracle)


# This is a dummy oracle to avoid having to import AiidaOracle at the
# top level of this file.
class AiidaOracle:
    pass


aiida_factory = ModuleFactory(AiidaOracle)


class OracleBuilder(ModuleBuilder):
    """
    Constructor for oracles added in the factory

    Set the factory to be used for the builder. The default is to use the
    oracle_factory generated at the end of this module. A user defined
    ModuleFactory can optionally be supplied instead.

    :param factory: a oracle factory |default| :data:`oracle_factory`
    :type factory: ModuleFactory
    """

    def __init__(self, factory: Optional[ModuleFactory] = oracle_factory):
        """
        Constructor for the OracleBuilder, sets the factory to build from

        :param factory: An oracle factory |default| :data:`oracle_factory`
        """
        name = factory.base_class.__name__
        # use string to avoid top-level aiida import
        valid_names = [Oracle.__name__, 'AiidaOracle']

        if name in valid_names:
            super().__init__(factory)
        else:
            raise Exception(f'Supplied factory {name} is not in {valid_names}')

    def build(self, oracle_type: str, oracle_args: dict = None) -> Oracle:
        """
        Return an instance of the specified oracle.

        The build method takes the specifier and input arguments to construct
        a concrete oracle instance.

        :param oracle_type: Token of an oracle which has been added to the
            factory.
        :param oracle_args: Dictionary of arguments needed to instantiate a
            given oracle class.
        :returns: Instantiated concrete Oracle.
        """
        match oracle_type:
            case 'QE':
                from .espresso import EspressoOracle
                try:
                    oracle_factory.add_new_module('QE', EspressoOracle)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'LAMMPSKIM':
                from .lammps import LAMMPSKIMOracle
                try:
                    oracle_factory.add_new_module('LAMMPSKIM', LAMMPSKIMOracle)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'LAMMPSSnap':
                from .lammps import LAMMPSSnapOracle
                try:
                    oracle_factory.add_new_module('LAMMPSnap',
                                                  LAMMPSSnapOracle)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'KIM':
                from .kim import KIMOracle
                try:
                    oracle_factory.add_new_module('KIM', KIMOracle)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'AiiDA-QE':
                # Have to overwrite dummy factory's base_class
                from .aiida.oracle_base import AiidaOracle
                aiida_factory.base_class = AiidaOracle

                from .aiida.espresso import AiidaEspressoOracle
                try:
                    aiida_factory.add_new_module('AiiDA-QE',
                                                 AiidaEspressoOracle)
                except ModuleAlreadyInFactoryError:
                    pass
            case 'AiiDA-VASP':
                # Have to overwrite dummy factory's base_class
                from .aiida.oracle_base import AiidaOracle
                aiida_factory.base_class = AiidaOracle

                from .aiida.vasp import AiidaVaspOracle
                try:
                    aiida_factory.add_new_module('AiiDA-VASP', AiidaVaspOracle)
                except ModuleAlreadyInFactoryError:
                    pass

        oracle_constructor = self.factory.select_module(oracle_type)
        return oracle_constructor(**oracle_args)


#: oracle builder object which can be imported for use in other modules
oracle_builder = OracleBuilder()
try:
    #: oracle builder object which can be used to build aiida oracles
    aiida_oracle_builder = OracleBuilder(factory=aiida_factory)
except NameError:
    # aiida factory does not exist, so don't create the builder
    pass
