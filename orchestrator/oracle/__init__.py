from .oracle_base import Oracle
from .factory import OracleBuilder, oracle_factory, oracle_builder

__all__ = [
    'Oracle',
    'OracleBuilder',
    'oracle_factory',
    'oracle_builder',
]

try:
    from .factory import aiida_oracle_builder  # noqa: F401
    from .aiida.oracle_base import AiidaOracle  # noqa: F401
    from .aiida.espresso import AiidaEspressoOracle  # noqa: F401
    from .aiida.vasp import AiidaVaspOracle  # noqa: F401
    __all__.extend([
        'aiida_oracle_builder',
        'AiidaOracle',
        'AiidaEspressoOracle',
        'AiidaVaspOracle',
    ])
except ImportError:
    pass
