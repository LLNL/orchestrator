class UnidentifiedPathError(Exception):
    __module__ = Exception.__module__


class UnidentifiedStorageError(Exception):
    __module__ = Exception.__module__


class DuplicateDatasetNameError(Exception):
    __module__ = Exception.__module__


class DatasetDoesNotExistError(Exception):
    __module__ = Exception.__module__


class UnknownKeyError(Exception):
    __module__ = Exception.__module__


class UnsupportedComparisonError(Exception):
    __module__ = Exception.__module__


class ProblematicJobStateError(Exception):
    __module__ = Exception.__module__


class JobSubmissionError(Exception):
    __module__ = Exception.__module__


class UnfullfillableDependenciesError(Exception):
    __module__ = Exception.__module__

    def __init__(self):
        super().__init__(('Could not run job, but it has dependencies! '
                          'Check your queue for stray jobs!'))


class InstallPotentialError(Exception):
    __module__ = Exception.__module__


class AnalysisError(Exception):
    __module__ = Exception.__module__


class DensityOOBError(Exception):
    __module__ = Exception.__module__


class CellTooSmallError(Exception):
    __module__ = Exception.__module__


class StrayFilesError(Exception):
    __module__ = Exception.__module__


class TestNotFoundError(Exception):
    __module__ = Exception.__module__


class KIMRunFlattenError(Exception):
    __module__ = Exception.__module__


class ModuleAlreadyInFactoryError(Exception):
    __module__ = Exception.__module__
