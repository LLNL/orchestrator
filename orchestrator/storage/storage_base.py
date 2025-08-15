from abc import ABC, abstractmethod
from ase import Atoms
from uuid import uuid4
from typing import Optional
from ..utils.recorder import Recorder


class Storage(Recorder, ABC):
    """
    Abstract base class for data storage

    The Storage class deals with all functionalities associated to data storage
    inside Orchestrator. Its functions include the initialization of the
    database, and data additions, updates, and queries. The Orchestrator uses
    a list of ASE Atoms as the internal data representation. A given database
    (Storage instance) can include multiple datasets (collections of
    configurations and properties) and generally persists in time.

    :param storage_args: dictionary with initialization parameters, including
        database_name and database_path. See module documentation for greater
        detail
    :type storage_args: dict
    """

    def __init__(self, **kwargs):
        """
        Set variables and initialize the recorder

        :param storage_args: dictionary with initialization parameters,
            including database_name and database_path. See module documentation
            for greater detail
        :type storage_args: dict
        """
        super().__init__()
        # this should be set by children
        self.STORAGE_ID_KEY = 'storage_id'

    def generate_dataset_name(
        self,
        root: str,
        specifier: str,
        counter: Optional[int] = None,
        check_uniqueness: Optional[bool] = True,
    ) -> str:
        """
        generate a detailed (mostly) human-readable dataset name

        The dataset name will be in the form: root_specifier[_counter] and if
        check_uniqueness is true, root_specifier[_counter]_unique_hash.

        :param root: root of the dataset name, this should be consistent across
            similar runs (i.e. a campaign name)
        :type root: str
        :param specifier: this argument gives more fine control of the dataset
            name, allowing differentiation within a given root
        :type specifier: str
        :param counter: iteration number of the present root and specifier
            combination. This can be used for versioning of datasets.
            |default| ``None``
        :type counter: int
        :param check_uniqueness: attaches a random hash to the dataset name if
            true, and ensures that the resulting dataset name is unique within
            the storage module. |default| ``True``
        :type check_uniqueness: boolean
        :returns: the dataset name
        :rtype: str
        """
        base_string = f'{root}_{specifier}'
        if counter is not None:
            base_string += f'_{counter}'
        if check_uniqueness:
            is_unique = False
            while not is_unique:
                # there may be a more robust way to do this?
                random_id = f'{uuid4().time_low:010}'
                full_name = f'{base_string}_{random_id}'
                is_unique = self.check_if_dataset_name_unique(full_name)
        else:
            full_name = base_string
        return full_name

    @abstractmethod
    def check_if_dataset_name_unique(self, dataset_name: str) -> bool:
        """
        check if the provided dataset_name is unique in the database

        :param dataset_name: name to check (human readable)
        :type dataset_name: str
        :returns: true if the database is not present in the database, false if
            it does exist
        :rtype: boolean
        """
        pass

    @abstractmethod
    def add_data(
        self,
        dataset_handle: str,
        data: list[Atoms],
        dataset_metadata: Optional[dict] = None,
    ) -> str:
        """
        Add new configurations (and associated properties) to the database

        This method is used to add to an existing dataset with new
        configurations. The new configurations may or may not have other
        properties associated with them.

        :param dataset_handle: name or ID of dataset
        :type dataset_handle: str or int
        :param data: list of ASE.Atoms objects containing the configurations
            and associated properties to add to the database. Note that
            configuration-specific metadata should be stored under the
            `atoms.info[METADATA_KEY]` field.
        :type data: list
        :param dataset_metadata: A dictionary of metadata specific to the
            dataset as a whole.
        :type dataset_metadata: dict
        :returns: handle for the dataset which includes the new additions
        :rtype: str or int
        """
        pass

    @abstractmethod
    def new_dataset(
        self,
        dataset_handle: str,
        data: list[Atoms],
        dataset_metadata: Optional[dict] = None,
    ) -> str:
        """
        Create a new dataset with the provided data and metadata

        The new dataset will have a human readable name specificed by
        dataset_handle and will ingest the data and metadata provided.

        :param dataset_handle: name of the dataset to be created
        :type dataset_handle: str
        :param data: list of ASE.Atoms objects containing the configurations
            and associated properties to add to the database. Note that
            configuration-specific metadata should be stored under the
            `atoms.info[METADATA_KEY]` field.
        :type data: list
        :param dataset_metadata: A dictionary of metadata specific to the
            dataset as a whole.
        :type dataset_metadata: dict
        :returns: unique handle for the dataset, i.e. its ID
        :rtype: str or int
        """
        pass

    @abstractmethod
    def update_data(
        self,
        dataset_handle: str,
        data: list[Atoms],
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Update an existing dataset - overwriting or adding new properties

        This method operates on existing configurations and/or properties. Data
        are provided as a KliFF dataset of properties that should be added to
        either the configuration as a new property or overwriting existing
        properties within the database.

        :param dataset_handle: name or ID of dataset
        :type dataset_handle: str or int
        :param data: list of ASE.Atoms objects containing the configurations
            and associated properties to add to the database. Note that
            configuration-specific metadata should be stored under the
            `atoms.info[METADATA_KEY]` field.
        :type data: list
        :param dataset_metadata: A dictionary of metadata specific to the
            dataset as a whole.
        :type dataset_metadata: dict
        :returns: unique handle for the dataset
        :rtype: str
        """
        pass

    @abstractmethod
    def get_data(
        self,
        dataset_handle: str,
        query_options: Optional[dict] = None,
    ) -> list[Atoms]:
        """
        Extract data from Storage

        Return the dataset specified by dataset_handle as a list of ASE Atoms.
        Further options for parameterizing the extraction can be provided by
        the query_options dictionary.

        :param dataset_handle: name or ID of dataset
        :type dataset_handle: str or int
        :param query_options: dict of options for data extraction and return
            |default| ``None``
        :type query_options: dict
        :returns: requested data as a list of ASE Atoms
        :rtype: list
        """
        pass

    @abstractmethod
    def delete_dataset(self, dataset_handle: str):
        """
        Remove the dataset specified by dataset_handle from the database

        :param dataset_handle: name or ID of dataset
        :type dataset_handle: str
        """
        pass

    @abstractmethod
    def list_data(self, dataset_handle: Optional[str] = None):
        """
        Utility function to query the database

        Prints an overview of the database contents if no dataset_handle is
        provided, otherwise provides information about the specific dataset
        contents.

        :param dataset_handle: name or ID of dataset |default| ``None``
        :type dataset_handle: str or int
        """
        pass
