from .storage_base import Storage
from ..utils.exceptions import DatasetDoesNotExistError
from ..utils.input_output import ase_glob_read, safe_write
from ..utils.data_standard import METADATA_KEY
import os
from os import system, walk, path, makedirs
from ase import Atoms
import json
from typing import Optional
from uuid import uuid4


class LocalStorage(Storage):
    """
    Class to store data in the local disk

    The Storage class deals with all functionalities associated to data storage
    inside Orchestrator. Its functions include the initialization of the
    database, and data additions, updates, and queries. The Orchestrator uses
    ASE Atoms as the internal data representation. A given database (Storage
    instance) can include multiple datasets (collections of configurations and
    properties) and generally persists in time.

    :param storage_args: dictionary with initialization parameters, including
        database_name and database_path. database_path defaults to
        './local_storage_database' while database_name defaults to the last
        string component from the database_path. LocalStorage does not require
        any additional arguments
    :type storage_args: dict
    """

    def __init__(
        self,
        database_path: Optional[str] = './local_storage_database',
        database_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Set variables and initialize the recorder

        :param database_path: Path to the local storage database |default|
            ``'./local_storage_database'``
        :type database_path: str
        :param database_name: Name of the local database
        :type databse_name: str
        """
        super().__init__(**kwargs)

        self.database_path = database_path
        # default database_name to last component of database_path if not set
        if database_name is None:
            self.database_name = self.database_path.strip('/').split('/')[-1]
        else:
            self.database_name = database_name
        self.storage_init_args = {
            "database_path": self.database_path,
            "database_name": self.database_name,
        }

        self.STORAGE_ID_KEY = 'storage_id'
        self.property_map = None

        makedirs(self.database_path, exist_ok=True)
        self.logger.info(f'Local database set as {self.database_path}')
        self._dataset_sizes = {}
        self._id_map = {}
        self._read_database_state()

    def _read_database_state(self):
        """
        Helper function to facilitate synchronizing a database

        Updates the internal _dataset_sizes dictionary with the current state
        of the database read from disk. This can be called on restarts or to
        ensure the internal representation is up to date before querying its
        state for other purposes.
        """
        root_dir_parsed = False
        for root, dirs, files in walk(self.database_path):
            if not root_dir_parsed:
                root_dir_parsed = True
                self.logger.info(f'Found {len(dirs)} datasets in {root}')
                if len(files) > 0:
                    self.logger.info(('It appears that your database has files'
                                      ' unassociated with any dataset!'))
            else:
                dataset = root.split('/')[-1]
                config_num = 0
                configs = ase_glob_read(root)
                config_num = len(configs)
                for i, config in enumerate(configs):
                    storage_id = config.info[METADATA_KEY]['storage_id']
                    config_id = i
                    _ = self._add_to_id_map(dataset, storage_id, config_id)
                self._dataset_sizes[dataset] = config_num

    def _add_to_id_map(
        self,
        dataset: str,
        storage_id: str,
        config_id: Optional[int] = None,
    ) -> int:
        """
        helper function to add to the storage --> config map, handles KeyError

        :param dataset: name of the dataset for the mapping
        :type dataset: str
        :param storage_id: UUID of the configuration
        :type storage_id: str
        :param config_id: configuration number of the config in this dataset if
            ``None``, get the config num from the current dataset_size
            |default| ``None``
        :type config_id: int
        :returns: the config number in the dataset (useful if config_id = None)
        :rtype: int
        """
        if config_id is None:
            config_id = self._dataset_sizes[dataset]
            self._dataset_sizes[dataset] += 1
        try:
            self._id_map[dataset][storage_id] = config_id
        except KeyError:
            self._id_map[dataset] = {storage_id: config_id}
        return config_id

    def check_if_dataset_name_unique(self, dataset_name: str) -> bool:
        """
        check if the provided dataset_name is unique in the database

        :param dataset_name: name to check (human readable)
        :type dataset_name: str
        :returns: true if the database is not present in the database, false if
            it does exist
        :rtype: boolean
        """
        if path.isdir(f'{self.database_path}/{dataset_name}'):
            return False
        else:
            return True

    def _insert_data_to_database(
        self,
        dataset_handle: str,
        data: list[Atoms],
        dataset_metadata: Optional[dict] = None,
    ) -> str:
        """
        Internal utility function for adding data to the database

        Called by add_data(), new_dataset(), and update_data()

        :param dataset_handle: name of dataset
        :type dataset_handle: str
        :param data: list of ASE.Atoms objects containing the configurations
            and associated properties to add to the database. Note that
            configuration-specific metadata should be stored under the
            `atoms.info[METADATA_KEY]` field.
        :type data: list
        :param dataset_metadata: A dictionary of metadata specific to the
            dataset as a whole.
        :type metadata: dict
        :returns: unique handle for the dataset
        :rtype: str
        """
        self._read_database_state()

        for configuration in data:

            if METADATA_KEY not in configuration.info:
                # this configuration is new to the dataset
                storage_id = uuid4().hex
                configuration.info[METADATA_KEY] = {'storage_id': storage_id}
                config_num = self._add_to_id_map(dataset_handle, storage_id)
            elif 'storage_id' in configuration.info[METADATA_KEY]:
                # this configuration already has an ID
                storage_id = configuration.info[METADATA_KEY]['storage_id']
                if storage_id in self._id_map[dataset_handle]:
                    # this config is in the dataset already
                    config_num = self._id_map[dataset_handle][storage_id]
                else:
                    # this config is new to the dataset
                    config_num = self._add_to_id_map(dataset_handle,
                                                     storage_id)
            else:
                # this configuration has metadata but no ID (new to dataset)
                storage_id = uuid4().hex
                configuration.info[METADATA_KEY]['storage_id'] = storage_id
                config_num = self._add_to_id_map(dataset_handle, storage_id)

            new_data_path = (f'{self.database_path}/{dataset_handle}/'
                             f'configuration_{config_num:05}.xyz')

            # overwrite current xyz file with the new configuration (and data)
            safe_write(new_data_path, configuration, format='extxyz')

        if dataset_metadata is not None:
            new_metadata_path = (f'{self.database_path}/{dataset_handle}/'
                                 f'metadata_global.json')
            with open(new_metadata_path, 'w') as fout:
                # TODO: hopefully this throws an error if not a dict?
                json.dump(dataset_metadata, fout, sort_keys=True, indent=4)

        return dataset_handle

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

        :param dataset_handle: name of dataset
        :type dataset_handle: str
        :param data: list of ASE.Atoms objects containing the configurations
            and associated properties to add to the database. Note that
            configuration-specific metadata should be stored under the
            `atoms.info[METADATA_KEY]` field.
        :type data: list
        :param dataset_metadata: A dictionary of metadata specific to the
            dataset as a whole.
        :type metadata: dict
        :returns: handle for the dataset which includes the new additions
        :rtype: str
        """
        if dataset_handle not in self._dataset_sizes:
            raise DatasetDoesNotExistError(
                (f'{dataset_handle} has not been created in '
                 f'{self.database_path}, use new_dataset()'))
        return self._insert_data_to_database(dataset_handle, data,
                                             dataset_metadata)

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
        :type metadata: dict
        :returns: name of the dataset
        :rtype: str
        """
        if path.isdir(f'{self.database_path}/{dataset_handle}'):
            self.logger.info((f'It appears dataset "{dataset_handle}" already '
                              f'exists and contains '
                              f'{self._dataset_sizes[dataset_handle]} '
                              f'configurations. Adding to it.'))
        else:
            system(
                f'mkdir -p {self.database_path}/{dataset_handle} 2> /dev/null')
            self._dataset_sizes[dataset_handle] = 0
            self._id_map[dataset_handle] = {}
        return self._insert_data_to_database(dataset_handle, data,
                                             dataset_metadata)

    def update_data(
        self,
        dataset_handle: str,
        data: list[Atoms],
        new_data_key: str,
    ) -> str:
        """
        Update an existing dataset - overwriting or adding new properties

        This method operates on existing configurations and/or properties. data
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
        :type metadata: dict
        :returns: unique handle for the dataset
        :rtype: str
        """
        configs = self.get_data(dataset_handle)
        for config in configs:
            storage_id = config.info[METADATA_KEY][self.STORAGE_ID_KEY]
            data_to_update = data[storage_id]
            if len(data_to_update) == len(config):
                config.set_array(new_data_key, data_to_update)
            else:
                config.info[METADATA_KEY][new_data_key] = data_to_update

        if dataset_handle.rsplit('_', 1)[-1][0] != 'v':
            # original dataset name, we'll append _v*
            index = 1
        else:
            index = int(dataset_handle.rsplit('_', 1)[-1][1:]) + 1
        base_name = dataset_handle.rsplit('_', 1)[0]
        new_handle = self.new_dataset(f'{base_name}_v{index}', configs)
        return new_handle

    def get_data(
        self,
        dataset_handle: str,
        query_options: Optional[dict] = None,
    ) -> list[Atoms]:
        """
        Extract data from storage

        Return the dataset specified by dataset_handle as a list of ASE Atoms.
        Further options for parameterizing the extraction can be provided by
        the query_options dictionary.

        :param dataset_handle: name of the dataset to extract
        :type dataset_handle: str
        :param query_options: dict of options for data extraction and return
            |default| ``None``
        :type query_options: dict
        :returns: requested data as a list of ASE Atoms
        :rtype: list
        """
        if query_options:
            self.logger.info('Querey options are not currently supported')
        configs = ase_glob_read(
            os.path.join(self.database_path, dataset_handle))
        # Note: it's assumed that the metadata is already in the ASE file

        return configs

    def delete_dataset(self, dataset_handle: str):
        """
        Remove the dataset specified by dataset_handle from the database

        :param dataset_handle: name or ID of dataset
        :type dataset_handle: str
        """
        system(f'rm -r {self.database_path}/{dataset_handle} 2> /dev/null')
        self.logger.info(f'Deleted dataset {dataset_handle} from storage')

    def list_data(self, dataset_handle: Optional[str] = None):
        """
        Utility function to query the database

        Prints an overview of the database contents if no dataset_handle is
        provided, otherwise provides information about the specific dataset
        contents.

        :param dataset_handle: name of dataset |default| ``None``
        :type dataset_handle: str
        """
        self._read_database_state()
        if dataset_handle is None:
            print(f'Database {self.database_name} has the following datasets:')
            for dataset in self._dataset_sizes:
                print(f'{dataset}: {self._dataset_sizes[dataset]}')
        else:
            dataset_size = self._dataset_sizes.get(dataset_handle)
            if dataset_size is None:
                print(
                    f'{self.database_name} does not contain {dataset_handle}')
            else:
                print(
                    f'{dataset_handle} contains {dataset_size} configurations')
