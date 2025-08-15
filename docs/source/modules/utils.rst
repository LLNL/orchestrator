Miscellaneous Utilities
=======================

See :ref:`util_api` for full API documentation. Beyond the utilities previously covered in this section, there are a number of other utility files which ensure smooth and convenient usage of the Orchestrator. We provide an overview of these components below.

Module Setup
------------

Functions provided in the ``setup_input`` file wrap around builders to directly instantiate and return a module instance, with basic input validation. :meth:`~orchestrator.utils.setup_input.setup_orch_modules` can be used to instantiate multiple modules at once from a given json file.

Templates
---------

Some workflows require the generation of input files which follow a specific structure and may share many similarities. In these cases, the :class:`~orchestrator.utils.templates.Templates` class and its :meth:`~orchestrator.utils.templates.Templates.replace` funciton can be used.

Data Standard
-------------

Orchestrator uses ASE Atoms objects as its internal configuration data structure. Key data is stored in the Atoms arrays and info dict, but ASE does not enforce any naming convention. To ensure consistency throughout the Orchestrator code suite, we define constant :ref:`data_keys` which should be used for both setting and accessing relevant data. Keys exist for quantities such as energies and forces, metadata, and selection/weight masks.

New structure IO
----------------

While Orchestrator will generally handle parsing of its own data, there are instances (typically around new data ingestion) where external data needs to be used. For converting xyz files into the internal Atoms representation, the method :meth:`orchestrator.utils.input_output.ase_glob_read` is available. Note that this method will not set keys according to the data standard (discussed above) since the xyz file format does not enforce any naming conventions. Thus for users adding data for storage, you will need to manually the keys as appropriate prior to storage with i.e. :meth:`~orchestrator.storage.storage_base.Storage.new_dataset`. A basic example is shown below where the xyz file stores forces as 'force' and energies as 'Energy':

.. code-block:: python

    from orchestrator.utils.input_output import ase_glob_read
    from orchestrator.utils.data_standard import ENERGY_KEY, FORCES_KEY

    new_configs = ase_glob_read('./path/to/xyzfiles')
    for config in new_configs:
        config.set_array(FORCES_KEY, config.arrays['force'])
        config.info[ENERGY_KEY] = config.info['Energy']
        # optionally delete the previous named entries
        del config.arrays['force']
        del config.info['Energy']
