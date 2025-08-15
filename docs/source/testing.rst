.. _testing:

Testing
=======

Module "Unit" Tests
-------------------

A comprehensive set of tests is provided in ``orchestrator/test/`` to ensure code consistency during development. These module tests cover basic use cases for individual modules, aiming for robust code coverage of required methods. The |simulator| and |oracle| tests, among others, can be run independently, while the |trainer| and |potential| tests are run together.

Although a fully automated test suite is planned for the future, modules can currently be tested manually. To facilitate this, a setup script ``setup_tests.zsh`` is provided, which prepares the test directories for execution. Run the script as follows::

   $ zsh setup_tests.zsh /path/where/tests_are_run SELECTION

- The first argument specifies the directory where the tests will be copied and executed.
- The second argument is either ``all`` (to run all tests) or the name of a specific module (e.g., ``descriptor``, ``oracle``, etc.).

Once set up, you can run individual tests by executing the associated ``driver.py`` file::

   [.../test_path/oracle/]$ python driver.py

Tests must be run in an environment that includes all code dependencies (as well as the KIM API, if testing KIM functionality). See the :ref:`installation` section for more details.

A ``run_all.zsh`` script is also provided to automatically run the full test suite (or the loaded subset). This script checks whether a given module has already been tested and skips rerunning completed tests.

Automatic checking with ``pytest`` is also available. See
:ref:`below <pytest_check>` for more information.

Trainer Test
^^^^^^^^^^^^

The |trainer| and |potential| tests cover the generation and training of an IAP using a small set of nine structures. The trained potential is then saved.

Currently, tests are provided for the following implementations:

* :class:`~.DUNNTrainer` with :class:`~.KliffBPPotential`
* :class:`~.ParametricModelTrainer` with :class:`~.KIMPotential`
* :class:`~.FitSnapTrainer` with :class:`~.FitSnapPotential`
* Tests are included for both command-line training and job queue submission.
* Integration tests are also provided for ensuring the KIM API and kimkit backend functionality is working as expected.

Oracle Test
^^^^^^^^^^^

The |oracle| test uses three structures to compute ground-truth forces, energies, and stresses (except for :class:`~orchestrator.oracle.kim.KIMOracle`, which does not compute stresses). Outputs are saved as extended XYZ files using ASE's write function. Most tests also integrate with :class:`~orchestrator.storage.colabfit.ColabfitStorage` to test saving and extraction of configurations.

Tested modules:

* :class:`~orchestrator.oracle.espresso.EspressoOracle`
* :class:`~orchestrator.oracle.kim.KIMOracle`
* :class:`~orchestrator.oracle.lammps.LAMMPSKIMOracle`
* :class:`~orchestrator.oracle.lammps.LAMMPSSnapOracle`
* :class:`~.AiidaVaspOracle`
* :class:`~.AiidaEspressoOracle`

Workflow Test
^^^^^^^^^^^^^

Several tests use different workflow implementations, such as:

* :class:`~orchestrator.workflow.slurm.SlurmWF`
* :class:`~orchestrator.workflow.lsf.LSFWF`
* :class:`~orchestrator.workflow.slurm_to_lsf.SlurmtoLSFWF`
* :class:`~orchestrator.workflow.aiida.AiidaWF`
* :class:`~orchestrator.workflow.local.LocalWF`

Refer to the ``workflow`` block in the input files for usage examples.

Simulator Test
^^^^^^^^^^^^^^

The |simulator| test generates a molecular dynamics simulation using a randomly selected initial configuration. Output is parsed using the :meth:`~orchestrator.simulator.simulator_base.Simulator.parse_for_storage` method, with excerpts saved for verification.

Tested modules:

* :class:`~orchestrator.simulator.lammps.LAMMPSSimulator`

Target Property Test
^^^^^^^^^^^^^^^^^^^^

The |target| tests run copper :class:`~orchestrator.target_property.melting_point.MeltingPoint` calculations at standard conditions. Example files are also provided for carbon (diamond) melting point calculations and copper at high pressure, using both regular and KIM potentials.

:class:`~orchestrator.target_property.elastic_constants.ElasticConstants` tests are provided for Si and Ta, using different types of KIM potentials. This module currently supports only KIM potentials.

:class:`~orchestrator.target_property.kimrun.KIMRun` has two tests for
W, confirming correct predictions for some basic material properties using
a KIM Simulator Model and KIM Tests found in KIMKit. The two tests invoke
the Singularity container using two different workflows --
:class:`~orchestrator.workflow.local.LocalWF` and
:class:`~orchestrator.workflow.slurm.SlurmWF`. An additional two tests,
both demonstrating the calculation of a cold curve for diamond Si using
Stillinger-Weber using :class:`~.LocalWF`, test the usage of :class:`~.KIMRun`
with :class:`~.potential_base.Potential`. :class:`~.KIMRun` works with
:class:`~.potential_base.Potential` objects whose
:meth:`~.Potential.save_potential_files` function saves an archive of a
directory that is installable using the KIM API.

.. note::

   The KIMRun test requires the following OpenKIM items and dependencies to be downloaded and added to KIMKit (separate from the KIM API). See the :ref:`KIMRun documentation <kimrun>` for instructions:

   * `Sim_LAMMPS_MEAM_Lenosky_2017_W__SM_631352869360_000 <https://openkim.org/id/Sim_LAMMPS_MEAM_Lenosky_2017_W__SM_631352869360_000>`_
   * `ElasticConstantsCubic_bcc_W__TE_866278965431_006 <https://openkim.org/id/ElasticConstantsCubic_bcc_W__TE_866278965431_006>`_
   * `CohesiveEnergyVsLatticeConstant_diamond_Si__TE_973027833948_004 <https://openkim.org/id/CohesiveEnergyVsLatticeConstant_diamond_Si__TE_973027833948_004>`_

Descriptor Test
^^^^^^^^^^^^^^^

The |descriptor| tests computes descriptors in both single and batched execution modes.

Tested modules:

* :class:`~.KLIFFDescriptor` module to compute ACSF descriptors
* :class:`~.QUESTSDescriptor` module to compute QUESTS descriptors

Score Test
^^^^^^^^^^

The |score| tests demonstrate the use of both :class:`~.AtomCenteredScore` and :class:`~.DatasetScore` modules to compute various score quantities. Tests include both single and batched calculations, as well as saving/accessing data with :class:`~.ColabfitStorage`.

Tests cover:

* :class:`~.LTAUForcesUQScore`
* :class:`~.QUESTSEfficiencyScore`
* :class:`~.QUESTSDiversityScore`
* :class:`~.QUESTSDeltaEntropyScore`
* :class:`~.FIMTrainingSetScore`
* :class:`~.FIMPropertyScore`
* :class:`~.FIMMatchingScore`

.. note::

    All FIM tests currently only support integration with :class:`~.KIMPotential`.



.. _pytest_check:

Semi-Automated Checks with pytest
---------------------------------

Curated outputs for unit tests are saved in the ``reference_output/`` directory and can be automatically checked against new test runs using ``pytest`` with the provided test files in each module's subdirectory. pytest compares new outputs to the reference set.

pytest is automatically run for any test completed via the ``driver.py`` file. You can also run pytest manually - it will search for any files that begin
with ``test_`` in the current directory and all
subdirectories and run these tests.::

   $ pytest -v

The ``-v`` flag increases verbosity, showing which tests are run. To run a specific test file simply specify it::

   $ pytest test_simulator.py


Adding Tests
------------

To add tests for new functionality or to increase coverage:

#. Check if existing unit tests apply to your use case. This will generally be the case if you are adding a new concrete module to a pre-existing module type. If not, write a new test and add it to the appropriate ``MODULE_unit_testers.py`` file.
#. Write the input file for your test and add it to the appropriate ``test_inputs`` directory. If additional inputs are needed, add them here as well. Input files should be machine-agnostic. If machine/user specific inputs are needed, they should be entered as ``<ABSTRACT_VALUES>`` which can be substituted by the ``setup_tests.zsh`` script.
#. Add adequate reference data to confirm successful test completion. Place this data in the appropriate ``reference_data/`` subdirectory, using distinguishable filenames. Keep files small where possible. Shared reference data across tests is allowed.
#. Update the pytest ``test_MODULE.py`` file to check your test output against your reference data.
#. Update the relevant ``driver.py`` file to run your new test condition, specifying any dependencies.
#. Update ``setup_tests.zsh`` to properly "install" the test module on different machines. Replace any machine-specific input as needed based on architecture. Refer to current tests and the setup script for examples.
#. Run ``setup_tests.zsh`` to load your new tests and ensure they run and pass their pytests.


.. |workflow| replace:: :class:`~orchestrator.workflow.workflow_base.Workflow`
.. |oracle| replace:: :class:`~orchestrator.oracle.oracle_base.Oracle`
.. |score| replace:: :class:`~orchestrator.computer.score.score_base.ScoreBase`
.. |simulator| replace::
     :class:`~orchestrator.simulator.simulator_base.Simulator`
.. |trainer| replace:: :class:`~orchestrator.trainer.trainer_base.Trainer`
.. |potential| replace::
     :class:`~orchestrator.potential.potential_base.Potential`
.. |target| replace::
     :class:`~orchestrator.target_property.property_base.TargetProperty`
.. |descriptor| replace::
     :class:`~orchestrator.computer.descriptor.descriptor_base.DescriptorBase`
