Simulator
=========

This module drives molecular dynamics simulations using codes such as
`LAMMPS <https://www.lammps.org/>`_. This functionality may be used to study
scientific questions, obtain additional training data, or test a potential.

As with Oracles and other Calculator-like modules, the central function of this
module is :meth:`~orchestrator.simulator.simulator_base.Simulator.run`. This function is
designed to generate a single input file by default. However, in certain scenarios, it may
be necessary to include additional input files after the run directory is created but prior to
executing the calculation. To address this need, there is an internal flag: ``external_setup``
which can be set to True. If this flag is switched on, then the Simulator will
call its :meth:`~orchestrator.simulator.simulator_base.Simulator._external_calculation_setup`
method, passing in the ``run_path``. This function will in turn call
``self.external_func(path)``, which should be set by an external module. See
``ElasticConstants``'s
:meth:`~orchestrator.target_property.elastic_constants.ElasticConstants.conduct_sim`
method for a practical example.

The abstract base class :class:`~.Simulator` provides the standard interface for all of the concrete implementations.

See the full API for the module at :ref:`simulator_module`.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.simulator.factory
   orchestrator.simulator.lammps
   :parts: 3
