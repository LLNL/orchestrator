Turn-key Module
---------------

This module supports the usage of Orchestrator in an "application" mode.

.. toctree::
   :maxdepth: 1

   modules/executor


Coordinating Modules
--------------------

These modules coordinate the usage of one or more atomic modules for both simple and complex objectives.

.. toctree::
   :maxdepth: 1

   modules/target_property

Atomic Modules
--------------

These modules are considered "atomic" in that they serve single functionalities and can be considered the building blocks for the rest of the Orchestrator framework.

.. toctree::
   :maxdepth: 1

   modules/potential
   modules/trainer
   modules/oracle
   modules/simulator
   modules/computer
   modules/descriptor
   modules/score
   modules/augmentor

Utility Modules
---------------

These modules coordinate key backend functionality.

.. toctree::
   :maxdepth: 1

   modules/workflow
   modules/storage
   modules/restart
   modules/factory
   modules/utils

Orchestrator Inheritance Graph
------------------------------

.. inheritance-diagram::
   orchestrator.augmentor.augmentor_base
   orchestrator.computer.descriptor.kliff
   orchestrator.computer.descriptor.quests
   orchestrator.computer.score.fim.fim_training_set
   orchestrator.computer.score.fim.fim_property
   orchestrator.computer.score.fim.fim_matching
   orchestrator.computer.score.ltau
   orchestrator.computer.score.quests
   orchestrator.oracle.aiida.espresso
   orchestrator.oracle.aiida.vasp
   orchestrator.oracle.espresso
   orchestrator.oracle.factory
   orchestrator.oracle.kim
   orchestrator.oracle.lammps
   orchestrator.oracle.aiida.espresso
   orchestrator.oracle.aiida.vasp
   orchestrator.potential.dnn
   orchestrator.potential.factory
   orchestrator.potential.fitsnap
   orchestrator.potential.kim
   orchestrator.simulator.factory
   orchestrator.simulator.lammps
   orchestrator.storage.colabfit
   orchestrator.storage.factory
   orchestrator.storage.local
   orchestrator.target_property.elastic_constants
   orchestrator.target_property.factory
   orchestrator.target_property.kimrun
   orchestrator.target_property.melting_point
   orchestrator.trainer.factory
   orchestrator.trainer.fitsnap
   orchestrator.trainer.kliff.kliff_dunn_trainer
   orchestrator.trainer.kliff.kliff_parametric_trainer
   orchestrator.utils.module_factory
   orchestrator.utils.restart
   orchestrator.utils.templates
   orchestrator.workflow.aiida
   orchestrator.workflow.factory
   orchestrator.workflow.local
   orchestrator.workflow.lsf
   orchestrator.workflow.slurm
   orchestrator.workflow.slurm_to_lsf
   :parts: 3
