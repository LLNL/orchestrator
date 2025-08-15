Introduction
============

The Orchestrator is intended to be used as an integrated software package for building, training, testing, augmenting, running, and analyzing interatomic potentials (IAPs) and their simulations. It provides a uniform API via abstract classes, enabling the use of common codes relevant to the IAP development and deployment cycle alongside novel Orchestrator functionality. The shared API also enables drop-in replacement of different concrete instances of a single abstract class, adding to the flexibility of the framework.

Beyond the module design, the code base supports multiple usage modes, either as a library for custom built scripts or through the ``Executor`` module which facilitates an application-like style.

.. note::

    The ``Executor`` module is still under active development and is not included in the 0.5 version release.

Examples for running the code are covered in the :ref:`examples section <example_intro>` and the :ref:`test suite <testing>`.

In the subsequent sections of the introduction, we cover the key steps needed to start using the Orchestrator, including instructions and tips for installation, how to generate a Jupyer kernel, and details on how to best leverage the Orchestrator's integration with KIM suite tools from the `KIM initiative <https://kim-initiative.org/>`_.

Jump to a specific topic:

.. toctree::
   :maxdepth: 1

   introduction/installation
   introduction/jupyter
   introduction/kim_api
   introduction/kimkit
   introduction/aiida_setup
