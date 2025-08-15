.. _trainer_module:

Trainer Module
==============

Abstract Base Class
-------------------

.. automodule:: orchestrator.trainer.trainer_base
   :members:
   :undoc-members:
   :show-inheritance:

Concrete Implementations
------------------------

KLIFF base class
^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.trainer.kliff.kliff.KLIFFTrainer
   :members:
   :undoc-members:
   :show-inheritance:

DNN sub-class
^^^^^^^^^^^^^

.. autoclass:: orchestrator.trainer.kliff.kliff_dunn_trainer.DUNNTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Parametric model sub-class
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.trainer.kliff.kliff_parametric_trainer.ParametricModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:

FitSnap class
^^^^^^^^^^^^^^^^^^

.. autoclass:: orchestrator.trainer.fitsnap.FitSnapTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Trainer Builder
---------------

.. automodule:: orchestrator.trainer.factory
   :members:
   :undoc-members:
   :show-inheritance:
