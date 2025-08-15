.. _kliff_trainer.rst:

Trainer Manifest for KLIFF
==========================

KLIFF uses YAML configuration files for training interatomic force fields with machine learning models. The configuration file consists of several key sections:

1. ``workspace``: Manages where training runs are stored and defines random seeds for reproducibility.
   Example:

.. code-block:: yaml

        workspace:
            name: test_run
            seed: 12345
            resume: False

2. ``dataset``: Configures how the training data is loaded, specifying dataset type (ASE, file paths, etc.), shuffling, and property keys.
   Example:

.. code-block:: yaml

       dataset:
           type: ase
           path: Si.xyz
           shuffle: True
           keys:
             energy: Energy
             forces: forces

3. ``model``: Defines the model backend (e.g., KIM or Torch) and its properties such as path, name, and input arguments.
   Example (Torch Model):

.. code-block:: yaml

        model:
            path: ./model_dnn.pt
            name: "TorchDNN"

4. ``transforms``: Modifies data or model parameters before or during training (e.g., parameter transformations or graph construction).
   Example:

.. code-block:: yaml

        transforms:
            parameter:
             - A
             - B
             - sigma:
                 transform_name: LogParameterTransform
                 value: 2.0
                 bounds: [[1.0, 10.0]]

5. ``training``: Controls the training loop, including loss function, optimizer, learning rate, dataset splitting, and hyperparameters like batch size and epochs.
   Example:

.. code-block:: yaml

        training:
           loss:
             function: MSE
             weights:
                energy: 1.0
                forces: 1.0
           optimizer:
             name: Adam
             learning_rate: 1.e-3
           batch_size: 2
           epochs: 20
           log_per_atom_pred: True

6. ``export (Optional)``: Exports the trained model for external usage, such as creating a KIM-API model.
   Example:

.. code-block:: yaml

        export:
           generate_tarball: True
           model_path: ./
           model_name: SW_StillingerWeber_trained_1985_Si__MO_405512056662_006

Example: Training a KIM Potential
---------------------------------
1. ``Dataset Setup``: Download training data.

.. code-block:: bash

     wget https://raw.githubusercontent.com/openkim/kliff/main/examples/Si_training_set_4_configs.tar.gz

2. ``Configuration``: Define workspace, dataset, model, and training settings.

.. code-block::

        workspace = {
            "name": "SW_train_example",
            "random_seed": 12345
        }
        dataset = {
            "type": "path",
            "path": "Si_training_set_4_configs",
            "shuffle": True
        }
        model = {
            "name": "SW_StillingerWeber_1985_Si__MO_405512056662_006"
        }
        transforms = {
            "parameter": ["A", "B", "sigma"]
        }
        training = {
            "loss": {
                "function" : "MSE",
                "weights": "weights.yaml" # per atom weight
            },
            "optimizer": {
                "name": "L-BFGS-B"
            },
            "training_dataset": {
                "train_size": 3
            },
            "validation_dataset": {
                "val_size": 1
            },
            "epoch" : 10,
            "log_per_atom_pred": True, # log per atom predictions
            "verbose": True
        }
        export = {
            "model_path": "./",
            "model_name": "MySW__MO_111111111111_000"
        }
        training_manifest = {
            "workspace": workspace,
            "model": model,
            "dataset": dataset,
            "transforms": transforms,
            "training": training,
            "export": export
        }

3. ``Train``: Pass configuration to trainer and begin training.

.. code-block:: python

     from kliff.trainer.kim_trainer import KIMTrainer
     trainer = KIMTrainer(training_manifest)
     trainer.train()
     trainer.save_kim_model()

This manifests the YAML configuration for KLIFF's training process, defining key sections
and settings to ensure a smooth model training experience.

Weights
=======

In the above example, the ``weights.yaml`` (extension of file should be ``yaml`` and not
``yml`` ) file is used to define the weights for each atom in the training set.
The weights are defined in a YAML file as follows:

.. code-block:: yaml

    - config: 1.0
      forces: [0.59918768, ...]
      energy: 1.0

    - config: 10.0
      forces: [0.97496481, ...]
      energy: 0.01

    - ...

Here each entry corresponds to a configuration in the dataset. Any missing item from the
yaml file is assumed to be 0.0 or ``None``. The weights are used to scale the loss function during training, allowing for more
or less emphasis on certain configurations or properties. You can also provide weights as a dictionary or datafile.

Per-atom predictions logging
============================
If the training manifest contains the ``log_per_atom_pred`` key, the trainer will log per-atom
predictions during training (currently only forces). This is useful for analyzing the model's
performance or uncertainty at the atomic level. The logged predictions can be found in the
``workspace`` directory, under the current run directory, as an ``lmdb`` file. The file name
will be ``per_atom_pred_database.lmdb``, and the properties are logged with key
``epoch_{i}|index_{j}``, where ``i`` is the epoch number and ``j`` is the index of the
configuration in the dataset. You need the ``lmdb`` library installed to enable this
functionality.

For more details, refer to the `KLIFF documentation <https://kliff.readthedocs.io/en/latest/index.html>`_

Default artifacts
=================

Below is the list of default artifacts and files that KLIFF may generate during the training. Most
of these can be named as per the user requirements. The detailed keywords are provided in the KLIFF
API documentation.

+-------------------------------+--------------------------------------------------------------+
| File / Folder                 | Description                                                  |
+===============================+==============================================================+
| ``kliff.log``                 | KLIFF’s own file logs, produced in the current working       |
|                               | directory (CWD)                                              |
+-------------------------------+--------------------------------------------------------------+
| ``fingerprints.pkl``          | Descriptors generated by the legacy descriptor module        |
+-------------------------------+--------------------------------------------------------------+
| ``finger...mean_and_std.pkl`` | Normalized descriptors generated by the legacy               |
|                               | descriptor module                                            |
+-------------------------------+--------------------------------------------------------------+
| ``final_model.pkl``           | Trained, serialized machine-learning model                   |
+-------------------------------+--------------------------------------------------------------+
| ``optimizer_state.pkl``       | Optimizer state for restarting                               |
+-------------------------------+--------------------------------------------------------------+
| ``orig_model.pkl``            | Original model serialization used by the UQ module           |
+-------------------------------+--------------------------------------------------------------+
| ``kliff_saved_model``         | Checkpoints and saved models                                 |
+-------------------------------+--------------------------------------------------------------+
