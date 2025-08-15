.. _jupyter:

Integrating Orchestrator with Jupyter Notebooks
===============================================

To use Orchestrator with Jupyter notebooks, you need a Jupyter kernel that
matches your Python environment. The steps below guide you through setting up
and configuring the kernel.

Creating the Kernel
-------------------

1. Begin by activating the Orchestrator environment. You can do this using the
   ``source_me.sh`` script created during installation (see
   :ref:`script_install`.)

2. Next, create the Jupyter kernel by running the following command
   (``ipykernel`` is installed with the Orchestrator):

   .. code-block:: console

        $ python -m ipykernel install --prefix=$HOME/.local/ --name 'orchestrator_kernel' --display-name 'Orchestrator kernel'

   Here, the ``--name`` flag sets the internal name of the kernel, while
   ``--display-name`` determines how the kernel appears in the Jupyter UI when
   selecting kernels.

3. Now, set the required environment variables for your kernel. Using the
   example above, locate the configuration file at
   ``.local/share/jupyter/kernels/orchestrator_kernel/kernel.json``.
   Add the following entries to the JSON file under an ``env`` block:

   .. code-block::

        "env": {
            "PATH": "YOUR_PATH_VARIABLE",
            "LD_LIBRARY_PATH": "YOUR_LD_LIBRARY_PATH_VARIABLE",
            "PYTHONPATH": "YOUR_PYTHONPATH_VARIABLE",
            "CMAKE_PREFIX_PATH": "YOUR_CMAKE_PREFIX_PATH_VARIABLE",
            "CC": "YOUR_CC_VARIABLE",
            "CXX": "YOUR_CXX_VARIABLE",
            "FC": "YOUR_FC_VARIABLE",
            "F90": "YOUR_F90_VARIABLE",
            "KIM_API_PORTABLE_MODELS_DIR": "YOUR_KIM_API_PORTABLE_MODELS_DIR_VARIABLE",
            "KIM_API_SIMULATOR_MODELS_DIR": "YOUR_KIM_API_SIMULATOR_MODELS_DIR_VARIABLE",
            "KIM_API_MODEL_DRIVERS_DIR": "YOUR_KIM_API_MODEL_DRIVERS_DIR_VARIABLE"
        }

   Replace each ``YOUR_X_VARIABLE`` with the output of ``echo $X``, where ``X``
   is the relevant environment variable name.

   .. note::
      Your environment setup script (``source_me.sh``) should set these
      variables automatically. For consistency, it is safest to copy and paste
      the output of ``echo $X`` directly into the ``env`` block.

   .. note::
      When adding the ``env`` block to the JSON file, ensure proper comma
      placement. If you add the block at the end of the file, place a comma
      after the previous block. Otherwise, add a comma at the end of the
      ``env`` block.

Using the Kernel in a Notebook
------------------------------

Once the kernel is set up, you can select it in Jupyter notebooks. For new
notebooks, choose the kernel from the "New" dropdown menu. For existing
notebooks, switch kernels by navigating to Kernel -> Change kernel and then
selecting the appropriate kernel name.

Example Notebooks
-----------------

To see practical examples of Orchestrator usage, refer to the
:ref:`Examples section<example_intro>`. Running these notebooks will require
the usage of your newly created kernel!
