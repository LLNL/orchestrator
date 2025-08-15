.. _kim_setup:

Using KIM Software
------------------

The Orchestrator uses several software packages developed by the
`OpenKIM project <https://openkim.org>`_, namely the `KIM API <https://openkim.org/kim-api/>`_,
kimpy_, and KliFF_.

The KIM API is the package enabling a universal interface between any compliant
interatomic model (one of the nearly 700 models available on
`openkim.org <https://openkim.org>`_, or user-created local models) and any
compliant simulator software, such as `LAMMPS <https://www.lammps.org>`_, ASE_,
and `many others <https://openkim.org/projects-using-kim/>`_.

KliFF (KIM-based Learning-Integrated Fitting Framework for interatomic potentials)
is what it says on the tin. The fitted potentials can be directly exported as
KIM-compliant models.

kimpy is a Python wrapper to the KIM API required by Pyhon software using the API,
such as KliFF and ASE.

In most cases, simply installing the ``kliff`` conda-forge package is sufficient to
get the required software, as kimpy and the KIM API are dependencies of it:

.. code-block:: console

   conda install -c conda-forge kliff

On some machines this may not be possible, in which case you should build the
packages from source. For detailed instructions, see
`OpenKIM's docs <https://openkim.org/doc/usage/obtaining-models/#installing_models>`_
and the KliFF_ and kimpy_ READMEs.

There are two types of KIM Models (`more info here <https://openkim.org/doc/repository/kim-content/>`_):

* Portable Models, which are compatible with any KIM-compliant simulator. In the vast majority of cases, the implementation is contained in a separate object called a Model Driver, while the Portable Model itself only contains the parametrization.
* Simulator Models, which are encapsulations of a set of simulator-specific instructions. Other than a small handful of exceptions, a user of LAMMPS or ASE need not distinguish between Portable and Simulator models in any way.

The current :ref:`test suite <testing>` uses two Model Drivers (one for a Stillinger
Weber potential, another for a neural network) and one Portable Model (Stillinger
Weber):

* `SW__MD_335816936951_005 <https://openkim.org/id/SW__MD_335816936951_005>`_
* `DUNN__MD_292677547454_000 <https://openkim.org/id/DUNN__MD_292677547454_000>`_
* `SW_StillingerWeber_1985_Si__MO_405512056662_006 <https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_006>`_

The simplest way to install these is to use the ``kim-api-collections-management`` utility
packaged with the KIM API to automatically install and download. For example, the following commands will install
all the required items (the SW model driver will be automatically installed with the model):

.. code-block:: console

   kim-api-collections-management install user DUNN__MD_292677547454_000
   kim-api-collections-management install user SW_StillingerWeber_1985_Si__MO_405512056662_006

Here, ``user`` is the *collection* where the items will be installed, in this case the user's local collection. For more
information, use ``kim-api-collections-management --help``. You can browse all the models in the OpenKIM repository
`here <https://openkim.org/browse/models/alphabetical>`_. Note that the drivers needed for specific models can be found listed under
"DOWNLOAD DEPENDENCY" at the end of the webpage of a KIM model in the OpenKIM database.

If you are on a machine where ``wget`` is not available, you can also install the items from a local directory. For example, to
install the SW Model Driver (which, if installing from a local directory, must be installed separately from the model), navigate
to its page linked above, click on ``Files`` in the top right to be taken to the bottom of the page, and download the ``.txz`` or
``.zip`` archive. Decompress the downloaded file, and run

.. code-block:: console

   kim-api-collections-management install user SW__MD_335816936951_005

The models and drivers are installed! Note that the models and drivers will now
be available "globally" to the user. If you don't want this global
availability, you can instead choose to install only in the present directory
by changing the ``user`` argumnet to ``CWD``. If you create a model with KLIFF,
the model name is changed to the directory where the model is saved. For
instance the :meth:`~orchestrator.trainer.kliff.kliff.KLIFFTrainer.save_model` method
of :class:`~orchestrator.trainer.kliff.kliff.KLIFFTrainer` saves the potential as
``kim_potential`` in the local directory.

Additionally, potentials can be installed into the KIM_API as simulator-models, which do not use
model-drivers and instead rely upon the simulator (e.g. LAMMPS) to implement the interactions between
particles with specific parameters read in from the potential's parameter files. For more information on
simulator-models see: https://openkim.org/doc/repository/kim-content/

More information on installing models and drivers can be found in OpenKIM's
`docs <https://openkim.org/doc/usage/obtaining-models/#installing_models>`_.
Also note that the drivers needed for specific models can be found listed under
"DOWNLOAD DEPENDENCY" at the end of the webpage of a KIM model in the OpenKIM
database.

The :class:`~orchestrator.target_property.kimrun.KIMRun` module requires `Singularity <https://apptainer.org/user-docs/3.8/>`_ to be installed.
It is commonly already installed on HPC clusters. You must build a Singularity image from the `KIM Developer Platform <https://github.com/openkim/developer-platform>`_ Docker image:

.. code-block:: console

   singularity build developer-platform.sif docker://ghcr.io/openkim/developer-platform:latest-minimal

Then, you should pass the path to the resulting ``developer-platform.sif`` as the initialization argument ``image_path`` to the :class:`~orchestrator.target_property.kimrun.KIMRun` object.

.. to do

In order to save files with kimkit, you must first add yourself as a
user. To do this, run the following two lines of code in python::

    from kimkit import users
    users.add_self_as_user('Your Name')

You will only need to do this once. Additionally, if you are saving a
model which *does not* currently have a driver saved in kimkit, you
will need to manually add that driver. Follow the example in
:meth:`~orchestrator.test.unit_testers.potential_kimkit_integration_test`

.. _ASE: https://wiki.fysik.dtu.dk/ase/
.. _kimpy: https://github.com/openkim/kimpy
.. _KliFF: https://github.com/openkim/kliff
