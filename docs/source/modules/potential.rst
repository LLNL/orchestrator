.. _potential_rst:

Potential
=========

The primary purpose of the potential class is to instantiate interatomic
potential models. These models can be built from scratch, read from a file,
and/or actively updated via training within the orchestrator. Note that not all
potentials will work with all trainers.

Potentials can be initialized with a kim_id, a list of which atomic species the potential supports,
a model-driver needed to run the physics calculations of the potential (if required), and an optional list of
the potential's parameter files, which if not provided will be initialized based on reasonable defaults
for each potential type. Presently, KIM potentials from openkim.org, FitSNAP potentials, and DNN-Potentials
are supported by the potential module, with support being added soon for ChIMES potentials.

The abstract base class :class:`~.Potential` provides the standard interface
for all of the concrete implementations.

See the full API for the module at :ref:`potential_module`.

The KIM_API
-----------

Many potentials are supported by the KIM_API (https://kim-api.readthedocs.io/en/latest/ ),
which provides an interface between the potential files and OpenKIM-compliant
simulator packages like LAMMPS. Potentials that comply with the OpenKIM standard for interatomic models are able to be
installed into the KIM_API via :meth:`~orchestrator.potential.kim.install_potential_in_kim_api()`.
This builds the potential against its model-driver, if any, and links it with the KIM_API where it is available for use
in KIM-Compliant simulators like LAMMPS and ASE. This allows for a wide variety of potentials to be used
with a uniform interface to calculate the behavior of atomic configurations as predicted by the potential.

KIM_API supported potentials additionally implement an :meth:`~orchestrator.potential.potential_base.Potential.evaluate()`
method, which installs the potential in the KIM_API if it isn't already, initializes
the potential's ASE KIM calculator from the potential,
and uses it to compute the energy, forces, and stress of a configuration in an ASE Atoms object.

Portable Models vs. Simulator Models
------------------------------------

Potentials can be installed into the KIM_API in one of two ways; either as a portable-model,
or as a simulator-model. The difference is how the compuation of the model is implemented. In a
portable-model, the actual implementation of the model's dynamics is in an auxiallary object called
a model-driver. These model-drivers are available on openkim.org for many potential types, and allow
the portable-models themselves to essentially just be parameter files that get passed into the driver.

Simulator-models do not use model-drivers, and instead rely on KIM_API-Compliant simulators (e.g. LAMMPS)
to implement the interactions between atoms in a way representing the chosen potential class
(e.g. pair style fitsnap), and the KIM_API acts as a compatibility layer between the user and the simulator,
so that the appropriate values from the parameter files are used where required. This makes simulator models a general
interface to whatever potential types LAMMPS can support, while portable models are limited to those that already have
drivers implemented on openkim.org.

For more information about the differences between portable-models and simulator-models see
https://openkim.org/doc/repository/kim-content/

KIMkit
------

To store trained potentials, the orchestrator uses KIMKit as a storage,
management, and version-control system for interatomic potentials. To save potentials to disk so they can be
referred to again the preferred method is :meth:`~orchestrator.potential.potential_base.Potential.save_potential_files`,
which acts as a wrapper method to save potentials wherever required. With its default arguments it will save
the potential into kimkit after staging its files in a temporary directory. By passing different boolean flags
it is possible to not use a temporary directory or kimkit and simply write the potential out to a path on the
filesystem instead. If saving to kimkit, and a potential with the same kim_id already exists in the local kimkit
repository, the potential will be automatically saved as a new version of the same potential, unless the flag
fork_potential = True is passed, in which case a new kim_id will be assigned.

The main
interfaces to KIMKit are :meth:`~orchestrator.potential.potential_base.Potential.save_potential_to_kimkit`, get_potential_files(),
create_new_version_of_potential(), fork_potential(), delete_potential(), and generate_new_kim_id().

:meth:`~orchestrator.potential.potential_base.Potential.save_potential_to_kimkit` takes a potential
name, a list of parameter files, an associated model-driver, a list of
supported atomic species, a path to the potential files, and optionally
a previous name the item was referred to. This method will generate a
CMakeLists.txt file that makes the potential installable into the KIM-API,
generates its required metadata.

:meth:`~orchestrator.potential.potential_base.Potential.get_potential_files()` just takes a kim_id to return files from,
and a destination path to save them to, and an optional boolean flag of
whether to export the driver dependency of the potential.

:meth:`~orchestrator.potential.potential_base.Potential.create_new_version_of_potential()` takes an kim_id to update and a temporary
path to save files on, along with an optional dict of metadata updates,
and writes the current potential object to disk and
saves it in KIMKit as a new version of the existing potential at kim_id. Similarly :meth:`~orchestrator.potential.potential_base.Potential.fork_potential()` will create a
new version of the potential, but assign it a new kim_id and with the user that requested the fork listed as its contributor, so that users can easily extend each other's potentials.

Additionally, potential_base implements :meth:`~orchestrator.potential.potential_base.Potential.generate_new_kim_id()` which takes
a human-readable ID prefix and a kim item type and returns a pseudorandomly
generated kimcode to be used for a new item.

When importing a potential, kimkit expects a dictionary of metadata to be
passed in, with specific requirements depending on the type of item being
imported. All items require:

	* description: a short human-readable description of the item
	* kim-api-version: version number of the KIM API that the item supports
	* kim-item-type: "portable-model", "simulator-model", or "model-driver"
	* title: a title for the item
	* extended-id: the item's assigned kimcode id designation

Additionally, all models must specify "potential-type", e.g. "eam", as well
as a list of "species" that they support, e.g. ["Fe","Cr"], and
simulator-models must additionally specify the name of the "simulator"
program that they run in, e.g. "lammps", as well as the internal name
of the "simulator-potential" they use, e.g. "eam/cd".

For potentials generated by the orchestrator, much of this metadata is set
automatically with reasonable defaults based on the potential instance attributes, but can be overridden by user input.

Further information on optional and required metadata fields for each kimkit
item type can be obtained by calling
kimkit.metadata.get_metadata_template_for_item_type(item_type), which returns
a dictionary of metadata fields and their requirements for each item type
defined in kimkit. Fields may be required for a certian item type, or may be
optional. Some fields are marked "conditionally-required", meaning that if
their value isn't set, kimkit will try to set a reasonable default. For
instance, it is not requried to manually set the date, as kimkit will add
a timestamp when the item is imported.

As of 2025-08-13 the current requirements are as follows:

====================================
MODEL DRIVER
====================================
**optional**
            :content-origin: str
            :content-other-locations: str,
            :disclaimer: str,
            :doi: str,
            :executables: list, (will be automatically populated)
            :funding: dict,
            :implementer: list, UUID4,
            :license: str,
            :simulator-potential-compatibility: list,
              (what potential in the chosen simulator the driver uses)
            :source-citations: dict,   (bibtex style citation dicts)
**required**
            :contributor-id: str, UUID4, conditionally-required,
            :date: str, conditionally-required,
            :description: str,
            :developer: list, UUID4, conditionally-required,
            :domain: str, conditionally-required,
            :extended-id: str,
            :kim-api-version: str,
            :kim-item-type: str,
            :maintainer-id: str, UUID4, conditionally-required,
            :repository: str, conditionally-required,
            :title: str

====================================
PORTABLE MODEL
====================================
**optional**
            :content-origin: str,
            :content-other-locations: str,
            :disclaimer: str,
            :doi: str,
            :executables: list,       (will be automatically populated)
            :funding: dict,
            :implementer: list, UUID4,
            :license: str,
            :model-driver: str,
            :source-citations: dict,    (bibtex style citation dicts)
            :training: list,
**required**
            :contributor-id: str, UUID4, conditionally-required,
            :date: str, conditionally-required,
            :description: str,
            :developer: list, UUID4, conditionally-required,
            :domain: str, conditionally-required,
            :extended-id: str,
            :kim-api-version: str,
            :kim-item-type: str,
            :maintainer-id: str, UUID4, conditionally-required,
            :potential-type: str,
            :repository: str, conditionally-required,
            :species: list,
            :title: str

==================================
SIMULATOR MODEL
==================================
**optional**
            :content-origin: str,
            :content-other-locations: str,
            :disclaimer: str,
            :doi: str,
            :executables: list,       (will be automatically populated)
            :funding: dict,
            :implementer: list, UUID4,
            :license: str,
            :source-citations: dict,      (bibtex style citation dicts)
            :training: list},
**required**
            :contributor-id: str, UUID4, conditionally-required,
            :date: str, conditionally-required,
            :description: str,
            :developer: list, UUID4, conditionally-required,
            :domain: str, conditionally-required,
            :extended-id: str,
            :kim-api-version: str,
            :kim-item-type: str,
            :maintainer-id: str, UUID4, conditionally-required,
            :potential-type: str,
            :repository: str, conditionally-required,
            :simulator-name: str, (name of the chosen simulator program)
            :simulator-potential: str,
              (potential type name internal to chosen simulator)
            :species: list,
            :title: str


.. _specific_potential_details:

Specific Potential Details
--------------------------
FitSNAP:
In addition to the species, model_driver, and kim_api that all potentials
require, the fitsnap potential requires settings for the FitSNAP program
as its fourth input. This can be done using a FitSNAP input file as shown
in the trainer unit tests. The documentation for FitSNAP input settings is
found at https://fitsnap.github.io/Run/Run_input.html . A python dictionary
with the same hierarchy can also be used instead, and there are utility functions
to switch between the two: :meth:`~.create_fitsnap_input_file` and
:meth:`~.convert_input_file_to_dict`.

N.B. The GROUPS and data sections are generally ignored! Datasets in
orchestrator are controlled by the storage objects and IDs and
data weighting information passed to the
:meth:`~orchestrator.trainer.trainer_base.Trainer.train`
and :meth:`~orchestrator.trainer.trainer_base.Trainer.submit_train`
functions.

Currently, this should support any linear or quadratic SNAP model.
SNAP with Neural Networks should function for training when
trainer.train(...per_atom_weights=False). ACE models will likely train
but not be uploaded to kimkit correctly just yet, but this is next on
the TODO list.

Currently, orchestrator will comment out the wselfallflag and switchinnerflag
lines in .snapparam files for compatibility with older versions of
the kim model and LAMMPS. Trying to fit with those options enabled will
not perform as expected.

KIMPotential:
This loads an existing KIM potential, so its first required argument is
the KIM ID.

KliffBPPotential:
This has 5 additional required arguments: cutoff_type, cutoff,
hyperparams, norm, and neurons.


Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.potential.factory
   orchestrator.potential.kim
   orchestrator.potential.dnn
   orchestrator.potential.fitsnap
   :parts: 3
