Descriptor
==========

The Descriptor class encapsulates the functionality for computing and storing
representations of atomic environments and atomic configurations (supercells).

See the full API for the module at :ref:`descriptor_module`.

Descriptor Types
----------------

**AtomCenteredDescriptor**
   Represents each local atomic environment (an atom and its neighbors within a
   cutoff) with its own descriptor. Results are stored in the ``.arrays``
   dictionary of ASE Atoms objects with the key ``{OUTPUT_KEY}_descriptors``.

**ConfigurationDescriptor**
   Represents an entire supercell using a single descriptor. Results are stored
   in the ``.info`` dictionary of ASE Atoms objects with the key
   ``{OUTPUT_KEY}_descriptors``.

Available Implementations
-------------------------

**KLIFFDescriptor**
   Leverages the KLIFF library for atomic environment descriptors:

   * ``symmetry_function``: Atom Centered Symmetry Functions (ACSF)
   * ``bispectrum``: Bispectrum descriptors

   Requires cutoff distances, cutoff function name, and hyperparameters.

**QUESTSDescriptor**
   Model-agnostic descriptors from the QUESTS library. Combines sorted neighbor
   distances with average triplet bond lengths. The descriptor dimensionality is
   ``(2 * num_nearest_neighbors) - 1``.

Usage Example
-------------

.. code-block:: python

   # KLIFF symmetry function descriptor
   kliff_desc = KLIFFDescriptor(
       descriptor_type='symmetry_function',
       cut_dists={'Cu-Cu': 4.0, 'O-O': 3.5, 'Cu-O': 3.7},
       cut_name='cos',
       hyperparams='set51'
   )

   # QUESTS descriptor
   quests_desc = QUESTSDescriptor(
       num_nearest_neighbors=32,
       cutoff=5.0
   )

   # Compute descriptors
   descriptors = kliff_desc.compute(atoms)
   descriptor_list = quests_desc.compute_batch(list_of_atoms)

The descriptors are automatically attached to the atomic configurations and can
be saved to storage with metadata for ColabFit integration.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.computer.descriptor.descriptor_base
   orchestrator.computer.descriptor.kliff
   orchestrator.computer.descriptor.quests
   :parts: 3
