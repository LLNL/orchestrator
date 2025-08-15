Workflow
========

This module handles the submission and retrieval of simulations to either a
local computer or HPC resources, managing the file structure of the simulations,
and retains information on the location of job files.

To see a list of currently implemented job schedulers, see the full API for the
module at :ref:`workflow_module`. The abstract base class
:class:`~orchestrator.workflow.workflow_base.Workflow` provides the standard
interface for all of the concrete implementations. We also provide an abstract
base class for HPC schedulers:
:class:`~orchestrator.workflow.workflow_base.HPCWorkflow`

The simplest implementation provides an interface with the local command line,
but interface with job schedulers or other more sophisticated tools, such as
`Merlin <https://merlin.readthedocs.io/en/latest/>`_ is also possible.

Use Cases
---------

:class:`~orchestrator.workflow.local.LocalWF`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation for running jobs locally on a personal computer or an interactive job session
:class:`~orchestrator.workflow.local.LocalWF`.  Note that
all of the modules define a default workflow which is used if a workflow is
needed but not supplied. This default is an instance of
:class:`~orchestrator.workflow.local.LocalWF` with the root directory set to
the module's name.

:class:`~orchestrator.workflow.slurm.SlurmWF`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A default script template for the slurm batch file is provided, but the user
can define their own and provide it's path via the ``default_template``
keyword in the ``workflow_args`` dictionary passed to the Workflow constructor.
Also note that if synchronous (blocking) behavior is desired, this can be
toggled with the ``synchronous`` keyword in the ``job_details`` dict provided
to :meth:`~orchestrator.workflow.workflow_base.Workflow.submit_job`.

The ``job_details`` dict also hosts any modifications to the batch job desired,
with the default batch template defining all possible options:

.. code-block:: bash

   #!/bin/bash
   #SBATCH -N <NODES>
   #SBATCH -p <QUEUE>
   #SBATCH -A <ACCOUNT>
   #SBATCH -t <WALLTIME>
   <EXTRA_HEADER>

   <PREAMBLE>

   <COMMAND>

   <POSTAMBLE>

In addition to these keywords (which should be set as lowercase, i.e.
'preamble'), default queue, account, walltime, and node parameters can be set.
Lastly, the frequency of calls to squeue are set by ``wait_freq``, which has a
default of 60 seconds.

The workflow is designed to have flexibility for heterogenous use cases.
To this end, default parameters can be set by the user when constructing the
Workflow via the ``workflow_args`` dict, but many of these parameters can be
overridden for any specific job by providing them in the ``job_details`` dict
of the :meth:`~orchestrator.workflow.workflow_base.Workflow.submit_job`
function.

When using an asynchronous workflow, it is important to use a blocking function
to ensure necessary calculations are done before proceding. An example is
:class:`~orchestrator.workflow.slurm.SlurmWF`'s
:meth:`~orchestrator.workflow.slrum.SlurmWF.block_until_completed` method,
which would be called right before the outcomes of any set of calculations
are needed by subsequent functions or modules.

:class:`~orchestrator.workflow.lsf.LSFWF`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~orchestrator.workflow.lsf.LSFWF` is provided as a mirror to
:class:`~orchestrator.workflow.slurm.SlurmWF` that enables the use of
IBM's LSF scheduler. Much of the previous description applies to this
scheduler as well. The differences will be highlighted below.

:class:`~orchestrator.workflow.slurm_to_lsf.SlurmtoLSFWF`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moreover, :class:`~orchestrator.workflow.slurm_to_lsf.SlurmtoLSFWF` is
provided as a mirror to :class:`~orchestrator.workflow.lsf.LSFWF` that
enables submitting jobs on a LSF machine while running Orchestrator on a
Slurm machine. To use this functionality, the preamble needs to be set in the
``job_details`` dict to do the necessary exports and sourcing for kim_api on
a LSF machine (see ``setup_tests.zsh`` for the details), so that LAMMPS can be
used on the LSF machine without activating Orchestrator.

:class:`~orchestrator.workflow.aiida.AiidaWF`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An interface for the AiiDA framework has been implemented as a Workflow for
the Orchestrator. This must be combined with any of the oracles found in
:ref:`aiida_oracle` API documentation. As
:class:`~orchestrator.workflow.aiida.AiidaWF` inherits from
:class:`~orchestrator.workflow.workflow_base.HPCWorkflow`, all of the variables
related to job submission are the same. These values can be seen at the
:class:`~orchestrator.workflow.workflow_base.HPCWorkflow` API documentation.

Slurm and LSF Differences
-------------------------

While Slurm and LSF perform the same function, there are subtle differences in
keyword selection and use cases. The LLNL LC reference pages for
`Slurm <https://hpc.llnl.gov/banks-jobs/running-jobs/slurm-user-manual>`_ and
`LSF <https://hpc.llnl.gov/banks-jobs/running-jobs/lsf-user-manual>`_ are good
places to start for details on these schedulers. Differences in flags used for
specifying the jobs can also be found `in the chart here
<https://hpc.llnl.gov/banks-jobs/running-jobs/\\
slurm-srun-versus-ibm-csm-jsrun>`_.

Full documentation for `Slurm sbatch <https://slurm.schedmd.com/sbatch.html>`_
and `LSF bsub <https://www.ibm.com/docs/en/spectrum-lsf/\\
10.1.0?topic=bsub-options>`_ can be found at the provided links.

Development Plan
----------------

As use cases for the Orchestrator are fleshed out, more complex workflows can
be developed. These may interface with tools such as
`Maestro <https://github.com/LLNL/maestrowf>`_ and/or
`Merlin <https://merlin.readthedocs.io/en/latest/>`_, or other software
entirely.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.workflow.factory
   orchestrator.workflow.local
   orchestrator.workflow.slurm
   orchestrator.workflow.lsf
   orchestrator.workflow.slurm_to_lsf
   orchestrator.workflow.aiida
   :parts: 3
