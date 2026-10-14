API reference
=============

This reference is generated from the source docstrings with
``sphinx.ext.autosummary``. The module list is curated: experimental modules
that are git-ignored, and superseded or legacy modules, are deliberately left
out, so the documented set is identical on every checkout.

Core time-dependent transport
-----------------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.TimedependentTransport
   Zandpack.Response
   Zandpack.Pulses
   Zandpack.PadeDecomp
   Zandpack.FittedSelfEnergy
   Zandpack.FittingTools

Drivers and control
-------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.wrapper
   Zandpack.recipes
   Zandpack.Help
   Zandpack.Loader
   Zandpack.Writer
   Zandpack.k0nfig

Numerics, utilities and MPI helpers
-----------------------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.decoupled_functions
   Zandpack.banded_matrix_funcs
   Zandpack.equations
   Zandpack.func_step_forward
   Zandpack.Interpolation
   Zandpack.convert
   Zandpack.DOP54
   Zandpack.td_constants
   Zandpack.plot
   Zandpack.mpi_funcs
   Zandpack.mpi_runge_kutta_methods
   Zandpack.mpi_splitter
   Zandpack.mpi_timer
   Zandpack.mpi_tools

Citation and help text
----------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.docstrings
   Zandpack.documentation
   Zandpack.documentation_athr
