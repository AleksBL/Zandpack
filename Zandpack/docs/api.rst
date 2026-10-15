API reference
=============

This reference is generated from the source docstrings with
``sphinx.ext.autosummary``. The module list is curated: experimental modules
that are git-ignored, and superseded or legacy modules, are deliberately left
out, so the documented set is identical on every checkout.

Every module has a page of its own, listing the classes and functions it
defines. Each entry there links to a page with the full documentation of that
object -- for a class, all of its methods, including the undocumented ones,
which still get their signature. A ``[source]`` link on an object opens the
highlighted implementation, so the code is one click away without being in the
way of the documentation.

Core time-dependent transport
-----------------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.TimedependentTransport
   Zandpack.PadeDecomp
   Zandpack.FittedSelfEnergy
   Zandpack.FittingTools

The command-line programs that drive the propagation of the time-dependent
calculations are documented with the other
:doc:`command-line tools <../cmdtoolsdoc/index>`:
:ref:`SCF <cmdtool-scf>`, :ref:`psinought <cmdtool-psinought>`,
:ref:`modify_occupations <cmdtool-modify-occupations>` and
:ref:`nozand <cmdtool-nozand>`. :ref:`What the tools depend on
<tool-dependencies>` lists, tool by tool, which of these modules and functions
each of them uses.

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
   Zandpack.func_step_forward
   Zandpack.convert
   Zandpack.DOP54
   Zandpack.td_constants
   Zandpack.mpi_funcs
   Zandpack.mpi_runge_kutta_methods
   Zandpack.mpi_splitter
   Zandpack.mpi_timer
   Zandpack.mpi_tools

Misc.
-----

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.Response
   Zandpack.Pulses
   Zandpack.equations
   Zandpack.Interpolation
   Zandpack.plot

Citation and help text
----------------------

.. autosummary::
   :toctree: api
   :nosignatures:

   Zandpack.docstrings
   Zandpack.documentation
   Zandpack.documentation_athr
