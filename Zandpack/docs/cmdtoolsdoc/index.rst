cmdtools documentation
======================

This is the documentation of the command-line executables that come with the
Zandpack code. The tools live in the ``cmdtools`` folder (which you may have
added to your ``PATH`` during installation); the ``zand`` / ``nozand``
time-propagation codes live in the ``mpi`` folder. Every tool can be run with
``--help`` to print the keywords it accepts.

The basic usage is:

.. code-block:: bash

   tool Dir=$PWD kw1=v1 kw2=v2 ....

``Dir`` sets the working directory (``Dir=$PWD`` covers the common case) and is
not needed with ``--help``. Keywords are assigned as ``keyword=value``, with no
spaces.

.. note::

   The ``Zandpack.wrapper`` module provides several ways to call most of these
   command-line tools from Python -- this is the preferred way to use them. See
   :doc:`../examples/generic_wrapper`.

.. _tool-workflow:

How the tools work together
---------------------------

A calculation runs through four steps. The solid arrows are the main flow; the
dashed boxes are the tools that can be run once the stage they hang off has
finished. The frame around step 4 is the ``wrapper.py`` module, which runs those
tools for you.

.. toolflow::
   :also: Also in cmdtools, used on occasion rather than as part of this flow: zzz, WriteDM, set_time, split_k, PSD_DM, dos2qp_guess, add_spin_component, compress_xi.

   Step 1: the initial steady state, from SIESTA, TranSIESTA or DFTB
     - siesta_python | electrode band structures, pristine Hamiltonians and the density matrix of the device region

   Step 2: transport quantities, from TBtrans
     - siesta_python.run_tbtrans_in_dir | transmission function, device DOS and the self-energy matrices, written to disk and read back by TD_Transport

   Step 3: Lorentzian fits of the sampled Γ_α | N-SC-N | can also be run once the fit exists | then step 4
     - TD_Transport.PoleGuess / TD_Transport.Fit | fit the sampled Γ_α, or use the AI_code / FittingRoutine tools instead. They do the same type of job, but the FittingRoutine does it better: it is slower, and has only recently been added to the code.
     - TD_Transport.inspect_* / Writer | inspect the fitted system properties; Writer saves the instance to disk

   Step 4: the command-line tools
   wrap 4 | wrapper.py (wrapper.Control) | drives the tools of this step from Python -- write_initial(), run_scf(), run_psinought(), run_nozand() -- so you do not have to type the commands yourself.
     - modify_occupations | gives the Fermi function the poles it needs from the fit of step 3
     - SCF | self-consistent density matrix of the device region, using the fit | viljas-cuevas, pocket-tddft, Adiabatic | all three can run once SCF has produced a density matrix
     - psinought | steady-state auxiliary mode wave-vectors, starting from the SCF density matrix
     - nozand (zand is deprecated) | time propagation, using the SCF density matrix and the auxiliary modes | td_info | inspect a propagation while it runs and after it finishes
     - compress_currents / lossydm | compress the output of a finished propagation

.. _tool-dependencies:

What the tools depend on
------------------------

Each tool is a stand-alone script that imports a handful of modules from
``Zandpack``. The graph below shows which ones: a line runs from the tool on the
left to a module on the right, and the thicker it is, the more names the tool
imports from that module. Hold the pointer over a tool to fade the rest of the
graph and leave only its own dependencies visible.

For the names themselves, open a tool underneath the graph. Each one expands
into a picture of what it imports, in which every module and every name links to
its page in the API reference, so a dependency can be followed all the way to
the definition.

The occasional tools -- the ones in the last box of the workflow above, plus
``qp_grid`` -- are left out here.

.. tooldeps:: TimedependentTransport.py:TD_Transport wrapper.py:wrapper
   :skip: add_spin_component, compress_xi, dos2qp_guess, N-SC-N, PSD_DM, qp_grid, set_time, split_k, WriteDM, zzz

Two of the tools listed above are Python objects rather than scripts:
``TD_Transport`` is the class that owns a calculation (see
:class:`Zandpack.TimedependentTransport.TD_Transport`), and ``wrapper`` holds
the ``Control`` interface that runs the tools for you -- see
:doc:`../examples/generic_wrapper`.

Core tools
----------

``modify_occupations`` is normally run before ``SCF``, since it sets the
occupations that ``SCF`` converges.

.. _cmdtool-scf:

SCF
~~~

Self-consistent determination of the equilibrium density matrix. Implements the
TranSIESTA method with an adaptive bias-window integration (orthogonal basis
assumed; largely NumPy). See Papior, Nick, et al. (2017) and references therein.

.. cmdhelp:: SCF

.. video:: videos/SCF.webm
   :width: 75%
   :muted:

.. _cmdtool-psinought:

psinought
~~~~~~~~~

Determines the auxiliary mode wavevectors ``psi`` for the steady state, without
and with bias.

.. cmdhelp:: psinought

.. video:: videos/psinought.webm
   :width: 75%
   :muted:

.. _cmdtool-modify-occupations:

modify_occupations
~~~~~~~~~~~~~~~~~~

Changes the Fermi function of the leads. It is also possible to add a new lead
by specifying a Python script in the ``new_lead`` keyword, but this should be
considered an advanced feature.

.. cmdhelp:: modify_occupations

.. video:: videos/modify_occupations.webm
   :width: 75%
   :muted:

.. _cmdtool-nozand:

nozand
~~~~~~

Propagates the system in time under a time-dependent bias, working in the
non-orthogonal basis (faster, thanks to more sparsity in the matrices). This is
the preferred propagation code. Run it under MPI, for example:

.. code-block:: bash

   export OMP_NUM_THREADS=1
   export NUMBA_NUM_THREADS=1
   mpirun -np 8 nozand Dir=$PWD > nozand.out

An ``Initial.py`` and a ``Bias.py`` file must be present in the working
directory. Input files are written by ``SCF`` and ``psinought``; if these were
not run beforehand, the propagation starts from the isolated device density
matrix and zero auxiliary modes, so it takes a long time to reach the steady
state. ``nozand --help`` prints the available keywords.

.. list-table:: nozand: keywords and default values
   :header-rows: 1
   :widths: 22 78
   :class: cmdhelp-table

   * - Keyword
     - Default
   * - ``Dir``
     - (required) absolute path to the working directory, e.g. ``Dir=$PWD``
   * - ``initial_file``
     - ``Initial.py``
   * - ``bias_file``
     - ``Bias.py``
   * - ``out_dir``
     - ``None``
   * - ``sigma_start``
     - ``None``
   * - ``psi_start``
     - ``None``
   * - ``omg_start``
     - ``None``

viljas-cuevas
~~~~~~~~~~~~~

Minimal Floquet photo-assisted current, following Viljas and Cuevas (2007). It
uses the same Zandpack files as the ``nozand`` / ``zand`` programs and returns
the current in units of ``e/fs`` (energies in eV).

.. cmdhelp:: viljas-cuevas

zand
~~~~

**Superseded by ``nozand``.** The orthogonal-basis variant of the propagation
code, invoked the same way (``Dir=...``); prefer ``nozand``.

Archiving tools
---------------

A time-dependent calculation with ``nozand`` (or ``zand``) writes its density
matrices and currents to disk in chunks as the calculation progresses, rather
than keeping everything in memory. A long run therefore leaves a directory full
of small files behind. The tools in this group collect those chunks into one
compressed file, so a finished calculation can be stored and transferred
cheaply.

.. note::

   The best way to archive a calculation is to use the ``wrapper.py`` module,
   :mod:`Zandpack.wrapper`, rather than driving the tools below by hand.
   :func:`Zandpack.wrapper.archive_calculation` copies the calculation folder to
   the archive name, deletes the intermediate density matrices, currents and
   ``nozand`` output, and leaves an ``ARCHIVE_COMPLETE.txt`` marker so a
   finished archive is recognisable. The command-line tools below are the
   lower-level building blocks it builds on.

lossydm
~~~~~~~

Collects the chunks of the time-dependent density matrix produced by ``zand`` /
``nozand`` into a single file and compresses it, using either the built-in
``numpy.savez_compressed`` or the ``xz`` tool. This is the lossy counterpart to
``compress_currents``.

.. cmdhelp:: lossydm

compress_currents
~~~~~~~~~~~~~~~~~

Collects the chunks of the time-dependent currents produced by ``zand`` /
``nozand`` into one file and compresses it. Unlike ``lossydm``, the currents are
stored losslessly: every point the calculation saved is kept.

.. cmdhelp:: compress_currents

Other tools
-----------

Analysis, post-processing and utility tools.

Adiabatic
~~~~~~~~~

Calculates steady-state properties: instantaneous currents, DOS, transmission
and quasiparticle states.

.. cmdhelp:: Adiabatic

.. video:: videos/Adiabatic.webm
   :width: 75%
   :muted:

td_info
~~~~~~~

Plots and quickly inspects the results of a time-dependent calculation.

.. cmdhelp:: td_info

.. video:: videos/td_info.webm
   :width: 75%
   :muted:

qp_grid
~~~~~~~

Makes a rectangular grid of energies and times to start the quasiparticle scheme
from.

.. cmdhelp:: qp_grid

N-SC-N
~~~~~~

Makes the device region superconducting. A pairing field has to be specified as
a Python script. (Advanced feature -- you should probably read the code to see
how it works.)

.. cmdhelp:: N-SC-N

.. video:: videos/N-SC-N.webm
   :width: 75%
   :muted:

add_spin_component
~~~~~~~~~~~~~~~~~~

Takes a spin-degenerate calculation and adds a spin component, i.e.
``(nk, ...) -> (2nk, ...)``.

.. cmdhelp:: add_spin_component

.. video:: videos/add_spin_component.webm
   :width: 75%
   :muted:

zzz
~~~

Compression and easy transfer of files for the Zandpack code, for example:

.. code-block:: bash

   zzz Dir=$PWD folder=tmp files=* command="rsync -a tmp.tar.xz user@host:/dir/" delete_all=True
   zzz Dir=$PWD folder=tmp.tar.xz mode=decompress

Further tools
-------------

These tools are also present in ``cmdtools`` but are not documented in detail
here; run them with ``--help`` or read the source:

* ``compress_xi`` -- compress the auxiliary-mode data from a calculation.
* ``dos2qp_guess`` -- starting guesses for the quasiparticle scheme.
* ``PSD_DM`` -- write a positive semi-definite density matrix.
* ``WriteDM`` -- write a density matrix to be used by the steady-state
  calculations.
* ``set_time``, ``split_k``, ``pocket-tddft`` -- assorted utilities.
