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

Core tools
----------

SCF
~~~

Self-consistent determination of the equilibrium density matrix. Implements the
TranSIESTA method with an adaptive bias-window integration (orthogonal basis
assumed; largely NumPy). See Papior, Nick, et al. (2017) and references therein.

.. cmdhelp:: SCF

.. video:: videos/SCF.webm
   :width: 75%
   :muted:

psinought
~~~~~~~~~

Determines the auxiliary mode wavevectors ``psi`` for the steady state, without
and with bias.

.. cmdhelp:: psinought

.. video:: videos/psinought.webm
   :width: 75%
   :muted:

modify_occupations
~~~~~~~~~~~~~~~~~~

Changes the Fermi function of the leads. It is also possible to add a new lead
by specifying a Python script in the ``new_lead`` keyword, but this should be
considered an advanced feature.

.. cmdhelp:: modify_occupations

.. video:: videos/modify_occupations.webm
   :width: 75%
   :muted:

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
