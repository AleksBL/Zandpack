Examples
========

Zandpack ships with a collection of example calculations under
``Zandpack/examples``. This section indexes them and, where useful, shows the
main input scripts inline so you can inspect the workflows without leaving the
documentation.

.. note::

   These examples are provided as-is. Most will run, but a few were written
   against older versions of Zandpack and may need small modifications. The
   newer notebook tutorials use refined methods that older scripts could also
   benefit from.

The typical Zandpack workflow is a four-step pipeline:

1. ``make_device.py`` -- build the device geometry and steady-state setup.
2. ``1xxx`` / ``2Load`` -- obtain the steady-state density matrix and fit the
   level-width functions (the auxiliary modes).
3. ``3Fit`` / ``make_fit.py`` -- fit the self-energies.
4. Propagate in time (``zand`` / ``nozand``, or an ``Initial.py`` + ``Bias.py``
   pair in a ``Calculate``/``Calc`` directory).

Tight-binding tutorials
-----------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Notebook tutorials
      :link: notebook_tutorials
      :link-type: doc

      The up-to-date, step-by-step tutorials (T1--T7).

Tight-binding systems
---------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: TB_ribbon
      :link: tb_ribbon
      :link-type: doc

      A nanoribbon bridge between graphene electrodes.

   .. grid-item-card:: GrRibGr
      :link: grribgr
      :link-type: doc

      A graphene nanoribbon between two graphene sheets.

   .. grid-item-card:: small_bound
      :link: small_bound
      :link-type: doc

      Small reference for the full device / fit / propagate pipeline.

   .. grid-item-card:: ZGNR+N
      :link: zgnr_n
      :link-type: doc

      A zigzag graphene nanoribbon.

   .. grid-item-card:: AGNR+Tip
      :link: agnr_tip
      :link-type: doc

      Armchair graphene nanoribbon with a scanning tip.

   .. grid-item-card:: Hubbard
      :link: hubbard
      :link-type: doc

      Mean-field Hubbard on a zigzag nanoribbon with a defect.

SIESTA / TranSIESTA
-------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: ThesisExample
      :link: thesis_example
      :link-type: doc

      Steps 1--4 on a simple carbon nanoribbon.

   .. grid-item-card:: AuBreakJunction
      :link: au_break_junction
      :link-type: doc

      An Au break junction with a linearized SIESTA Hamiltonian.

   .. grid-item-card:: GroH
      :link: groh
      :link-type: doc

      Hydrogen on graphene with a real-space self-energy.

   .. grid-item-card:: C60
      :link: c60
      :link-type: doc

      A C60 molecular junction.

DFTB+
-----

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Pohl
      :link: pohl
      :link-type: doc

      A DFTB+ based calculation on a molecular junction.

   .. grid-item-card:: Pohl2
      :link: pohl2
      :link-type: doc

      An unfinished calculation derived from Pohl.

Advanced features
-----------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: SuperCond (N-SC-N)
      :link: supercond
      :link-type: doc

      A time-dependent calculation with a superconducting device region.

   .. grid-item-card:: generic_wrapper
      :link: generic_wrapper
      :link-type: doc

      Using the ``wrapper`` class for step 4.

   .. grid-item-card:: MoS2 spin-orbit coupling
      :link: mos2_soc
      :link-type: doc

      Spin-orbit coupling in MoS2 via the ``SiP`` class.

   .. grid-item-card:: Graphite surface
      :link: graphite_surface
      :link-type: doc

      A surface calculation on a bulk electrode.

Reproducibility and checks
--------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Reproduce_and_checks
      :link: reproduce_and_checks
      :link-type: doc

      Scripts used to verify the steps of the code.

   .. grid-item-card:: Smoke tests
      :link: smoke_tests
      :link-type: doc

      Minimal device / fit setups.

.. toctree::
   :hidden:

   notebook_tutorials
   tb_ribbon
   grribgr
   small_bound
   zgnr_n
   agnr_tip
   hubbard
   thesis_example
   au_break_junction
   groh
   c60
   pohl
   pohl2
   supercond
   generic_wrapper
   mos2_soc
   graphite_surface
   reproduce_and_checks
   smoke_tests
