GroH
====

A calculation of a hydrogen atom adsorbed on a graphene sheet. A surrounding
self-energy (``Real_space_SE`` from the ``siesta_python`` class) models the
infinite graphene sheet accurately. The hydrogen atom is modelled with one
orbital, and this orbital should exhibit two different spin states depending on
the tip bias: with no bias (``delta_0 = delta_1 = 0``) the system should be
unpolarised, while with ``delta_1 = 0.5`` and the Hubbard term enabled in the
Hamiltonian the system should polarise.

.. note::

   This example requires the ``add_spin_component`` tool to duplicate the system
   before the Hubbard ``U`` is added.

*Location:* ``Zandpack/examples/GroH``

.. dropdown:: ``make_device.py``

   .. literalinclude:: ../../examples/GroH/make_device.py
      :language: python

.. dropdown:: ``make_fit.py``

   .. literalinclude:: ../../examples/GroH/make_fit.py
      :language: python
