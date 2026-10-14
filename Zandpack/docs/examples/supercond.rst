SuperCond (N-SC-N)
==================

A time-dependent calculation with a superconducting device region, demonstrating
the ``N-SC-N`` tool.

.. warning::

   This is an older, lightly polished example. The device is a cluster of
   hydrogen atoms -- not where you would expect superconductivity -- and it is
   included for the *workflow*, not the physics. The level-width functions are
   **not** made positive semi-definite, and extracting the total current from
   the ``Pi`` matrices is left to the user. The useful part is the pairing-field
   script. The code also uses some deprecated ``TD_Transport`` methods, but it
   should run.

*Location:* ``Zandpack/examples/SuperCond``

.. dropdown:: ``Calc/PairingField.py``

   .. literalinclude:: ../../examples/SuperCond/Calc/PairingField.py
      :language: python

.. dropdown:: ``make_device.py``

   .. literalinclude:: ../../examples/SuperCond/make_device.py
      :language: python

.. dropdown:: ``make_fit.py``

   .. literalinclude:: ../../examples/SuperCond/make_fit.py
      :language: python
