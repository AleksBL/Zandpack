ThesisExample
=============

Steps 1--4 in a series of scripts on a simple carbon nanoribbon. The steady
state is initially obtained using TranSIESTA, but for the time-dependent
calculation (step 4) a much simpler ``U * sigma_ii`` term is introduced on one
site in the middle of the ribbon.

*Location:* ``Zandpack/examples/ThesisExample``

.. dropdown:: ``5a.py``

   .. literalinclude:: ../../examples/ThesisExample/5a.py
      :language: python

.. dropdown:: ``Initial.py``

   .. literalinclude:: ../../examples/ThesisExample/Initial.py
      :language: python

.. dropdown:: ``Bias.py``

   .. literalinclude:: ../../examples/ThesisExample/Bias.py
      :language: python
