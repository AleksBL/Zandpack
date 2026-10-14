generic_wrapper
===============

A generic script that demonstrates how to use the ``wrapper`` class to do step 4
of a Zandpack calculation. Steps 1, 2 and 3 are assumed to already have been
done; the previous result is read from the ``init_file`` variable.

.. note::

   This is the **preferred way to use the command-line tools**: rather than
   running ``zand`` / ``nozand`` (or the ``cmdtools``) by hand, drive them from
   Python through the ``Zandpack.wrapper`` class.

*Location:* ``Zandpack/examples/generic_wrapper``

.. dropdown:: ``wrapper_script.py``

   .. literalinclude:: ../../examples/generic_wrapper/wrapper_script.py
      :language: python
