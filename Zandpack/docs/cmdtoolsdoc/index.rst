cmdtools documentation
======================
This is the documentation of the othe commandline executables that comes with the Zandpack code. These tools are located in the cmdtools folder (you might have added the path to your PATH variable in the installation). These tools can be executed with the --help flag, which will print the keywords that can be specified with them. 

The basic usage of the tools is as such: 

.. code-block:: bash

   tool Dir=$PWD kw1=v1 kw2=v2 ....

The "Dir" variable is however not needed when using the --help flag. It does however specify which directory the code should have as working directory and is needed when the code is actually used.

SCF
---
Calculates the steady-state density matrix, without and with bias.

.. code-block:: bash

    SCF --help

.. video:: videos/SCF.webm
   :width: 100%
   :figwidth: 60%
   :muted: 
   

psinought
---------
Calculates the auxilliary mode wavevectors in the steady-state, without and with bias. 

.. code-block:: bash

    psinought --help

.. video:: videos/psinought.webm
   :width: 100%
   :figwidth: 60%
   :muted: 

Adiabatic
---------

Calculate properties in the steady state. Current, Transmission, DOS etc. 

.. code-block:: bash

    Adiabatic --help

.. video:: videos/Adiabatic.webm
   :width: 100%
   :figwidth: 60%
   :muted: 
    
    
modify_occupations
------------------
Allows the user to change the Fermi function of the leads. It is also possible to add a new lead by specifying a python script in the new_lead keyword, but this should be considered an advanced feature. 

.. code-block:: bash

    modify_occupations --help

.. video:: videos/modify_occupations.webm
   :width: 100%
   :figwidth: 60%
   :muted: 


td_info
-------
Tool for plotting calculated time-dependent current. 

.. code-block:: bash

    td_info --help

.. video:: videos/td_info.webm
   :width: 100%
   :figwidth: 60%
   :muted: 


N-SC-N
------
Tool for adding a making the device region superconducting. A pairing field has to be specified as a python script. (THIS SHOULD BE CONSIDERED AN ADVANCED FEATURE. YOU SHOULD PROBABLY READ THE CODE TO SEE HOW IT WORKS.)

.. code-block:: bash

    N-SC-N --help

.. video:: videos/N-SC-N.webm
   :width: 100%
   :figwidth: 60%
   :muted: 


add_spin_component
------------------

.. code-block:: bash

    add_spin_component --help

.. video:: videos/add_spin_component.webm
   :width: 100%
   :figwidth: 60%
   :muted: 



