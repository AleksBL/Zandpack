from Zandpack import __version__
latest_update = "(26.02.2026)"
MPI_ProgramName = """▂▃▄▅▆▇█▓▒░zand (v. """+str(__version__)+""")░▒▓█▇▆▅▄▃▂"""
MPI_ProgramNameNO = """▂▃▄▅▆▇█▓▒░nozand (v. """+str(__version__)+""")░▒▓█▇▆▅▄▃▂"""
psi0_name ="""▂▃▄▅▆▇█▓▒░psinought (v. """+str(__version__)+""")░▒▓█▇▆▅▄▃▂"""
scf_name  = """▂▃▅▆▇█ S █ C █ F █(v. """+str(__version__)+""")█▇▆▅▃▂"""

ProgramName = """Zandpack"""
ZP_BIBTEX   = "@article{zandpackref,\n\
         title={Zandpack: A General Tool for Time-dependent \n\
                Transport Simulation of Nanoelectronics},\n\
         author={Lorentzen, Aleksander Bach and Croy, Alexander and \n\
                 Jauho, Antti-Pekka and Brandbyge, Mads},\n\
         journal={Computer Physics Communications},\n\
         pages={110087},\n\
         year={2026},\n\
         publisher={Elsevier}\
  }"

CiteString  = "Please cite this article:\n       "+ZP_BIBTEX+"  \n"
WebPage     = "Github.com/AleksBL/Zandpack"

CoolLine    = "▂▃▅▆▇█ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █▇▆▅▃▂\n"
DocString_MPI_implementationNO = \
  """

  """+MPI_ProgramNameNO+"""
  (This is an improved version propagating the density matrix in the nonorthogonal basis.
  PLEASE CHECK WITH THE ORIGINAL ZAND EXECUTABLE TO CHECK YOUR RESULT.)
  Program developed for time-dependent transport in nanostructures
  at the Technical University of Denmark (DTU).

  """ + CiteString +"""

  The program is distributed under the MPLv2.0 licence.

  Author: Aleksander Bach Lorentzen, DTU ( aleksander.bl@proton.me )
  Supervisor: Prof. Mads Brandbyge, DTU.
  
  Contributors:
    - Dr. Nick Papior from DTU compute has constributed with performance tips.
    - Prof. Alexander Croy from Friedrich Schiller University has contributed
      with methods for consistency checks and steadystate solution. \n\n
  Visit """ + WebPage +""" for tutorials

  Basic Function of this Program:

  The program solves the time-dependent density matrix equation of motion (EOM)

      dDM(t)/dt  = -1j/hbar[H0 + dH(t), DM(t)] - dissipator(t,DM(t)) + current terms
    
       - H0: Static Hamiltonian
       - dH(t): Time-dependent part
       - dissipator(t,DM(t)): Optional term for physical or numerical damping
       - current term: Particle flux from electrodes
  
  The bias function controls the applied bias of the electrodes.
  For example, an electromagnetic pulse can be defined as:
      V(t) = E(t)*d
      bias(t,0) = V(t)/2
      bias(t,1) = -V(t)/2

  and have dH model the potential drop over the device as a linear ramp.
  The dissipator term should be set to zero unless you have some
  physical model for what you put in here. 
  The dissipator term is not particle-conserving and can be used for:
  - Physical models (e.g., electron-phonon coupling, needs advanced modelling)
  - Numerical damping (e.g., dissipator(t, DM) = eta*(DM - DM_eq))

  The particle flux into the device is given by:
     dN/dt = Tr[dissipator(t, DM)] + (∑_α J_α)
  Setting the dissipator to zero is recommended for standard transport calculations.

  Recommended course use of program:
      0. Determine equillibrum density matrix with SCF program
      1. Solve for steady state psi the psinought program
      2. Propagate a bit to see/confirm if the system is sufficiently steadystate.
      3. Do the calculation you want to do using the steady-state as initial state.

  Template files for this is located at
      """ + __file__[:-13]+ """boilerplatecode

  Experimental and general Pulseforms can be imported from
      """+__file__[:-13]+"""Pulses.py
  as
      from TimedependentTransport.Pulses import air_photonics_pulse, generic_pulse.

  Helper tools for getting
      Initial steady-state density matrix (SCF tool),
      Steady-state transport (Adiabatic tool, transport mode)
      Instantaneos quasiparticles, (Adiabatic tool, quasiparticle mode)
      Initial file manipulation    (split_k, add_spin_component tools)
  are located at
     """+ __file__[:-13]+ """cmdtools.
  You can always write
      toolname --help
  to get a list of the keywords you can specify.

  Look into the k0nfig file of the Zandpack folder to
  change some of the methods used, and printed below.
  To change the Runge-Kutta method, specify environment variable
      RK_method="RK45", "RK78", "DOP54","cash-karp"

  There are two scripts that needs to accompany this program:

  The "Initial.py" script, which has the information of the
  initial and final times, error tolerance, and how often the script
  should save the device state and some other periferal things.
  (Templastes in the Zandpack directory  'boilerplatecode')

  The "Bias.py" should contain a function called bias that returns a
  number and has arguments bias = bias(t,a), where t is time and a
  is the lead index.
  It should also contain a function called dH, which returns an array
  with the shape of the Hamiltonian (i.e. (nk, no, no)). Its
  arguments are dH = dH(t,sigma), where t is time and sigma is the
  density matrix (same shape as H).
  (Templastes in the Zandpack directory  'boilerplatecode')
  Lastly it should contain a function called "dissipator".

  You can of course further link the Bias and Initial files to
  more libraries. Nothing is stopping you from writing your own Hartree-Fock
  or DFT codes to use in the dH function.

"""

DocString_MPI_implementation = \
  """
  
  """+MPI_ProgramName+"""
  
  Program developed for time-dependent transport in nanostructures 
  at the Technical University of Denmark (DTU).
  
  """ + CiteString +"""

  The program is distributed under the MPLv2.0 licence.

Author: Aleksander Bach Lorentzen, DTU ( aleksander.bl@proton.me )
  Supervisor: Prof. Mads Brandbyge, DTU.
    - Dr. Nick Papior from DTU compute has constributed with performance tips.
    - Prof. Alexander Croy from Friedrich Schiller University has contributed
      with methods for consistency checks and steadystate solution. \n\n
  Visit """ + WebPage +""" for tutorials.
  
  Basic function of this program:

  You have control over the density matrix EOM on the form

      dDM(t)/dt  = -1j/hbar[H0 + dH(t), DM(t)] - dissipator(t,DM(t)) + current term

  and the bias function, which controls the applied bias of the electrodes.
  In a simple example of an electromagnetic pulse, you would define
      V(t) = E(t)*d, bias_0(t,0) = V(t)/2,  bias(t,1) = -V(t)/2

  and have dH model the potential drop over the device as a linear ramp.
  The dissipator term should be set to zero unless you have some
  physical model for what you put in here. Other uses are to provide a
  dampening effect away from equillibrum if it for example is introduced as

      dissipator(t, DM) = eta*(DM - DM_eq)

  Please note that the dissipator is not particle-conserving in general and
  will remove / add electrons as if it was an electrode, depending on how
  the term is constructed.

  The particle flux into the device will be
      dN/dt = Tr[dissipator(t, DM)] + ( \\sum_\\alpha J_\\alpha )
  i.e. the dissipator will introduce a loss of electrons. Setting the dissipator
  to zero is the safe option for a normal transport calculation.

  Recommended course use of program:
      0. Determine equillibrum density matrix with SCF program
      1. Solve for steady state psi the psinought program
      2. Propagate a bit to see/confirm if the system is sufficiently steadystate.
      3. Do the calculation you want to do using the steady-state as initial state.

  Template files for this is located at
      """ + __file__[:-13]+ """boilerplatecode

  Experimental and general Pulseforms can be imported from
      """+__file__[:-13]+"""Pulses.py
  as
      from TimedependentTransport.Pulses import air_photonics_pulse, generic_pulse.

  Helper tools for getting
      Initial steady-state density matrix (SCF tool),
      Steady-state transport (Adiabatic tool, transport mode)
      Instantaneos quasiparticles, (Adiabatic tool, quasiparticle mode)
      Initial file manipulation    (split_k, add_spin_component tools)
  are located at
     """+ __file__[:-13]+ """cmdtools.
  You can always write
      toolname --help
  to get a list of the keywords you can specify.

  Look into the k0nfig file of the Zandpack folder to
  change some of the methods used, and printed below.
  To change the Runge-Kutta method, specify environment variable
      RK_method="RK45", "RK78", "DOP54","cash-karp"

  There are two scripts that needs to accompany this program:

  The "Initial.py" script, which has the information of the 
  initial and final times, error tolerance, and how often the script 
  should save the device state and some other periferal things.
  (Templastes in the Zandpack directory  'boilerplatecode')

  The "Bias.py" should contain a function called bias that returns a 
  number and has arguments bias = bias(t,a), where t is time and a 
  is the lead index.
  It should also contain a function called dH, which returns an array 
  with the shape of the Hamiltonian (i.e. (nk, no, no)). Its 
  arguments are dH = dH(t,sigma), where t is time and sigma is the 
  density matrix (same shape as H).
  (Templastes in the Zandpack directory  'boilerplatecode')
  Lastly it should contain a function called "dissipator".

  You can of course further link the Bias and Initial files to
  more libraries. Nothing is stopping you from writing your own Hartree-Fock
  or DFT codes to use in the dH function.

"""


psi0_docstring = \
    """
    
    """+ psi0_name +"""
    
    Author: Aleksander Bach Lorentzen, DTU ( abalo@dtu.dk / aleksander.bl@proton.me )
    Supervisor: Mads Brandbyge, DTU.
      - Nick Papior from DTU compute has constributed with performance tips.
      - Alexander Croy from Friedrich Schiller University has contributed 
        with methods for consistency checks and steadystate solution. \n\n
    """ + CiteString +"""

    """
scf_docstring = \
    """
    """ + scf_name + """
    
    (self-consistent field solver)
    Implements the TranSIESTA method with adaptive bias window integration.
    An orthogonal basis is assumed and is implemented in largely in NumPy.
        See Papior, Nick, et al. (2017) and Refs. therein
    """
