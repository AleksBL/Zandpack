from Zandpack import __version__

MPI_ProgramName = """▂▃▄▅▆▇█▓▒░zand (v. """+str(__version__)+""")░▒▓█▇▆▅▄▃▂"""
psi0_name ="""▂▃▄▅▆▇█▓▒░psinought (v. """+str(__version__)+""")░▒▓█▇▆▅▄▃▂"""
scf_name  = """▂▃▅▆▇█ S █ C █ F █(v. """+str(__version__)+""")█▇▆▅▃▂"""

ProgramName     = """Zandpack"""
CiteString = "Please cite this article:\n       ***Cool Article Bibtex***  \n"
WebPage    = "Github.com/AleksBL/??"

CoolLine = "▂▃▅▆▇█ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █ - █▇▆▅▃▂\n"
DocString_MPI_implementation = \
  """
  
  """+MPI_ProgramName+"""
  
  Program developed for time-dependent transport in nanostructures 
  at the Technical University of Denmark (DTU).
  
  """ + CiteString +"""
  
  Author: Aleksander Bach Lorentzen, DTU ( aleksander.bach@dipc.org / aleksander.bl.mail@gmail.com )
  Supervisor: Prof. Mads Brandbyge, DTU.
    - Dr. Nick Papior from DTU compute has constributed with performance tips.
    - Dr. Alexander Croy from Friedrich Schiller University has contributed 
      with methods for consistency checks and steadystate solution. \n\n
  Visit """ + WebPage +""" for tutorials.
  
  Basic function of this program:
  There are two scripts that needs to accompany this program:
 
  The "Initial.py" script, which has the information of the 
  initial and final times, error tolerance, and how often the script 
  should save the device state and some other periferal things.

  The "Bias.py" should contain a function called bias that returns a 
  number and has arguments bias = bias(t,a), where t is time and a 
  is the lead index.
  It should also contain a function called dH, which returns an array 
  with the shape of the Hamiltonian (i.e. (nk, no, no)). Its 
  arguments are dH = dH(t,sigma), where t is time and sigma is the 
  density matrix (same shape as H).
  Lastly it should contain a function called "dissipator".
  
  You have control over the density matrix EOM on the form
  
      dDM(t)/dt  = -1j/hbar[H0 + dH(t), DM(t)] - dissipator(t,DM(t)) + current term
  
  and the bias function, which controls the applied bias of the electrodes. 
  In a simple example of an electromagnetic pulse, you would define 
  
     V(t) = E(t)*d, bias_0(t,0) = V(t)/2,  bias(t,1) = -V(t)/2
  
  and have dH model the potential drop over the device as a linear ramp. 
  You could then furthermore include a finite particle lifetime 
  in the device if you know the  equillibrium density matrix for the 
  different voltages you apply. This would be on the form
      dissipator(t, DM) = eta*(DM - DM_eq)
  Please note that the dissipator is not particle-conserving in general.
  The particle flux into the device will be
      dN/dt = Tr[dissipator(t, DM)] + ( \sum_\alpha J_\alpha )
  i.e. the dissipator will introduce a loss of electrons.
  
  You can of course further link the Bias and Initial files to
  more libraries. Nothing is stopping you from writing your own Hartree-Fock 
  or DFT codes to use in the dH function.
  
  Recommended course use of program:
      0. Determine equillibrum density matrix with SCF program
      1. Solve for steady state psi the psinought program
      2. Propagate a bit to see if the system is sufficiently steadystate.
      3. Do the calculation you want to do using the steady-state as initial state.
  
  The default ODE-solver is the RKF45 method, which samples t0 
  and five intermediate steps t + h[i] before deciding whenever or not
  to step fourth the system or make the stepsize smaller. Therefore,
  if you want to utilise more involved dH functions, you can been track
  of the density matrix by saving every sixth DM being input to the dH 
  function.
  
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
  It is recommended to use the SCF tool to obtain the equillibrium 
  density matrix. The default density matrix in the *TDT_file* directory
  is either a direct subbing on a transiesta DM or the isolated system
  DM = Uf(E_i)U^\dagger.
  
  Look into the k0nfig file of the TimedependentTransport module to
  change some of the methods used, and printed below.
  
"""


psi0_docstring = \
    """
    
    """+ psi0_name +"""
    
    Author: Aleksander Bach Lorentzen, DTU ( abalo@dtu.dk / aleksander.bl.mail@gmail.com )
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
