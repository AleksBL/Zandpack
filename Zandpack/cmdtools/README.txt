Two tools for making the steady state current to compare to the timedependent transport code

"Adiabatic":
           Calculates instantaneous currents, DOS and Transmission, AND Quasiparticle states (Bound states)
"dos2qp_guess":           
           Makes guesses for starting the quasiparticle scheme with
"qp_grid":                
           Makes a rectangular grid of Energies and times to try start out the quasiparticle scheme at.
"WriteDM":                
           Writes a density matrix to be taken account of in the steadystate calculations if the Adiabatic depends on the density.
"Basischange":            
           Applies a unitary transformation to the original basis used in the calculation. 
"add_spin_component":     
           Takes a spindegenerate calculation and adds a spin component like this: (nk, ....) -> (2nk, ....)
"SCF":                    
           Self-consistent determination of equilibrium density matrix
"psinought":              
           Determination of wavevectors psi for steady-state.
"SCF_default_Contour.npy":  
           NumPy array with equilibrium contour for use in the SCF program.
"td_info":
           Tool for plotting and quickly accessing information in a TD-calculation.

Get help by doing "executable --help" on any of these
See tutorials for more info


