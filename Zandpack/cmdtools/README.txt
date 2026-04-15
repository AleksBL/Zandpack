The Zandpack.wrapper module provides several ways to call most of these commandline tools from python.

Get help by doing "executable --help" on any of these.

"Adiabatic":
   Calculates instantaneous currents, DOS and Transmission, and Quasiparticle states
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
"td_info":
   Tool for plotting and quickly accessing information in a TD-calculation.

See examples and notebooks for more info.
