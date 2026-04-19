NOTE OF CAUTION: Only useful part here is how to start using the N-SC-N tool I would say
This is an older example, which has not been polished very thoroughly, but contains the files for using the N-SC-N tool.
There are a couple of things that are wrong here though:
   - The device is a collection hydrogen atoms in the shape of a molecule, probably not where you would expect to see any superconductivity. This system is probably also very unphysical?
   - The levelwidth function is not made positive semi-definite, a somewhat significant lacking.
   - You might need to manually do the current extraction from the current matrices (Pi matrices), meaning youll have to find out which parts of the current matrix actually contributes to the total current
     in a N-SC-N setup like this.
Whats right:
   - The pairing field script (PairingField.py) is present in the Calc directory. This script is needed for the N-SC-N tool
     to make the superconducting Hamiltonian.
   - Gives a runable example with the right ingredients that demonstrates how to do a time-dependent calculation with a superconducting
     device region.
   - You can build on and introduce a density matrix dependence from here.

Another note of caution, the code also uses some depricated methods of the TD_Transport class, but it should be runable.
