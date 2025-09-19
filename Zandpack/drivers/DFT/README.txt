This is an easily adaptable driver to link siesta to the Zand code.


DFT_v2 and DFTCalculator runs in one job (job_dft, submit file to LSF10 cluster),
while Zand runs in another job (job_td).

You can read all the communication that goes on. Signalling new DMs to calculate the
Hamiltonian is done through text files.

Relies on the HSetupOnly keyword in siesta, so make sure you have one of the newer versions
of siesta.

