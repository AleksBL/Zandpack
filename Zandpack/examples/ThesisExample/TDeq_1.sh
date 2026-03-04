# Equilibrium state (SCF-Psi^0)
D=$PWD
File=TDZGNR
#1a: We make the contour first
make_ts_contour Dir=$D E1=-100.0 \
   N_C=100 N_F=15 name=tscontour pp_path=../../pp
#1b: Remove old calculation
rm -rf TDZGNR; rm -rf TDZGNR_save
#1c: Modify terms in eq. %\textcolor{green}{\eqref{eq:FermiExp}}%
modify_occupations Dir=$PWD  N_F=20 eigtol=1e-4\
   file=TDZGNR_src outfile=TDZGNR > mod.out
#2: Determine the equilibrium RDM
SCF Dir=$D file=$File Contour=tscontour.npy \
   Nonequilibrium=True > scf.out
#3: Then we deterimine the equilibrium Psi^0
psinought Dir=$D file=$File > psi0.out
