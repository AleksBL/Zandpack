import os
import numpy as np


# def write_to_file_compressed(A, dirname):
#     try:
#         os.mkdir(dirname)
#     except:
#         pass
#     os.chdir(dirname)
#     with open('Info.txt', 'w') as f:
#         f.write(INFO)
#     f.close()
#     dic = {}
#     def add(n,v):
#         dic.update({n:v})
    
#     #Save arrays
#     add('H_Ortho',        A.Hdense[:,0])
#     add('DM_Ortho',       A.sigma)
#     add('psi_shape',      A.psi_shape)
#     add('omg_shape',      A.omega_shape)
    
#     add('S^(-0.5)',       A.Ldense[:,0])
#     add('Xpp',            A.Xpp)
#     add('Xpm',            A.Xpm)
#     add('GG_P',           A.GG_P)
#     add('GL_P',           A.GL_P)
#     add('GG_M',           A.GG_M)
#     add('GL_M',           A.GL_M)
#     add('num_leads',      A.num_leads)
#     add('num_poles_fermi',A.num_poles   )
#     add('num_lorentzians',A.NumL        )
#     add('noT',            A.max_orbital_idx + 1 )
#     add('diff_GGP_GLP',   A.diff_ggp_glp)
#     add('diff_GGM_GLM',   A.diff_ggm_glm)
#     add('xi',             A.xi)
#     add('Ixi',            A.Ixi)
#     add('_Gl_Eigenvalues',A.Gl_eig)
#     add('_GpB_Eigenvalues',A.GpB_eig)
#     add('_GpC_Eigenvalues',A.GpC_eig)
    
    
#     add('Positions',      A.Device.pos_real_space)
#     add('Species',        A.Device.s)
#     add('pivot',          np.array(A.pivot))
#     add('Fermi Poles',    A.F_poles)
#     add('mu_i',           A.mu_i)
#     add('kT_i',           A.kT_i)
#     add('coeffs_fermi',   A.coeffs_fermi)
#     add('zero_tol',A._zero_tol)
    
#     for i,L in enumerate(A.fitted_lorentzians):
#         add('Centres_Lorentzian_'+str(i), L.ei)
#         add('Broadening_Lorentzian_'+str(i), L.gamma)
#     if hasattr(A, "Hamiltonian_renormalisation_correction"):
#         add('Hamiltonian_renormalisation_correction', A.Hamiltonian_renormalisation_correction)
#     np.savez_compressed('Arrays.npz',**dic)
#     os.chdir('../')


def write_to_file_compressed(A, dirname):
    try:
        os.mkdir(dirname)
    except:
        pass
    os.chdir(dirname)
    with open('Info.txt', 'w') as f:
        f.write(INFO)
    f.close()
    
    try:
        os.mkdir('Arrays')
    except:
        pass
    
    os.chdir('Arrays')
    #Save arrays
    np.save('H_Ortho',         A.Hdense[:,0])
    np.save('DM_Ortho',        A.sigma)
    np.save('psi_shape',       A.psi_shape)
    np.save('omg_shape',       A.omega_shape)
    
    np.save('S^(-0.5)',        A.Ldense[:,0])
    np.save('Xpp',             A.Xpp)
    np.save('Xpm',             A.Xpm)
    np.save('GG_P',            A.GG_P)
    np.save('GL_P',            A.GL_P)
    np.save('GG_M',            A.GG_M)
    np.save('GL_M',            A.GL_M)
    np.save('num_leads',       A.num_leads)
    np.save('num_poles_fermi', A.num_poles   )
    np.save('num_lorentzians', A.NumL        )
    np.save('noT',             A.max_orbital_idx + 1 )
    np.save('diff_GGP_GLP',    A.diff_ggp_glp)
    np.save('diff_GGM_GLM',    A.diff_ggm_glm)
    np.savez_compressed('xi',  A.xi)
    np.savez_compressed('Ixi', A.Ixi)
    np.save('_Gl_Eigenvalues', A.Gl_eig)
    np.save('_GpB_Eigenvalues',A.GpB_eig)
    np.save('_GpC_Eigenvalues',A.GpC_eig)
    np.save('Positions',       A.Device.pos_real_space)
    np.save('Species',         A.Device.s)
    np.save('pivot',           np.array(A.pivot))
    np.save('Fermi Poles',     A.F_poles)
    np.save('mu_i',            A.mu_i)
    np.save('kT_i',            A.kT_i)
    np.save('coeffs_fermi',    A.coeffs_fermi)
    np.save('zero_tol',A._zero_tol)
    
    for i,L in enumerate(A.fitted_lorentzians):
        np.save('Centres_Lorentzian_'+str(i), L.ei)
        np.save('Broadening_Lorentzian_'+str(i), L.gamma)
    if hasattr(A, "Hamiltonian_renormalisation_correction"):
        np.save('Hamiltonian_renormalisation_correction', A.Hamiltonian_renormalisation_correction[:,0])
    
    L = A.Ldense[:,0]
    for ie in range(len(A.Sig0)):
        Sig = A.Sig0[ie]
        np.save('Sig0_NO_'+str(ie), Sig)
        np.save('Sig0_'+str(ie),  L@Sig@L)
    for ie in range(len(A.Sig1)):
        Sig = A.Sig1[ie]
        np.save('Sig1_NO_'+str(ie), Sig)
        np.save('Sig1_'+str(ie), L@Sig@L)
    
    os.chdir('../')
    os.chdir('../')
    
def write_to_file(A, dirname):
    try:
        os.mkdir(dirname)
    except:
        pass
    os.chdir(dirname)
    with open('Info.txt', 'w') as f:
        f.write(INFO)
    f.close()
    
    try:
        os.mkdir('Arrays')
    except:
        pass
    
    os.chdir('Arrays')
    #Save arrays
    np.save('H_Ortho',        A.Hdense[:,0])
    np.save('DM_Ortho',       A.sigma)
    np.save('psi_shape',      A.psi_shape)
    np.save('omg_shape',      A.omega_shape)
    
    np.save('S^(-0.5)',       A.Ldense[:,0])
    np.save('Xpp',            A.Xpp)
    np.save('Xpm',            A.Xpm)
    np.save('GG_P',           A.GG_P)
    np.save('GL_P',           A.GL_P)
    np.save('GG_M',           A.GG_M)
    np.save('GL_M',           A.GL_M)
    np.save('num_leads',      A.num_leads)
    np.save('num_poles_fermi',A.num_poles   )
    np.save('num_lorentzians',A.NumL        )
    np.save('noT',            A.max_orbital_idx + 1 )
    np.save('diff_GGP_GLP',   A.diff_ggp_glp)
    np.save('diff_GGM_GLM',   A.diff_ggm_glm)
    np.save('xi',             A.xi)
    np.save('Ixi',            A.Ixi)
    np.save('_Gl_Eigenvalues',A.Gl_eig)
    np.save('_GpB_Eigenvalues',A.GpB_eig)
    np.save('_GpC_Eigenvalues',A.GpC_eig)
    np.save('Positions',      A.Device.pos_real_space)
    np.save('Species',        A.Device.s)
    np.save('pivot',          np.array(A.pivot))
    np.save('Fermi Poles',    A.F_poles)
    np.save('mu_i',           A.mu_i)
    np.save('kT_i',           A.kT_i)
    np.save('coeffs_fermi',   A.coeffs_fermi)
    np.save('zero_tol',A._zero_tol)
    
    for i,L in enumerate(A.fitted_lorentzians):
        np.save('Centres_Lorentzian_'+str(i), L.ei)
        np.save('Broadening_Lorentzian_'+str(i), L.gamma)
    if hasattr(A, "Hamiltonian_renormalisation_correction"):
        np.save('Hamiltonian_renormalisation_correction', A.Hamiltonian_renormalisation_correction[:,0])
    
    for iE in range(len(A.Sig0)):
        Sig = A.Sig0[iE]
        np.save('Sig0_NO_static_'+str(iE), Sig)
        # np.save('Sig0_NO_static_'+str(iE), Sig)
    
    for iE in range(len(A.Sig1)):
        Sig = A.Sig1[iE]
        np.save('Sig1_NO_static_'+str(iE), Sig)
    
    os.chdir('../')
    os.chdir('../')

# # Info
INFO=''+\
"This directory concontains the files needed to propagate the reduced density and auxillary modes (see Croy&Popescu paper from 2016)\n"+\
"H_Ortho : The Hamiltonian for the system in the orthogonal Lowdin basis\n"+\
"DM_Ortho: An initial density matrix in the Lowdin basis. If an initial TranSiesta calculation is the basis of the calculation, this is the \n"+\
"          density matrix originating from this TranSiesta Calculation. Else it is simply obtained by diagonalising the Hamiltonian \n"+\
"          and doing DM = U f(ei) U*. This is not in general the equilibrium density matrix. f is the fermi-dirac distribution function\n"+\
"S^(-0.5): Matrix for Lowdin Transform. Applying this transformation takes you from the nonorthogonal basis to the orthogonal. \n"+\
"Xpp     : Poles in the upper half of the complex plane. \n"+\
"Xpm     : Poles in the lower half of the complex plane. \n"+\
"GG_P    : The symbols in Croy2016 denoted by a big gamma and indexed with a \'greater\'-symbol together with a plus\n"+\
"GL_P    : The symbols in Croy2016 denoted by a big gamma and indexed with a \'lesser\'-symbol together with a plus\n"+\
"GG_M    : The symbols in Croy2016 denoted by a big gamma and indexed with a \'greater\'-symbol together with a minus\n"+\
"GL_M    : The symbols in Croy2016 denoted by a big gamma and indexed with a \'lesser\'-symbol together with a minus\n"+\
"xi      : The eigenvectors of the Fitting matrices \n"+\
"Ixi     : The dual(?) of the eigenvectors of the fitting matrices (Denoted by the big Lambda) \n"+\
"          Ixi and xi satisfies:\n"+\
"                   xi[k, a, x, c].dot(Ixi[k,a,x,c\']) = delta_cc\'\n"+\
"psi_shape: the shape of the complex-valued array of zeros which psi is to start out with\n"+\
"omg_shape: the shape of the complex-valued array of zeros which omega is to start out with\n\n\n"+\
"You'll need to supply the delta_a and H_d(t) to make your own script to propagate this\n if what supplied with this code is not sufficient.\n"
