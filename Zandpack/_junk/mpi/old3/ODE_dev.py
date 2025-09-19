import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import os
import sys
from TimedependentTransport.TimedependentTransport import PI as PI_a
from TimedependentTransport.TimedependentTransport import Jk
from TimedependentTransport.mpi_RK4pars_dev import A, B, C, CT, CH, RK_N
from TimedependentTransport.mpi_splitter import partition, get_sources
from TimedependentTransport.mpi_funcs import MM, idxholder, TERR, step_fourth, OuterSubtraction, OuterSubtractionAssign
import TimedependentTransport.k0nfig as config
from TimedependentTransport.docstrings import DocString_MPI_implementation as docstring
from time import time, sleep


# Initial Navigation to directory given
args = sys.argv[1:]
original_folder=os.getcwd()
ChDir = False
Check = ['Dir' in v for v in args]
if sum(Check)>1:
    print('Dont specify several directories. Killing processes')
    print('\n\n\n')
    assert 1 == 0
if True in Check:
    i = Check.index(True)
    if "=" not in args[i]:
        print('Specify path like this: ')
        print('Dir = /path/to/somewhere/ ')
        print('Killing processes')
        print('\n\n\n')
        
        assert 1 == 0
    
    path = args[0].replace(' ', '').split('=')[1]
    os.chdir(path)
    new_folder = os.getcwd()
    sys.path.append(new_folder)
    

from Initial import Arrs as D
from Initial import t0, t1, stepsize, print_timings, usesave,saveevery, name, eps, hbar
from Bias import dH, bias


#MPI stardard import
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Shorthands
nax       = np.newaxis
_multiply = np.multiply
# Commandline args
# More options in the future
args = sys.argv[1:]
options_dic = {'saveplot':False,
               }

if len(args)>0:
    if 'saveplot' in args:
        options_dic['saveplot'] = True

#Readins from saved folder, imported from "Initial"
NK = D['H_Ortho'].shape[0]
NO = D['H_Ortho'].shape[-1]
NL = D['psi_shape'][1]
NX = D['psi_shape'][2]

# Bookkeeping for sending and recieving data
# This is all handled in the mpi_splitter module
# Here we just give some tags for the various processes
# for when we have to send data back and fourth
src = get_sources(size)

tag_count = 10
PI_tags = []
PI_idx  = []
for v in src[0]:
    PI_tags   += [tag_count]
    PI_idx    += [partition(v, size, NK, NL, NX)[1:]]
    tag_count += 1

PSI_tags = []
PSI_idx  = []
for v in src[1]:
    PSI_tags  += [tag_count]
    PSI_idx   += [partition(v, size, NK, NL, NX)[1:]]
    tag_count += 1
OMG_tags = []
OMG_idx  = []
for v in src[2]:
    OMG_tags  += [tag_count]
    OMG_idx   += [partition(v, size, NK, NL, NX)[1:]]
    tag_count += 1

ROLE_INFO = partition(rank, size, NK, NL, NX)
ROLE = ROLE_INFO[0]

#print('Process ', rank, ' has role', ROLE)
# Initial Info
def start_print():
    print(docstring)
    print('\nMASTER PROCESS STARTLED\n')
    print('Running on ' + str(size) + ' nodes')
    print('Using Tolerance: '+str(eps))
    print('Using Folder: ' + str(name))
    print('Using PI version:', str(config.PI_VERSION))
    print('Using Parallel PI:', str(config.PARALLEL_PI))
    print('Using Parallel outer subtraction:', str(config.NUMBA_OUTER_SUBTRACTION_PARALLEL))
    print('Lazy Omega: '+str(config.MPI_LAZY_OMEGA))
    print('OMP_NUM_THREADS: ' + str(os.environ['OMP_NUM_THREADS']))
    print('Folder Path: '+os.getcwd())
    print('Propagation algorithm: adaptive Runge-Kutta-Fehlberg (4/5 Order)')
    print('NumPy Config:')
    #np.show_config()
    print('\n')
    try:
        print('NUMBA_NUM_THREADS: ' + str(os.environ['OMP_NUM_THREADS']))
    except:
        print('NUMBA_NUM_THREADS: UNKNOWN')
    
    print('Args: ' + str(args) )
    print('Source-scheme:\n')
    print('Pi calculated on nodes:    '+ str(src[0]))
    print('Psi calculated on nodes:   '+ str(src[1]))
    if config.MPI_LAZY_OMEGA:
        print('Part of omega calculated on nodes: '+ str(src[2]))
    else:
        print('Omega calculated on nodes: '+ str(src[2]))
    print('\n')
    print('Options:')
    for i in options_dic.keys():
        print('    '+str(i)+':  ' +str(options_dic[i]))
    print("\n\nStart of Run!")
    

def ODE_SOLVER(stepsize=stepsize, T0=t0, T1=t1):
    sleep(2)
    start_print()
    # Stepsize updater
    # eps: error-tolerance
    # TE : calculated error
    def hnew(h, TE): return 0.9 * h * (eps / TE) ** 0.2
    #shorthand
    DT = np.complex128
    # Common Variables
    # Baseline Hamiltonian
    H0      = D['H_Ortho']
    # Ht is for communicating H(t) between nodes
    Ht      = np.zeros(H0.shape, dtype=DT)
    # Possibility to restart calculation, we load saved state
    if usesave == False:
        sig0    = D['DM_Ortho'].copy()
        psi0    = np.zeros(D['psi_shape']).astype(DT)
        omg0    = np.zeros(D['omg_shape']).astype(DT)
    else:
        sig0    = np.load(name+'_save/'+'last_sig.npy')
        psi0    = np.load(name+'_save/'+'last_psi.npy')
        omg0    = np.load(name+'_save/'+'last_omg.npy')
        t0      = float(np.load(name+'_save/'+'last_time.npy'))
        h       = float(np.load(name+'_save/'+'last_stepsize.npy'))
    
    #Arrays to place values of various intermediate samplings in the RK4 method
    tmp_sig = D['DM_Ortho'].copy()
    tmp_psi = np.zeros(D['psi_shape'], dtype=DT)
    tmp_omg = np.zeros(D['omg_shape'], dtype=DT)
    tmp_PI  = np.zeros((NK, NL, NO, NO), dtype = DT)
    
    # first six indices meant for RK sampling between 0 and 1, while 7 is
    # the initial step at t_loc = 0
    SIGS = np.zeros((RK_N+1,)+sig0.shape, dtype=DT)
    PSIS = np.zeros((RK_N+1,)+psi0.shape, dtype=DT)
    OMGS = np.zeros((RK_N+1,)+omg0.shape, dtype=DT)
    SIGS[-1] = sig0
    PSIS[-1] = psi0
    OMGS[-1] = omg0
    
    _nleads, noT =omg0.shape[1], omg0.shape[3]
    _Xpp = D['Xpp'][:,:,:,:omg0.shape[3]]
    _Xpm = D['Xpm']
    delta = np.zeros(_nleads)
    # Everything takes place in SIGS, PSIS, and OMGS, so we delete these now
    del sig0, psi0, omg0
    
    # some assignments of values comming from the Initial file
    num_elecs = tmp_psi.shape[1]
    if usesave==False:
        t0 = T0
        h  = stepsize
    
    t1 = T1
    
    # Assemblers
    PI_buff = []
    for v in src[0]:
        _Rinf = partition(v, size, NK, NL, NX)
        shape = (len(_Rinf[1]), len(_Rinf[2]), NO, NO)
        PI_buff += [np.zeros(shape, dtype=DT)]
    
    PSI_buff = []
    for v in src[1]:
        _Rinf = partition(v, size, NK, NL, NX)
        shape = (len(_Rinf[1]), len(_Rinf[2]), len(_Rinf[3]), D['psi_shape'][-2], D['psi_shape'][-1])
        PSI_buff += [np.zeros(shape, dtype=DT)]
    
    OMG_buff = []
    for v in src[2]:
        _Rinf = partition(v, size, NK, NL, NX)
        shape = (len(_Rinf[1]), len(_Rinf[2]), len(_Rinf[3]),  D['omg_shape'][-4], NL, NX, NO)
        OMG_buff += [np.zeros(shape, dtype=DT)]
    
    DM    = [SIGS[-1].copy()]
    times = [t0]
    
    with open('MASTER.txt','w') as file:
        file.write('\nfile for Master node\n\n')
        
    try:
        os.mkdir(name+'_save')
    except:
        pass
    files = os.listdir(name+'_save')
    for f in files:
        if f[0:2]=='DM':
            os.remove(name+'_save/'+f)
    
    TIME_START = time()
    step_count = 0
    currents   = [[] for i in range(num_elecs)]
    
    while t0 <= t1:
        TE = 10 * eps
        while TE > eps:
            dt = A * h
            for i in range(RK_N):
                
                _s1 = time()
                _idx_i = idxholder[i]
                # tmp_* keeps the intermediate sigmas, psis and omegas.
                if i == 0:
                    tmp_sig[:,:,:]         = SIGS[-1]
                    tmp_psi[:,:,:,:,:]     = PSIS[-1]
                    tmp_omg[:,:,:,:,:,:,:] = OMGS[-1]
                else:
                    MM(SIGS[_idx_i].transpose(1, 2, 3, 0)             , B[i, _idx_i], tmp_sig)
                    MM(PSIS[_idx_i].transpose(1, 2, 3, 4, 5, 0)       , B[i, _idx_i], tmp_psi)
                    MM(OMGS[_idx_i].transpose(1, 2, 3, 4, 5, 6, 7, 0) , B[i, _idx_i], tmp_omg)
                
                ts = t0 + dt[i]; 
                Ht = H0 + dH(ts, tmp_sig)
                if config.MPI_LAZY_OMEGA:
                    for not_i in range(_nleads): delta[not_i] = bias(ts, not_i)
                    Xpp_delta = (_Xpp + delta[nax,:,nax,nax])
                    Xpm_delta = (_Xpm + delta[nax,:,nax,nax])
                
                _s2 = time()
                
                h   = comm.bcast(h , root=0)
                ts  = comm.bcast(ts, root=0)
                
                _s3 = time()
                
                # Send out arrays with capital method
                # PI
                
                for k, v in enumerate(src[0]):
                    kidx, lidx, xidx = PI_idx[k]
                    comm.Send(CA(tmp_psi[kidx[:, nax, nax], lidx[nax,:, nax], xidx[nax, nax, :]]), 
                                 dest = v, tag = PI_tags[k])
                _s4 = time()
                #Psi
                #Here we send the reduced arrays to each node
                for k, v in enumerate(src[1]):
                    kidx, lidx, xidx = PSI_idx[k]
                    comm.Send(  CA(tmp_sig[kidx,:,:]),
                                dest = v, tag = PSI_tags[k] )
                    comm.Send( CA(tmp_psi[kidx[:, nax, nax],lidx[nax,:,nax], xidx[nax, nax,:]]),
                               dest = v, tag = PSI_tags[k] )
                    comm.Send( CA(tmp_omg[kidx[:, nax, nax],lidx[nax,:,nax], xidx[nax, nax,:]]),
                               dest = v, tag = PSI_tags[k] )
                    comm.Send( CA(Ht[kidx,:,:]),
                               dest = v, tag = PSI_tags[k] )
                _s5 = time()
                
                #Omega
                #Here we send the full Psi to each node because the index structure necessitates this
                for k, v in enumerate(src[2]):
                    kidx, lidx, xidx = OMG_idx[k]
                    comm.Send( CA(tmp_psi[kidx]),
                               dest = v, tag = OMG_tags[k] )
                    # If we do "Lazy Omega", we evaluate the term with omega in the ODE
                    # on the master node and save the time needed for sending omega to the nodes
                    
                    if config.MPI_LAZY_OMEGA==False:
                        comm.Send( CA(tmp_omg[kidx[:, nax, nax],lidx[nax,:,nax], xidx[nax, nax,:]]),
                                  dest = v, tag = OMG_tags[k] )
                
                _s6 = time()
                
                # Calculate commutator on master node, while we wait for the results
                # reset array
                tmp_PI[:,:,:,:] = ( 0.0 + 0.0j )
                D_tmp_sig = -1j*(Ht@tmp_sig - tmp_sig@Ht)
                
                # We also calculate a part of omega while we wait for the results
                # if we furthermore have the lazy-omega method enabled
                if config.MPI_LAZY_OMEGA:
                    OuterSubtractionAssign(Xpp_delta * (1j*h/hbar),
                                           Xpm_delta * (1j*h/hbar),
                                           OMGS[-1],
                                           OMGS[i])
                # Recieve PI arrays
                for kk in range(len(src[0])):
                    comm.Recv( PI_buff[kk],
                               source = src[0][kk], tag = PI_tags[kk])
                    tmp_PI[PI_idx[kk][0][:, None],
                           PI_idx[kk][1][None, :],:,:] += PI_buff[kk]
                _s7 = time()
                if i == 0:
                    for elec_count in range(num_elecs):
                        currents[elec_count] += [2*np.trace(tmp_PI[:,elec_count], axis1 = 1, axis2 = 2).real/hbar]
                #    print(np.array(currents)[:,-1])
                #    sleep(1)
                PI_sum = tmp_PI.sum(axis = 1)
                # Calculate derivative of DM while we wait for the rest of the threads(?)
                D_tmp_sig += PI_sum + PI_sum.conj().transpose(0, 2, 1)
                D_tmp_sig *= (h/hbar)
                SIGS[i, :, :, :] = D_tmp_sig[:, :, :]
                # Sigma ODE results collected and done
                
                _s8 = time()
                
                
                # Calculate derivative of PSIs
                for kk in range(len(src[1])):
                    comm.Recv(PSI_buff[kk],
                              source=src[1][kk], tag=PSI_tags[kk])
                    #print(np.sum(PSI_buff[kk]))
                    PSIS[i, PSI_idx[kk][0][:,    None, None],
                            PSI_idx[kk][1][None,    :, None],
                            PSI_idx[kk][2][None, None, :   ],
                            :, :] = PSI_buff[kk]
                # Psi ODE results collected and done
                _s9 = time()
                # Calculate derivative of OMEGAs and place in array
                for kk in range(len(src[2])):
                    comm.Recv(OMG_buff[kk],
                              source=src[2][kk], tag=OMG_tags[kk])
                    if config.MPI_LAZY_OMEGA==False:
                        OMGS[i,OMG_idx[kk][0][:,    None, None],
                               OMG_idx[kk][1][None,    :, None],
                               OMG_idx[kk][2][None, None, :   ]
                               ] = OMG_buff[kk]
                    elif config.MPI_LAZY_OMEGA:
                        OMGS[i,OMG_idx[kk][0][:,    None, None],
                               OMG_idx[kk][1][None,    :, None],
                               OMG_idx[kk][2][None, None, :   ]
                               ] += OMG_buff[kk]
                # Omega ODE results collected and done
                _s10 = time()
                # Optionally write out some timings to file
                if print_timings:
                    with open('MASTER.txt','a') as file:
                        file.write('Total Time:    '  + str(_s10- _s1) + '\n')
                        file.write('Initial Matmul '  + str(_s2 - _s1) + '\n')
                        file.write('bcast numbers: '  + str(_s3 - _s2) + '\n')
                        file.write('Send to PI:    '  + str(_s4 - _s3) + '\n')
                        file.write('Send to PSI:   '  + str(_s5 - _s4) + '\n')
                        file.write('Send to OMG:   '  + str(_s6 - _s5) + '\n')
                        file.write('Recieve PI:    '  + str(_s7 - _s6) + '\n')
                        file.write('Calc DSIG:     '  + str(_s8 - _s7) + '\n')
                        file.write('Recieve PSI:   '  + str(_s9 - _s8) + '\n')
                        file.write('Recieve OMG:   '  + str(_s10 - _s9)+ '\n')
            # Evaluate error and deceide if we should repeat or step fourth, 
            # see outer while loop
            TE = TERR(SIGS[0:RK_N], PSIS[0:RK_N], OMGS[0:RK_N], CT)
            h  = hnew(h,  TE)
        
        # Step fourth and possibly save stuff
        step_fourth(SIGS, PSIS, OMGS, CH)
        t0    += h
        DM         += [SIGS[-1]]
        times      += [t0]
        step_count += 1
        
        if np.mod(step_count, saveevery)==0 and step_count>=saveevery:
            np.save(name+'_save/'+'last_sig',     SIGS[-1])
            np.save(name+'_save/'+'last_psi',     PSIS[-1])
            np.save(name+'_save/'+'last_omg',     OMGS[-1])
            np.save(name+'_save/'+'last_time',      t0)
            np.save(name+'_save/'+'last_stepsize',   h)
            np.save(name+'_save/DM'+str(step_count), DM)
            np.save(name+'_save/times'+str(step_count), times[-len(DM):])
            for k, c in enumerate(currents):
                np.save(name+'_save/'+'current_'+str(k) + '_'+str(step_count), c[-len(DM):])
            DM = []
            print('t = ' + str(t0) + ' | number of steps = ' + str(step_count) + ' | local error:' +str(format(TE, '.3e')))
            for k, c in enumerate(currents):
                _cur = np.array(c[-1])
                print('   Current in lead ' + str(k) + ': ' + str(np.round(_cur,4))[1:-1] + '| Total: ' + str(np.round(sum(c[-1]),4)))
            print('Latest steptime: ' + str(format(_s10-_s1, '.4e')) + 'seconds')
            print('\n')
    
    TIME_STOP = time()
    print('RUNTIME: ', TIME_STOP - TIME_START)
    
    #for k, c in enumerate(currents):
    #    np.save(name+'_save/'+'current_'+str(k)+ '_'+str(step_count), c[-len(DM):])
    #np.save(name+'_save/'+'last_sig',     sig0)
    #np.save(name+'_save/'+'last_psi',     psi0)
    #np.save(name+'_save/'+'last_omg',     omg0)
    #np.save(name+'_save/'+'last_time',      t0)
    #np.save(name+'_save/'+'last_stepsize',   h)
    #np.save(name+'_save/DM'+str(step_count), DM)
    #np.save(name+'_save/times'+str(step_count), times[-len(DM):])
    

def PI(role_info):
    assert 'PI' in role_info[0]
    DT   = np.complex128
    h,ts = 0.0, 0.0
    
    def pickout(Arr):return np.ascontiguousarray(Arr[k_idx   [:, None, None], 
                                                     lead_idx[None, :, None], 
                                                     pole_idx[None,None,:]] )
    k_idx    = role_info[1]
    lead_idx = role_info[2]
    pole_idx = role_info[3]
    number   = int(role_info[0][-1])
    noT      = int(D['noT'])
    tmp_psi  = np.zeros((len(k_idx), len(lead_idx), len(pole_idx), noT, NO), dtype = DT)
    PI_array = np.zeros((len(k_idx), len(lead_idx),                NO,  NO), dtype = DT)
    Ixi      = pickout(D['Ixi'])
    
    with open(role_info[0]+'.txt','w') as file:
        file.write('\nfile for worker node\n')
    
    while True:
        _s1 = time()
        h       = comm.bcast(h, root=0)
        ts      = comm.bcast(ts, root=0)
        _s2 = time()
        #tmp_psi = np.zeros(tmp_psi.shape, dtype = DT)
        comm.Recv(tmp_psi,  source = 0,  tag = PI_tags[number])
        _s3 = time()
        
        for i,a in enumerate(lead_idx):
            PI_array[:, i, :, :] = PI_a(tmp_psi[:, i], Ixi[:, i])
        _s4 = time()
        comm.Send(PI_array, dest=0, tag=PI_tags[number])
        _s5 = time()
        
        if print_timings:
            with open(role_info[0]+'.txt','a') as file:
                file.write('Total time    '+ str(_s5 - _s1) + '\n')
                file.write('Idle Time     '+ str(_s2 - _s1) + '\n')
                file.write('Time Recieve  '+ str(_s3 - _s2) + '\n')
                file.write('Time Calculate'+ str(_s4 - _s3) + '\n')
                file.write('Time Send     '+ str(_s5 - _s4) +'\n')

def PSI(role_info):
    assert 'DPSI' in role_info[0]
    DT      = np.complex128
    Ht      = np.zeros(D['H_Ortho'].shape, dtype=DT)
    
    k_idx    = role_info[1]
    lead_idx = role_info[2]
    pole_idx = role_info[3]
    number   = int(role_info[0][-1])
    noT      = int(D['noT'])
    
    def pickout(Arr):return np.ascontiguousarray(Arr[k_idx   [:, None, None], 
                                                     lead_idx[None, :, None], 
                                                     pole_idx[None,None,:]] )    
    tmp_sig = np.zeros((len(k_idx), NO, NO), 
                       dtype = DT)
    tmp_psi = np.zeros((len(k_idx), len(lead_idx), len(pole_idx), noT, NO),   
                       dtype = DT)
    tmp_omg = np.zeros((len(k_idx), len(lead_idx), len(pole_idx), noT, NL, NX, NO), 
                       dtype = DT)
    Ht      = np.zeros((len(k_idx), NO,NO), 
                       dtype = DT)
    D_tmp_psi = np.zeros(tmp_psi.shape, dtype=DT)
    alloc_ps  = np.zeros(tmp_psi.shape, dtype=DT)
    
    with open(role_info[0]+'.txt','w') as file:
        file.write('\nfile for worker node\n')
    
    
    h    = 0.0
    ts   = 0.0
    nk   = len(k_idx)    
    Ntot = len(pole_idx)
    nl   = len(lead_idx)
    no   = tmp_psi.shape[-1]
    Xpp          = pickout(D['Xpp'])
    GL_P         = pickout(D['GL_P'])
    diff_GGP_GLP = pickout(D['diff_GGP_GLP'])
    xi           = pickout(D['xi'])
    all_xi       = D['xi'][k_idx] / hbar**2
    
    om_shape = (nk, nl*Ntot*noT, NL*NX*no)
    all_xi_shape = (nk, NL*NX*no, -1) # still parallelized over k, this is why nk is here and not NK
    alloc_ps2 = np.zeros((nk, nl*Ntot*noT,  no),dtype = DT)
    
    
    delta_t = np.zeros(nl)
    while True:#cont:
        _s1 = time()
        
        h       = comm.bcast(h,       root=0)
        ts      = comm.bcast(ts,      root=0)
        
        _s2 = time()
        
        #Recv relevant psi and omega
        comm.Recv(tmp_sig, source = 0, tag = PSI_tags[number])
        comm.Recv(tmp_psi, source = 0, tag = PSI_tags[number])
        comm.Recv(tmp_omg, source = 0, tag = PSI_tags[number])
        comm.Recv(Ht     , source = 0, tag = PSI_tags[number])
        
        _s3 = time()
        
        for i,a in enumerate(lead_idx): delta_t[i] = bias(ts, a)
        
        Xpp_delta = Xpp + delta_t[nax, :, nax, nax]
        
        #D_tmp_psi  = tmp_psi @ np.expand_dims(Ht.transpose((0, 2, 1)), (1, 2)) / hbar
        MM(tmp_psi,  np.expand_dims((Ht/hbar).transpose((0, 2, 1)), (1, 2)), D_tmp_psi)
        
        #D_tmp_psi -= np.expand_dims(Xpp_delta[:, :,:, :noT] / hbar, 4) * tmp_psi 
        _multiply(np.expand_dims(Xpp_delta[:, :,:, :noT] / hbar, 4), tmp_psi, out = alloc_ps)
        D_tmp_psi -= alloc_ps
        
        #D_tmp_psi += np.expand_dims(GL_P[:, :, :, :noT], 4) * xi[:, :, :, :noT]
        _multiply(np.expand_dims(GL_P[:, :, :, :noT], 4), xi[:, :, :, :noT],  out = alloc_ps)
        D_tmp_psi += alloc_ps
        
        #D_tmp_psi += np.expand_dims(diff_GGP_GLP[:, :, :, :noT], 4) * (
        #                            xi[:, :, :,:noT] @ np.expand_dims(tmp_sig.transpose((0, 2, 1)), (1, 2))
        #                            )
        MM(xi[:, :, :,:noT], np.expand_dims(tmp_sig.transpose((0, 2, 1)), (1, 2)), alloc_ps)
        _multiply(np.expand_dims(diff_GGP_GLP[:, :, :, :noT], 4) , alloc_ps, out = alloc_ps)
        D_tmp_psi += alloc_ps
        
        # for last term use all_xi because it is a sum over all axc 
        #D_tmp_psi += (tmp_omg.reshape(om_shape) @ all_xi.reshape(all_xi_shape)
        #              ).reshape(D_tmp_psi.shape) / (hbar ** 2)
       # print((tmp_omg.reshape(om_shape) @ all_xi.reshape(all_xi_shape)).shape)
       # print(alloc_ps2.shape)
        MM(tmp_omg.reshape(om_shape), all_xi.reshape(all_xi_shape), alloc_ps2 )
        D_tmp_psi += alloc_ps2.reshape(D_tmp_psi.shape)
        
        # note h in factor
        D_tmp_psi *= (-1j * h)
        # Send back result
        
        _s4 = time()
        comm.Send(D_tmp_psi, dest=0, tag=PSI_tags[number])
        _s5 = time()
        
        if print_timings:
            with open(role_info[0]+'.txt','a') as file:
                file.write('Total Time     '+ str(_s5 - _s1) + '\n')
                file.write('Idle  Time     '+ str(_s2 - _s1) + '\n')
                file.write('Time Recieve   '+ str(_s3 - _s2) + '\n')
                file.write('Time Calculate '+ str(_s4 - _s3) + '\n')
                file.write('Time Send      '+ str(_s5 - _s4) + '\n')
                file.write('\n')
        
        
        
        
        
        

def OMG(role_info):
    assert 'DOMG' in role_info[0]
    DT = np.complex128
    k_idx    = role_info[1]
    lead_idx = role_info[2]
    pole_idx = role_info[3]
    noT  = int(D['noT'])
    def pickout(Arr):return np.ascontiguousarray(Arr[k_idx   [:, None, None], 
                                                     lead_idx[None, :, None], 
                                                     pole_idx[None,None,:]] )    
    
    TMP_psi = np.zeros((len(k_idx), NL, NX, noT, NO ),
                       dtype = DT)
    tmp_omg = np.zeros((len(k_idx), len(lead_idx), len(pole_idx), noT, 
                        NL, NX, NO), dtype = DT)
    psi_tilde = np.zeros(TMP_psi.shape,dtype = DT)
    
    number = int(role_info[0][-1])
    
    with open(role_info[0]+'.txt','w') as file:
        file.write('\nfile for worker node\n')
    
    h    = 0.0
    ts   = 0.0
    nk   = len(k_idx) 
    Ntot = len(pole_idx)
    nl   = len(lead_idx)
    no   = TMP_psi.shape[-1]
    Xpp          = pickout(D['Xpp'])
    Xpm_all      = D['Xpm'][k_idx]

    diff_GGP_GLP     = pickout(D['diff_GGP_GLP'])
    diff_GGM_GLM_all = D['diff_GGM_GLM'][k_idx]
    xi           = pickout(D['xi' ])
    Ixi_all      = D['Ixi'][k_idx]
    delta_t = np.zeros(nl)
    
    alloc_1 = np.zeros((nk, NL*NX*no,no),dtype = DT)
    alloc_2 = np.zeros((nk, nl*Ntot*noT, NL*NX*no),dtype = DT)
    alloc_3 = np.zeros((nk, nl*Ntot*noT, no), dtype = DT)
    alloc_4 = np.zeros((nk, nl*Ntot*noT, NL*NX*noT),dtype = DT)
    
    D_tmp_omg = np.zeros(tmp_omg.shape, dtype = DT)
    while True:
        _s1 = time()
        h       = comm.bcast(h,       root=0)
        ts      = comm.bcast(ts,      root=0)
        _s2 = time()
        
        comm.Recv(TMP_psi, source = 0, tag = OMG_tags[number])
        if config.MPI_LAZY_OMEGA==False:
            comm.Recv(tmp_omg, source = 0, tag = OMG_tags[number])
        
        _s3 = time()
        
        #pick out the needed matrix element for this worker:
        np.conjugate(TMP_psi,out = psi_tilde)   # All elements (except k)
        
        #tmp_psi  = np.ascontiguousarray(TMP_psi[:, lead_idx[:,nax], pole_idx[nax,:]])  # Reduced elements
        tmp_psi  = TMP_psi[:, lead_idx[:,nax], pole_idx[nax,:]]
        for i,a in enumerate(lead_idx): 
            delta_t[i] = bias(ts, a)
        
        Xpp_delta      = Xpp     + delta_t[nax,:, nax, nax]     
        Xpm_all_delta  = Xpm_all + delta_t[nax,:, nax, nax]
        
        # Thoughout this code,  we parallelize over the four first indices (no prime),
        # The primed indices are refered by CAPITAL, while local
        # are refered without capital letters
        #D_tmp_omg  = (tmp_psi[:, :, :, :noT, :].reshape(nk, nl*Ntot*noT, no)      # Reduced indices
        #              @
        #              ((diff_GGM_GLM_all.reshape(nk, NL*NX*no, 1)                 # Primed indices
        #                * Ixi_all.reshape(nk, NL*NX*no, no)).transpose((0, 2, 1))
        #              )
        #             ).reshape(tmp_omg.shape)
        _multiply(diff_GGM_GLM_all.reshape(nk, NL*NX*no, 1), Ixi_all.reshape(nk, NL*NX*no, no),
                  out = alloc_1)
        MM(tmp_psi[:, :, :, :noT, :].reshape(nk, nl*Ntot*noT, no), alloc_1.transpose((0,2,1)), 
           alloc_2)
        
        D_tmp_omg += alloc_2.reshape(tmp_omg.shape)
        
        # D_tmp_omg[:,:,:,0:noT, :, :, 0:noT] += ((diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1) # Reduced
        #                                          * xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no)
        #                                         ) 
        #                                         @ 
        #                                         (psi_tilde.reshape(nk, NL*NX*noT, no)                 # Primed
        #                                          .transpose((0, 2, 1))
        #                                         )
        #                                        ).reshape(nk, nl, Ntot, noT, NL, NX, noT)
        
        _multiply(diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1), xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no),
                  alloc_3)
        MM(alloc_3, psi_tilde.reshape(nk, NL*NX*noT, no).transpose((0, 2, 1)),
           alloc_4)
        D_tmp_omg[:,:,:, 0:noT,:,:,0:noT] += alloc_4.reshape(nk, nl, Ntot, noT, NL, NX, noT)
        
        #D_tmp_omg += (  Xpp_delta    [:, :  , :  ,:noT,  nax, nax, nax]*(1j/hbar)    # Reduced
        #              - Xpm_all_delta[:, nax, nax, nax,  :  , :  , :  ]*(1j/hbar)    # Primed
        #              ) * tmp_omg
        # In case the MPI_LAZY_OMEGA is set to True in config, we dont do this term here,
        # but on the master node instead, to save sending to this node in the first place,
        # and only sending some elements back.
        if config.MPI_LAZY_OMEGA==False:
            OuterSubtraction(Xpp_delta    [:, :  , :  ,:noT]*(1j/hbar),
                             Xpm_all_delta[:, :  , :  , :  ]*(1j/hbar),
                             tmp_omg,
                             D_tmp_omg)
        
        D_tmp_omg *= h
        
        _s4 = time()
        comm.Send(D_tmp_omg, dest = 0, tag = OMG_tags[number])
        D_tmp_omg[:,:,:,:,:,:,:] = 0.0 + 0j
        _s5 = time()
        if print_timings:
            with open(role_info[0]+'.txt','a') as file:
                file.write('Total Time     '+ str(_s5 - _s1) + '\n')
                file.write('Idle  Time     '+ str(_s2 - _s1) + '\n')
                file.write('Time Recieve   '+ str(_s3 - _s2) + '\n')
                file.write('Time Calculate '+ str(_s4 - _s3) + '\n')
                file.write('Time Send      '+ str(_s5 - _s4) + '\n')
        

def CA(Arr):
    return Arr#np.ascontiguousarray(Arr)

if ROLE == 'MASTER':
    ODE_SOLVER()
elif 'PI' in ROLE:
    PI(ROLE_INFO)
elif 'DPSI' in ROLE:
    PSI(ROLE_INFO)
elif 'DOMG' in ROLE:
    OMG(ROLE_INFO)
