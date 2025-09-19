import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from TimedependentTransport.TimedependentTransport import PI as PI_a
from TimedependentTransport.TimedependentTransport import Jk
from Initial import Arrs as D
from Initial import t0, t1, stepsize, print_timings, usesave,saveevery, name, eps, hbar
from Bias import dH, bias
from TimedependentTransport.mpi_RK4pars import A, B, C, CT, CH
from TimedependentTransport.mpi_splitter import partition, get_sources
from time import time, sleep
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nax= np.newaxis
NK = D['H_Ortho'].shape[0]
NO = D['H_Ortho'].shape[-1]
NL = D['psi_shape'][1]
NX = D['psi_shape'][2]

src = get_sources(size)
#mk_pi, mk_psi, mk_omg = assemble_prescriptions(size)

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
#NUMBER    = int(ROLE[-1])
print('Process ', rank, ' has role', ROLE)

def ODE_SOLVER(stepsize=stepsize, T0=t0, T1=t1):
    sleep(5)    
    print('\nMASTER PROCESS STARTED\n')
    def TERR(y1, y2, y3):
        res  = np.sum(np.abs((y1.transpose(1, 2, 3,        0)@CT))**2)
        res += np.sum(np.abs((y2.transpose(1, 2, 3, 4, 5,    0)@CT))**2)
        res += np.sum(np.abs((y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CT))**2)
        return np.sqrt(res)

    def step_fourth(yp1, yp2, yp3, Y1, Y2, Y3):
        res = (yp1 + Y1.transpose(1, 2, 3,        0)@CH,
               yp2 + Y2.transpose(1, 2, 3, 4, 5,    0)@CH,
               yp3 + Y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CH)
        return res

    def hnew(h, TE): return 0.9 * h * (eps / TE) ** 0.2
    DT = np.complex128
    # Common Variables
    H0      = D['H_Ortho']
    Ht      = np.zeros(H0.shape, dtype=DT)
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
    
    tmp_sig = D['DM_Ortho'].copy()
    tmp_psi = np.zeros(D['psi_shape'], dtype=DT)
    tmp_omg = np.zeros(D['omg_shape'], dtype=DT)
    tmp_PI  = np.zeros((NK, NL, NO, NO), dtype = DT)
    
    SIGS = np.zeros((6,)+sig0.shape, dtype=DT)
    PSIS = np.zeros((6,)+psi0.shape, dtype=DT)
    OMGS = np.zeros((6,)+omg0.shape, dtype=DT)
    
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
    
    DM    = [sig0.copy()]
    times = [t0]
    
    with open('MASTER.txt','w') as file:
        file.write('\nfile for Master node\n')
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
            for i in range(6):
                _s1 = time()
                
                if i == 0:
                    tmp_sig = sig0.copy()
                    tmp_psi = psi0.copy()
                    tmp_omg = omg0.copy()
                else:
                    tmp_sig = sig0 + SIGS[0:i].transpose(1, 2, 3, 0)             @ B[i, 0:i]
                    tmp_psi = psi0 + PSIS[0:i].transpose(1, 2, 3, 4, 5, 0)       @ B[i, 0:i]
                    tmp_omg = omg0 + OMGS[0:i].transpose(1, 2, 3, 4, 5, 6, 7, 0) @ B[i, 0:i]
                
                ts = t0 + dt[i]; 
                Ht = H0 + dH(ts, tmp_sig)
                
                _s2 = time()
                
                h   = comm.bcast(h , root=0)
                ts  = comm.bcast(ts, root=0)
                
                _s3 = time()
                
                # Send out arrays with capital method
                #PI
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
                    comm.Send( CA(tmp_omg[kidx[:, nax, nax],lidx[nax,:,nax], xidx[nax, nax,:]]),
                               dest = v, tag = OMG_tags[k] )
                
                _s6 = time()
                
                # Calculate commutator on master node, while we wait for the results
                tmp_PI[:,:,:,:] = 0.0 +0.0j
                D_tmp_sig = -1j*(Ht@tmp_sig - tmp_sig@Ht)
                
                for kk in range(len(src[0])):
                    comm.Recv( PI_buff[kk],
                               source = src[0][kk], tag = PI_tags[kk])
                    tmp_PI[PI_idx[kk][0][:, None],
                           PI_idx[kk][1][None, :],:,:] += PI_buff[kk]
                _s7 = time()
                if i == 0:
                    for elec_count in range(num_elecs):
                        currents[elec_count] += [2*np.trace(tmp_PI[:,elec_count], axis1 = 1, axis2 = 2).real/hbar]
                
                PI_sum = tmp_PI.sum(axis = 1)
                # Calculate derivative of DM while we wait for the rest of the threads(?)
                D_tmp_sig += PI_sum + PI_sum.conj().transpose(0, 2, 1)
                D_tmp_sig *= (h/hbar)
                SIGS[i, :, :, :] = D_tmp_sig[:, :, :]
                _s8 = time()
                
                # Calculate derivative of PSIs
                for kk in range(len(src[1])):
                    comm.Recv(PSI_buff[kk],
                              source=src[1][kk], tag=PSI_tags[kk])
                    PSIS[i, PSI_idx[kk][0][:,    None, None],
                            PSI_idx[kk][1][None,    :, None],
                            PSI_idx[kk][2][None, None, :   ],
                            :, :] = PSI_buff[kk]
                _s9 = time()
                # Calculate derivative of OMEGAs and place in array
                for kk in range(len(src[2])):
                    comm.Recv(OMG_buff[kk],
                              source=src[2][kk], tag=OMG_tags[kk])
                    OMGS[i,OMG_idx[kk][0][:,    None, None],
                           OMG_idx[kk][1][None,    :, None],
                           OMG_idx[kk][2][None, None, :   ]
                           ] = OMG_buff[kk]
                _s10 = time()
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
            
            TE = TERR(SIGS, PSIS, OMGS)
            h  = hnew(h,  TE)        
        
        sig0, psi0, omg0 = step_fourth(sig0, psi0, omg0, SIGS, PSIS, OMGS)
        t0    += h
        
        DM         += [sig0]
        times      += [t0]
        step_count += 1
        
        if np.mod(step_count, saveevery)==0 and step_count>=saveevery:
            np.save(name+'_save/'+'last_sig',     sig0)
            np.save(name+'_save/'+'last_psi',     psi0)
            np.save(name+'_save/'+'last_omg',     omg0)
            np.save(name+'_save/'+'last_time',      t0)
            np.save(name+'_save/'+'last_stepsize',   h)
            np.save(name+'_save/DM'+str(step_count), DM)
            np.save(name+'_save/times'+str(step_count), times[-len(DM):])
            for k, c in enumerate(currents):
                np.save(name+'_save/'+'current_'+str(k) + '_'+str(step_count), c[-len(DM):])
            DM = []
    
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
    all_xi       = D['xi'][k_idx]
    
    om_shape = (nk, nl*Ntot*noT, NL*NX*no)
    all_xi_shape = (nk, NL*NX*no, -1) # still parallelized over k, this is why nk is here and not NK
    
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
        
        D_tmp_psi  = tmp_psi @ np.expand_dims(Ht.transpose((0, 2, 1)), (1, 2)) / hbar
        D_tmp_psi -= np.expand_dims(Xpp_delta[:, :,:, :noT] / hbar, 4) * tmp_psi 
        
        D_tmp_psi += np.expand_dims(GL_P[:, :, :, :noT], 4) * xi[:, :, :, :noT]
        D_tmp_psi += np.expand_dims(diff_GGP_GLP[:, :, :, :noT], 4) * (
                                    xi[:, :, :,:noT] @ np.expand_dims(tmp_sig.transpose((0, 2, 1)), (1, 2))
                                    )
        # for last term use all_xi because it is a sum over all axc 
        D_tmp_psi += (tmp_omg.reshape(om_shape) @ all_xi.reshape(all_xi_shape)
                      ).reshape(D_tmp_psi.shape) / (hbar ** 2)
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
    
    while True:
        _s1 = time()
        h       = comm.bcast(h,       root=0)
        ts      = comm.bcast(ts,      root=0)
        _s2 = time()
        
        comm.Recv(TMP_psi, source = 0, tag = OMG_tags[number])
        comm.Recv(tmp_omg, source = 0, tag = OMG_tags[number])
        
        _s3 = time()
        
        #pick out the needed matrix element for this worker:
        psi_tilde= TMP_psi.conj()   # All elements (except k)
        tmp_psi  = np.ascontiguousarray(TMP_psi[:, lead_idx[:,nax], pole_idx[nax,:]])  # Reduced elements
        
        for i,a in enumerate(lead_idx): delta_t[i] = bias(ts, a)
        
        Xpp_delta      = Xpp     + delta_t[nax,:, nax, nax]     
        Xpm_all_delta  = Xpm_all + delta_t[nax,:, nax, nax]
        
        # Thoughout this code,  we parallelize over the four first indices (no prime),
        # but we might some components which are usually parallelized over.
        # The primed indices are refered by CAPITAL, while local
        # are refered without capital letters
        D_tmp_omg  = (tmp_psi[:, :, :, :noT, :].reshape(nk, nl*Ntot*noT, no)      # Reduced indices
                      @
                      ((diff_GGM_GLM_all.reshape(nk, NL*NX*no, 1)                 # Primed indices
                        * Ixi_all.reshape(nk, NL*NX*no, no)).transpose((0, 2, 1))
                      )
                     ).reshape(tmp_omg.shape)
        
        D_tmp_omg[:,:,:,0:noT, :, :, 0:noT] += ((diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1) # Reduced
                                                 * xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no)
                                                ) 
                                                @ 
                                                (psi_tilde.reshape(nk, NL*NX*noT, no)                 # Primed
                                                 .transpose((0, 2, 1))
                                                )
                                               ).reshape(nk, nl, Ntot, noT, NL, NX, noT)
        
        D_tmp_omg += (  Xpp_delta    [:, :  , :  ,:noT,  nax, nax, nax]*(1j/hbar)    # Reduced
                      - Xpm_all_delta[:, nax, nax, nax,  :  , :  , :  ]*(1j/hbar)    # Primed
                      ) * tmp_omg 
        
        D_tmp_omg *=h
        _s4 = time()
        comm.Send(D_tmp_omg, dest = 0, tag = OMG_tags[number])
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
