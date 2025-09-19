import numpy as np
import sys
from Zandpack.TimedependentTransport import Jk
from Zandpack.TimedependentTransport import PI
import k0nfig
from time import time
import matplotlib.pyplot as plt
#print(__file__[:-17])
sys.path.append(__file__[:-17])

if k0nfig.GPU:
    print('Turn off GPU SUPPORT\n')
    print('Turn off GPU SUPPORT\n')
    print('Turn off GPU SUPPORT\n')
    print('Turn off GPU SUPPORT\n')
    print('Turn off GPU SUPPORT\n')
    print('Turn off GPU SUPPORT\n')

def AdaptiveRK4_Opti(f, sig0, psi0, omega0, eps, t0, t1,
                     dH, delta_func, Ixi,
                     h_guess = None, dH_given = True,
                     print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
                     write_func = None, print_step = 50, plot = False, use_GPU = False,
                     atol = None, rtol = None,
                     elec_names = ['left', 'right']):
    
    # Adaptive timestep RK4
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    B = np.array([[np.nan   ,      np.nan,    np.nan ,    np.nan, np.nan],
                  [1/4      ,      np.nan,    np.nan ,    np.nan, np.nan],
                  [3/32     ,      9/32  ,    np.nan ,    np.nan, np.nan],
                  [1932/2197, -7200/2197 , 7296/2197 ,    np.nan, np.nan],
                  [439/216  , -8         , 3680/513  , -845/4104, np.nan],
                  [-8/27    , 2          , -3544/2565, 1859/4104, -11/40]
                  ]
                  )
    
    C  = np.array([25/216, 0, 1408 / 2565 , 2197 / 4104, -1/5, np.nan ])
    CH = np.array([16/135, 0, 6656 /12825 , 28561/56430, -9/50 , 2/55 ])
    CT = np.array([1/360 , 0, -128/4275   , -2197/75240,  1/50 , 2/55 ])
    
    if k0nfig.GPU and use_GPU:
        assert  1 == 0
        xp  = cp
        Cur = Jk_gpu
        PI_x = PI_gpu
        save_array = cp.asnumpy
    else:
        xp         = np
        Cur        = Jk
        PI_x       = PI
        save_array = np.asarray
    
    def TERR(y1, y2, y3):
        res = 0.0
        res+=xp.sum(xp.abs((y1.transpose(1,2,3,        0)@CT))**2)
        res+=xp.sum(xp.abs((y2.transpose(1,2,3,4,5,    0)@CT))**2)
        res+=xp.sum(xp.abs((y3.transpose(1,2,3,4,5,6,7,0)@CT))**2)
        return xp.sqrt(res)
    
    def step_fourth(yp1,yp2,yp3,Y1,Y2,Y3):
        res  =(
               yp1 + Y1.transpose(1,2,3,        0)@CH,
               yp2 + Y2.transpose(1,2,3,4,5,    0)@CH,
               yp3 + Y3.transpose(1,2,3,4,5,6,7,0)@CH,
              )
        
        return res    
    
    def hnew(h, eps, TE):
        return 0.9 * h * (eps / TE) ** (1 / 5)
    
    def scalar_mult(Arr,number):
        Arr*=number

    current_left   = []
    current_right  = []
    density_matrix = []
    times          = []
    
    if sig0 is None or psi0 is None or omega0 is None:
        state_sig = xp.load(name+'_last_sig.npy')
        state_psi = xp.load(name+'_last_psi.npy')
        state_omg = xp.load(name+'_last_omega.npy')
        if t0 is None:
            t0        = float(xp.load(name+'_last_time.npy'))
            if h_guess is None:
                h_guess = float(xp.load(name + '_last_dt.npy'))
    else:
        state_sig =  sig0.copy()
        state_psi =  psi0.copy()
        state_omg =  omega0.copy()
    
    if h_guess is None:
        h = (t1-t0)/1000
    else:
        h  = 0.0
        h += h_guess
    
    step =  0
    T0   =  0
    T0  += t0
    
    with open(name+'.txt','w') as file:
        file.write('\nStart of Propagation\n')
    time_start= time()
    data = dict()
    for e in elec_names:
        data.update({'current_'+e:[]})
    
    SIGS   = np.zeros((6,)+state_sig.shape, dtype = state_sig.dtype)
    PSIS   = np.zeros((6,)+state_psi.shape, dtype = state_psi.dtype)
    OMGS   = np.zeros((6,)+state_omg.shape, dtype = state_omg.dtype)
    
    while t0 <= t1:
        TE = 10 * eps
        while TE > eps:
            dt = A * h
            
            SIGS[0],PSIS[0], OMGS[0] =     f(t0+dt[0], state_sig, state_psi, state_omg, 
                                             dH, delta_func, dH_given = dH_given )
            scalar_mult(SIGS[0],h)
            scalar_mult(PSIS[0],h)
            scalar_mult(OMGS[0],h)
            
            SIGS[1],PSIS[1], OMGS[1] =     f(t0+dt[1], state_sig + B[1,0] * SIGS[0], 
                                                       state_psi + B[1,0] * PSIS[0], 
                                                       state_omg + B[1,0] * OMGS[0],
                                             dH, delta_func, dH_given = dH_given )
            
            scalar_mult(SIGS[1],h)
            scalar_mult(PSIS[1],h)
            scalar_mult(OMGS[1],h)
            
            SIGS[2],PSIS[2], OMGS[2] =     f(t0+dt[2], state_sig + SIGS[0:2].transpose(1,2,3,        0) @ B[2,0:2],
                                                       state_psi + PSIS[0:2].transpose(1,2,3,4,5,    0) @ B[2,0:2], 
                                                       state_omg + OMGS[0:2].transpose(1,2,3,4,5,6,7,0) @ B[2,0:2], 
                                             dH, delta_func, dH_given = dH_given)
            scalar_mult(SIGS[2],h)
            scalar_mult(PSIS[2],h)
            scalar_mult(OMGS[2],h)
            
            SIGS[3],PSIS[3], OMGS[3] =     f(t0+dt[3], state_sig + SIGS[0:3].transpose(1,2,3,        0) @ B[3,0:3],
                                                       state_psi + PSIS[0:3].transpose(1,2,3,4,5,    0) @ B[3,0:3], 
                                                       state_omg + OMGS[0:3].transpose(1,2,3,4,5,6,7,0) @ B[3,0:3], 
                                             dH, delta_func, dH_given = dH_given)
            scalar_mult(SIGS[3],h)
            scalar_mult(PSIS[3],h)
            scalar_mult(OMGS[3],h)
            
            SIGS[4],PSIS[4], OMGS[4] =     f(t0+dt[4], state_sig + SIGS[0:4].transpose(1,2,3,        0) @ B[4,0:4],
                                                       state_psi + PSIS[0:4].transpose(1,2,3,4,5,    0) @ B[4,0:4], 
                                                       state_omg + OMGS[0:4].transpose(1,2,3,4,5,6,7,0) @ B[4,0:4], 
                                             dH, delta_func, dH_given = dH_given)
            scalar_mult(SIGS[4],h)
            scalar_mult(PSIS[4],h)
            scalar_mult(OMGS[4],h)
            
            SIGS[5],PSIS[5], OMGS[5] =     f(t0+dt[5], state_sig + SIGS[0:5].transpose(1,2,3,        0) @ B[5,0:5],
                                                       state_psi + PSIS[0:5].transpose(1,2,3,4,5,    0) @ B[5,0:5], 
                                                       state_omg + OMGS[0:5].transpose(1,2,3,4,5,6,7,0) @ B[5,0:5], 
                                             dH, delta_func, dH_given = dH_given)
            
            scalar_mult(SIGS[5],h)
            scalar_mult(PSIS[5],h)
            scalar_mult(OMGS[5],h)
            
            if fixed_mode:
                break
            else:
                TE  =  TERR(SIGS,PSIS,OMGS)
                h   =  hnew(h, eps, TE)
        
        density_matrix += [save_array(state_sig[:,:,:])]
        for _ie, e in enumerate(elec_names):
            data['current_'+e]+=[save_array(Cur(PI_x(state_psi[:, _ie],Ixi[:, _ie]) ) ) ]
        
        state_sig, state_psi, state_omg = step_fourth(state_sig, state_psi, state_omg, SIGS,PSIS,OMGS)
        times  +=  [t0]
        t0     +=   h
        
        if np.mod(step, print_step) == 0:
            with open(name + '.txt', 'a') as file:
                file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
                file.write('current timestep: '+str(h) +'fs\n')
                file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
                if write_func is not None:
                    write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
                if plot==True:
                    plt.show()
                    for _ie, e in enumerate(elec_names):
                        plt.plot(xp.array(times), xp.array(data['current_'+e]), label = str(_ie))
                    plt.xlabel('Time [fs]', size = 20)
                    plt.savefig('Current(t)',dpi =300)
                    plt.show()
                    plt.pause(0.05)
            
            xp.save(name + '_last_sig',   state_sig)
            xp.save(name + '_last_psi',   state_psi)
            xp.save(name + '_last_omega', state_omg)
            xp.save(name + '_last_time', xp.array(times[-1]))
            xp.save(name + '_last_dt',xp.array(h))
            
            xp.save('_times', xp.array(times))
            current_keys = [k for k in data.keys() if 'current' in k]
            for ck in current_keys:
                xp.save('_'+ck, save_array(data[ck]))
            
            xp.save('_#electrons_device',xp.trace(save_array(density_matrix),axis1 = 2, axis2=3 ))
            
        step+=1
    
    data.update({'density matrix': save_array(density_matrix)})
    
    xp.save(name + '_last_sig', state_sig)
    xp.save(name + '_last_psi', state_psi)
    xp.save(name + '_last_omega', state_omg)
    xp.save(name + '_last_time', xp.array(times[-1]))
    xp.save('_times', xp.array(times))
    xp.save(name + '_last_dt',xp.array(h))
    runtime = time()-time_start
    
    with open(name + '.txt', 'a') as file:
        file.write('100%\n')
        file.write('Runtime: ' + str(runtime) + 'seconds')
    
    return np.array(times), data

