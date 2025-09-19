"""
GPU: Depricated flag.
NUMBA: Use numba at  (set true)
NUMBA_PARALLEL: set parallel flag in numba decorator to true
NUMBA_OUTER... Parallel impl. of outer subtraction. set to true if you use mixed paralllelism
CACHE: Set false unless you really want a fast startup time. Downside is you can get a function that has been compiled for another cpu
Parallel_PI: Set to True
PI: NUMBA OR NUMPY. NUMBA is faster
MPI_LAZY_OMEGA: Depricated flag.
Supress_Parallel_k: change behavior of splitting of k-points
FASTMATH: numba flag.
RK_Method: RK45, RK78, DOP54 (RKF45, RKF78, DOPRI5). See mpi_RK4pars_dev.py for more info

"""



GPU                              = False
NUMBA                            = True
NUMBA_PARALLEL                   = True
NUMBA_OUTER_SUBTRACTION_PARALLEL = True
CACHE                            = False
PARALLEL_PI                      = True
PI_VERSION                       = 'NUMBA'
MPI_LAZY_OMEGA                   = False
Check_partition_scheme           = False
Supress_parallel_K               = False
FASTMATH                         = True
RK_Method                        = 'DOP54'
N_pi_summers                     = 1