#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:27:11 2026

@author: aleks
"""

from pydoc import render_doc
from inspect import getsource
from siesta_python.siesta_python import SiP
from Zandpack.TimedependentTransport import TD_Transport
from Zandpack.wrapper import transiesta_hook, Control, Input
from Block_matrices.Block_matrices import block_sparse
from functools import partial
from Zandpack.equations import Eqs

# import Zandpack
SiP_method_table= dir(SiP)
drop_names_SiP =[
              'Add_Bounded_Plane', 'Add_Charged_Bounded_Plane', 
              'Add_Charged_Box', 'Add_Charged_Sphere',
              'E_field', 'Find_edges','__class__', '__delattr__', 
              '__dict__', '__dir__', '__doc__', '__eq__', 
              '__firstlineno__', '__format__', '__ge__', 
              '__getattribute__', '__getstate__', '__gt__', 
              '__hash__', '__init_subclass__', 
              '__le__', '__lt__', '__module__', '__ne__', 
              '__new__', '__reduce__', '__reduce_ex__', 
              '__repr__', '__setattr__', '__sizeof__', 
              '__static_attributes__', '__str__', '__subclasshook__', 
              '__weakref__', '_has_tbt_nc_files', 'add_atom',
              'barebone_fdf', 'calculate_2E_RSSE', 'calculate_bulksurf_RSSE',
              'cell_deviation', 'cell_offset', 'center_cell',
              'create_added_stuff_dic','create_mixed_pseudo_dic',
              'dQ', 'delete_fdf', 'dump_pseudo_list_to_struct', 'fast_dftb_hk',
              'find_polygons', 'fois_gras',  'gen_basis', 'get_RS_pos', 
              'get_contour_from_failed_RSSE', 'gin', 'hear',
              'make_concentrated_k_points', 'make_device_coords', 'make_meshed_k_grid',
              'manual_k_points', 'move', 'move_atom', 'mulliken', 'nnr', 
              'pickle', 'plot_calculated_bands', 'pretty_cell', 'projection',
              'ps_mixer','read_basis_and_geom', 'read_dftb_hamiltonian', 'read_minimal_TSHS',
              'read_relax', 'read_rssi_surface_TSHS','save_file', 'scream',  'set',
              'set_methfessel_paxton', 'set_parallel_k', 'set_struct', 'set_synthetic_mixes', 'sisl_sub',
              'sleep', 'solve_qp_equation', 'standardise_cell', 'tbtrans_H_S_btd_pivot', 'tile_dm',
              'to_TB_model', 'undo_RSSE', 'which_type_SE', 'write_TB_model','reararange', 
              'write_analysis_results', 'write_and_run_dftb_in_dir', 'write_tb_trans_kp', 'x_to_z',
              'custom_bandlines', 'dftb2sisl', 'dftb_find_equiv', 'dftb_hsd', 'dftb_mullikencharge_interface',
              'Dipole_Correction_Vacuum', 'Real_space_SE', 'Real_space_SI', 'Real_space_SI_3D',
              'add_mixed_pseudo_dic', 'add_stuff','make_pretty', 'is_RSSE', 'fatband_fdf', 
              'Put_Variable_Charged_Sphere_On_All', 'run_fatbands', 'Wrap_unit_cell',
              'reset_minimal_hamiltonian', "Passivate_with_molecule",'run_gulp_in_dir',
              'add_elecs', 'get_labelled_indices','ideal_cell_charge','is_siesta_done',
              'run_analyze_in_dir', 'relaxed_system_energy', "add_buffer","get_potential_energy",
              "get_pseudo_paths","self_energy_from_tbtrans", "fdf_relax",
              
              ]
SiP_method_table = [v for v in SiP_method_table if v not in drop_names_SiP]
TD_Transport_method_table= dir(TD_Transport)
drop_names_TDT = [ '__class__','__delattr__', '__dict__',
 '__dir__','__doc__', '__eq__', '__firstlineno__','__format__',
 '__ge__','__getattribute__','__getstate__','__gt__',
 '__hash__','__init_subclass__','__le__','__lt__',
 '__module__','__ne__',
 '__new__','__reduce__','__reduce_ex__','__repr__',
 '__setattr__','__sizeof__','__static_attributes__',
 '__str__','__subclasshook__','__weakref__', 'make_device',
 'make_f','make_f_experimental',
 'make_f_general','make_f_gpu',
 'make_f_purenp', 'Electrodes', 'run_device',
 'run_device_non_eq', 'run_electrodes', 'get_orbital_values',
 'get_H_interpolation_function', 'get_dense_matrices','printLval'
 ]
TD_Transport_method_table = [v for v in TD_Transport_method_table if v not in drop_names_TDT]

transiesta_hook_method_table = dir(transiesta_hook)
generic_drop = ['__class__',
 '__delattr__', '__dict__',
 '__dir__', '__doc__',
 '__eq__', '__firstlineno__',
 '__format__', '__ge__',
 '__getattribute__', '__getstate__',
 '__gt__', '__hash__',
 '__init_subclass__',
 '__le__', '__lt__',
 '__module__', '__ne__',
 '__new__', '__reduce__',
 '__reduce_ex__', '__repr__',
 '__setattr__', '__sizeof__',
 '__static_attributes__', '__str__',
 '__subclasshook__', '__weakref__',]
control_drop = ["create_subfolder", "create_wd", "init_from_other",
                "into_wd", "out_wd", "rawlog", "write_rawlog", "write_log",
                ""]
transiesta_hook_method_table = [v for v in transiesta_hook_method_table 
                                if v not in generic_drop]
Control_method_table = dir(Control)
Control_method_table = [v for v in Control_method_table 
                                if v not in generic_drop + control_drop]
Input_method_table = dir(Input)
Input_method_table = [v for v in Input_method_table 
                                if v not in generic_drop]


block_sparse_method_table = ['__init__', 
                             'iterative_PSD', 
                             'is_positive_semidefinite',
                             'is_hermitian', 
                             'fit_sampling_and_Lorenzian',
                             'Block', 
                             ]




def md_ds(Name):
    return """
    Method to get documentation on the given function name from the """+Name+""" class
    Args:
        method_name: name of function (see """+Name+"""_method_table)
    Returns:
        pydoc rendering of the method.
    """
def sc_ds(Name):
    return """
    Method to source_code on the given function name from the """+Name+""" class
    Args:
        method_name: name of function (see """+Name+"""_method_table)
    Returns:
        pydoc rendering of the method.
    """
def getclass(n):
    if n == "SiP": return SiP
    if n == "TD_Transport": return TD_Transport
    if n == "transiesta_hook": return transiesta_hook
    if n == "Control": return Control
    if n == "Input": return Input
def class_method_description(classname: str, method_name: str)-> str:
    Class = getclass(classname)
    try:
        return str(render_doc(Class.__dict__[method_name]))
    except:
        return "ERROR: Didnt find the method name"
def class_method_source_code(classname: str, method_name: str)-> str:
    Class = getclass(classname)
    try:
        return str(getsource(Class.__dict__[method_name]))
    except:
        return "ERROR: Didnt find the method name"

TD_Transport_method_description         = partial(class_method_description, "TD_Transport")
TD_Transport_method_description.__doc__ = md_ds("TD_Transport")
TD_Transport_method_description.__name__="TD_Transport_method_description"
TD_Transport_method_source_code         = partial(class_method_source_code, "TD_Transport")
TD_Transport_method_source_code.__doc__ = sc_ds("TD_Transport")
TD_Transport_method_source_code.__name__="TD_Transport_method_source_code"


SiP_method_description                  = partial(class_method_description, "SiP")
SiP_method_description.__doc__          = md_ds("SiP")
SiP_method_description.__name__         = "SiP_method_description"
SiP_method_source_code                  = partial(class_method_source_code, "SiP")
SiP_method_source_code.__doc__          = sc_ds("SiP")
SiP_method_source_code.__name__         = "SiP_method_source_code"

transiesta_hook_method_description             = partial(class_method_description, "transiesta_hook")
transiesta_hook_method_description.__doc__     = md_ds("transiesta_hook")
transiesta_hook_method_description.__name__    = "transiesta_hook_method_description"
transiesta_hook_method_source_code             = partial(class_method_source_code, "transiesta_hook")
transiesta_hook_method_source_code.__doc__     = sc_ds("transiesta_hook")
transiesta_hook_method_source_code.__name__    = "transiesta_hook_method_source_code"

Control_method_description             = partial(class_method_description, "Control")
Control_method_description.__doc__     = md_ds("Control")
Control_method_description.__name__    = "Control_method_description"
Control_method_source_code             = partial(class_method_source_code, "Control")
Control_method_source_code.__doc__     = sc_ds("Control")
Control_method_source_code.__name__    = "Control_method_source_code"


Input_method_description             = partial(class_method_description, "Input")
Input_method_description.__doc__     = md_ds("Input")
Input_method_description.__name__    = "Input_method_description"
Input_method_source_code             = partial(class_method_source_code, "Input")
Input_method_source_code.__doc__     = sc_ds("Input")
Input_method_source_code.__name__    = "Input_method_source_code"


block_sparse_method_description             = partial(class_method_description, "block_sparse")
block_sparse_method_description.__doc__     = md_ds("block_sparse")
block_sparse_method_description.__name__    = "block_sparse_method_description"
block_sparse_method_source_code             = partial(class_method_source_code, "block_sparse")
block_sparse_method_source_code.__doc__     = sc_ds("block_sparse")
block_sparse_method_source_code.__name__    = "block_sparse_method_source_code"

def table2str(table):
    out = ""
    for s in table:
        out += s +'\n'
    return out

structure_description =f"""CODE INFORMATION
A Zandpack calculation uses several classes. The TD_Transport, transiesta_hook, Control
Input, SiP and block_matrices is described here. Any of the class methods listed below can be called with the function name
to obtain source code or function documentation.

TD_Transport:
```python
from Zandpack.TimedependentTransport import TD_Transport
```
Use TD_Transport_method_description for method description.
Use TD_Transport_method_source_code for method source code.
TD_Transport class has methods:
""" +table2str(TD_Transport_method_table)+"""

transiesta_hook:
```python
from Zandpack.wrapper import transiesta_hook
```
Use transiesta_hook_method_description for method description.
Use transiesta_hook_method_source_code for method source code.
transiesta_hook class has methods:
""" +table2str(transiesta_hook_method_table)+"""

Control:
```python
from Zandpack.wrapper import Control
```
Use Control_method_description for method description.
Use Control_method_source_code for method source code.
Control class has methods:
""" +table2str(Control_method_table)+"""

Input:
```python
from Zandpack.wrapper import Input
```
Use Input_method_description for method description.
Use Input_method_source_code for method source code.
Input class has methods:
""" +table2str(Input_method_table)+"""
Comments on Control, Input and transiesta_hook:
Control wraps the calling of the commandline tools from the
Zandpack/cmdtools and Zandpack/mpi folders. Most importantly it the calls (Step 4.1, 4.2 and 4.3)
```bash
# The directory in which these commands are run contains the Bias.py and Initial.py files.
# SCF and psinought steps are done using the orthogonal basis, while the
# zand/nozand step can be done in either the orthogonal basis (using zand) or the nonorthognal basis (nozand)
modify_occupations Dir=$PWD file=SourceFile outfile=YourFileName ....
SCF Dir=$PWD file=YourFileName ....
psinought Dir=$PWD YourFileName ....
# or nozand -> zand
mpirun nozand Dir=$PWD
```
The Input class writes the Initial.py and Bias.py files automatically (see examples/genetic_wrapper/wrapper_script.py).
The transiesta_hook provides the callable function H_from_DFT(nosig) to get the Hamiltonian given the nonorthogonal density matrix (nosig).

The siesta_python code contains the SiP class, described here.
```python
from siesta_python.siesta_python import SiP
```
Use SiP_method_description for method description.
Use SiP_method_source_code for method source code.
SiP class has methods:
""" +table2str(SiP_method_table)+"""

The Block_matrices code contains the block_sparse class, described here.
```python
from Block_matrices.Block_matrices import block_sparse
```
Use block_sparse_method_description for method description.
Use block_sparse_method_source_code for method source code.
block_sparse class has methods:
""" +table2str(block_sparse_method_table)+"""

"""

_basedir = __file__[:-len("documentation.py")]
examples = [('examples/generic_wrapper/wrapper_script.py', 
             "generic implementation of Zandpack Step 4 using the wrapper module."
             ),
            ('examples/ThesisExample/5a.py', 
             "Zandpack step 1. Simple carbon nanoribbon structure using Transiesta."
             ),
            ('examples/ThesisExample/5b.py', 
             "Zandpack step 2. Read in TBtrans results. Retain only pz orbital using sub_orbital keyword."
             ),
            ('examples/ThesisExample/5d.py', 
             "Zandpack step 3. Fit level-width functions of electrodes. Ensure levelwidth function is positive semidefinite using iterative_PSD."
             ),
            ("examples/ThesisExample/TDeq_1.sh",
             "Zandpack step 4. Set up and carry out timedependent propagation"
             ),
            ("examples/ThesisExample/Bias.py",
             "Zandpack step 4. Bias.py defines the Hamiltonian dependence on the density matrix and defines the junction bias."
             ),
            ("examples/ThesisExample/Initial.py",
             "Zandpack step 4. Initial.py defines the propagation parameters, t0, t1, precision etc."
             ),
            ("examples/AuBreakJunction/Ham.py",
             "Zandpack step 1. Structure setup of gold (Au) break junction. Run TranSIESTA."
             ),
            ("examples/AuBreakJunction/Init.py",
             "Zandpack step 2. Read in results from TBtrans."
             ),
            ("examples/AuBreakJunction/Fit.py",
             "Zandpack step 3. R Fit level-width functions of electrodes. Ensure levelwidth function is positive semidefinite using iterative_PSD."
             ),
            ("examples/AuBreakJunction/calc0/jobpsi0",
             "Zandpack step 4. slurm job script that runs psinought and the zand code."
             ),
            ("examples/C60/1_make_device.py",
             "Zandpack step 1. Uses ASE to setup C60 molecule bewteen two metallic electrodes and runs TranSIESTA."
             ),
            ("examples/C60/2_read_calc.py",
             "Zandpack step 2. Read in the results from TBtrans into the TD_Transport class."
             ),
            ("examples/C60/3_make_fit.py",
             "Zandpack step 3. Fit the levelwidth functions to Lorentzians. Ensure levelwidth function is positive semidefinite using iterative_PSD."
             ),
            ("examples/C60/calculate/Bias.py",
             "Zandpack step 4. Bias.py defines the Hamiltonian dependence on the density matrix and defines the junction bias. Here is no density matrix in this simple example."
             ),
            ("examples/C60/calculate/Initial.py",
             "Zandpack step 4. Initial.py defines the propagation parameters, t0, t1, precision etc."
             ),
            ("examples/Hubbard/1Hamiltonian.py",
             "Zandpack step 1. Use the Hubbard mean field code to get the Initial self-consistent electronic Hamiltonian. "
             ),
            ("examples/Hubbard/2Load.py",
             "Zandpack step 2. Read in results from custom transport calculation in this case. Dont do any device electrode overlap corrections."
             ),
            ("examples/Hubbard/3Fit.py",
             "Zandpack step 3. Fit levelwidth function. Ensure levelwidth function is positive semidefinite using iterative_PSD."
             ),
            ("examples/Hubbard/Calculation/Bias.py",
             "Zandpack step 4. Bias.py for TD propagation. density matrix dependence is specified here as the same used in the Hubbard mean field code."
             ),
            ("examples/Hubbard/Calculation/Initial.py",
             "Zandpack step 4. Initial.py defines the propagation parameters, t0, t1, precision etc."
             ),
            ("examples/AGNR+Tip/Hamiltonian.py",
             "Zandpack step 1."
             ),
            ("examples/AGNR+Tip/IntialTD.py",
             "Zandpack step 2."
             ),
            ("examples/AGNR+Tip/Fit.py",
             "Zandpack step 3."
             ),
            ("examples/AGNR+Tip/3E_6_2/DFTB_driver.py",
             "Allows one to use the DFTB+ code to get the electronic Hamiltonian for both steady-state SCF and timedependent calculations."
             ),
            ("examples/AGNR+Tip/3E_6_2/Bias.py",
             "Zandpack step 4. Imports Hamiltonian function from DFTB_driver.py and uses it to define the density matrix dependence of the Hamiltonian. Also contains linearization schemes for faster Hamiltonian evaluation."
             ),
            ("examples/AGNR+Tip/3E_6_2/Initial.py",
             "Zandpack step 4. Normal Initial.py file."
             ),
#            ("",
#             ""
#             ),
            ]

def get_example(examplename: str) -> str:
    try:
        txt = open(_basedir+examplename,"r").read()
    except:
        txt = 'FileNotFound'
    return txt

structure_description += f"""
EXAMPLES
There are  examples available through calling the get_example tool. 
These are important to get the context of how the classes from the CODE INFORMATION section works together.
The get_example_tool should be called as \"get_example(\"examples/Dir/file\")\", possibly with additional directory depth. 
A list of examples are listed below with small comments:
"""
for e in examples:
    txt, cmt = e
    structure_description += txt + " (" +cmt +")\n"
def get_equation(name):
    try:
        return Eqs[name]
    except:
        return "EquationNotFound"

structure_description+=f"""
There are furthermore more examples than what is listed here, but you will have to see the zandpack_directory_tree to see all examples. You will also find jupyter notebooks that you can get information from.

EQUATIONS
Equations for what is being solved is also available using the get_equations tool. It will return equations for the following names:
"""
for k in Eqs.keys():
    structure_description += k +'\n'
banned_folders = set(["__pycache__", "mpitest", "_devel", "_junk", "dep_mpi_jena", "Pade_data", "nb_tutorial_old", "svg"])
banned_files = ["DOP54.py", "GETPATH.py", "HartreeFromDensity.py",
                "Interpolation.py", "LDA.py", "LanczosAlg.py",
                "Linalg_factorisation.py", "Plotting.py","Louivillian.py",
                "Optimized_RK45.py", "basicbackground.txt", "mpi_RK4_pars.py"]
import os
def is_dirs_banned(dirs):
    for d in dirs:
        #print(d)
        if d in banned_folders:
            # print("Found Banned")
            return True
    return False
def get_zandpack_readme(Input: str) -> str:
    """
    Args:
       Any argument will get the readme
    Returns
       README.md from Zandpack top folder
    """
    return open(_basedir+"../README.md", "r").read()
def get_cmdtools_readme(Input: str) -> str:
    """
    Args:
       Any argument will get the readme
    Returns
       README.txt from cmdtools folder
    """
    return open(_basedir+"cmdtools/README.txt", "r").read()
def get_mpi_readme(Input: str) -> str:
    """
    Args:
       Any argument will get the readme
    Returns
       README.txt from mpi folder
    """
    return open(_basedir+"mpi/README.txt","r").read()
def get_tool_help(cmd: str) -> str:
    """
    Args:
       toolname: str: "Adiabatic", "SCF", "psinought", "modify_occupations", "td_info"
    """
    return open(_basedir+'cmdtools/'+cmd+'_help.txt',"r").read()
def zandpack_directory_tree(Input: str) -> str:
    """
    Args:
       Any argument will get the file tree
    Returns
       Directory tree structure for Zandpack package
    """
    startpath = _basedir
    tree = "Zandpack directory structure"
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in banned_folders]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 2 * (level)
        tree += '{}{}/'.format(indent, os.path.basename(root)) +"\n"
        subindent = ' ' * 2 * (level + 1)
        for f in files:
            if any([ff in f for ff in banned_files]):
                pass
            else:
                tree += '{}{}'.format(subindent, f)+"\n"
    return tree
structure_description+=f"""
Tool for total Zandpack package README file:
get_zandpack_readme (takes argument "true")
Tool for cmdtools README file (description for each tool in one file):
get_cmdtools_readme (takes argument "true" )
Tool for mpi README file (description for zand and nozand usage):
get_mpi_readme (takes argument "true")
Tool for Zandpack directory file structure (file tree):
zandpack_directory_tree (takes argument "true")
Tool for full keyword list for some of the tools in the cmdtools folder:
get_tool_help (takes string argument "Adiabatic", "SCF", "psinought", "modify_occupations", "td_info")
"""

available_functions = {"TD_Transport_method_description": TD_Transport_method_description,
                       "TD_Transport_method_source_code": TD_Transport_method_source_code,
                       "SiP_method_description": SiP_method_description,
                       "SiP_method_source_code": SiP_method_source_code,
                       "transiesta_hook_method_description":transiesta_hook_method_description,
                       "transiesta_hook_method_source_code":transiesta_hook_method_source_code,
                       "Control_method_description":Control_method_description,
                       "Control_method_source_code":Control_method_source_code,
                       "Input_method_description":Input_method_description,
                       "Input_method_source_code":Input_method_source_code,
                       "block_sparse_method_description":block_sparse_method_description,
                       "block_sparse_method_source_code":block_sparse_method_source_code,
                       "get_example": get_example,
                       "get_equation":get_equation,
                       "get_zandpack_readme":get_zandpack_readme,
                       "get_cmdtools_readme":get_cmdtools_readme,
                       "get_mpi_readme":get_mpi_readme,
                       "zandpack_directory_tree":zandpack_directory_tree,
                       "get_tool_help":get_tool_help,
                       }
