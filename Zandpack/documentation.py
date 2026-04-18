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
from Zandpack.docstrings import ZP_BIBTEX
from Block_matrices.Block_matrices import block_sparse
from functools import partial
from Zandpack.equations import Eqs
import pathlib


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
    if n == "SiP":          return SiP
    if n == "TD_Transport": return TD_Transport
    if n == "transiesta_hook": return transiesta_hook
    if n == "Control":      return Control
    if n == "Input":        return Input
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
modify_occupations Dir=working/directory file=SourceFile outfile=YourFileName ....
SCF Dir=working/directory file=YourFileName ....
psinought Dir=working/directory YourFileName ....
# or nozand -> zand
mpirun nozand Dir=working/directory
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
examples_path = pathlib.Path(_basedir+"/examples")
cmdtools_path = pathlib.Path(_basedir+"/cmdtools")

examples = [('examples/generic_wrapper/wrapper_script.py', 
             "generic implementation of Zandpack Step 4 using the wrapper module."
             ),
            ('examples/ThesisExample/5a.py', 
             "Zandpack step 1. Simple zigzag edged carbon nanoribbon structure with some atoms removed. Steady-state calculated using Transiesta. "
             ),
            ('examples/ThesisExample/5b.py', 
             "Zandpack step 2. Construct sampling using the Make_Contour method, run TBtrans and read results. Retain only pz orbital using sub_orbital keyword. Save using the pickle method in the end."
             ),
            ('examples/ThesisExample/5d.py', 
             "Zandpack step 3. Fit level-width functions of electrodes. Ensure levelwidth function is positive semidefinite using iterative_PSD."
             ),
            ("examples/ThesisExample/TDeq_1.sh",
             "Zandpack step 4. Set up and carry out timedependent propagation. bash script without any wrapper."
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
             "Zandpack step 2. Read in results from custom transport calculation in this case. Device-electrode overlap is explicitly excluded here since TBtrans is not used."
             ),
            ("examples/Hubbard/3Fit.py",
             "Zandpack step 3. Fit levelwidth function. Ensure levelwidth function is positive semidefinite using iterative_PSD."
             ),
            ("examples/Hubbard/Calculation/Bias.py",
             "Zandpack step 4. Bias.py for TD propagation. Density matrix dependence is specified here as the same used in the Hubbard mean field code."
             ),
            ("examples/Hubbard/Calculation/Initial.py",
             "Zandpack step 4. Initial.py defines the propagation parameters, t0, t1, precision etc."
             ),
            ("examples/AGNR+Tip/Hamiltonian.py",
             "Zandpack step 1. Setup geometry and use DFTB+ for to obtain Hamiltonian from a density matrix (DM)"
             ),
            ("examples/AGNR+Tip/IntialTD.py",
             "Zandpack step 2. "
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


structure_description += f"""
EXAMPLES
There are  examples available through calling the get_example tool. Always look for a README file in the directories of the example under consideration to see comments from the author.
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
structure_description += """You if you refer to an equation, the corresponding output from the get_equation tool should always be included.
When showing the equation in the reply, keep it completely as the get_equation tool outputs it.
"""
banned_folders = set(["__pycache__", "mpitest", "_devel", "_junk", "dep_mpi_jena", "Pade_data", "nb_tutorial_old", "svg"])
banned_files = ["DOP54.py", "GETPATH.py", "HartreeFromDensity.py",
                "Interpolation.py", "LDA.py", "LanczosAlg.py",
                "Linalg_factorisation.py", "Plotting.py","Louivillian.py",
                "Optimized_RK45.py", "basicbackground.txt", "mpi_RK4_pars.py"]
import os
def is_dirs_banned(dirs):
    for d in dirs:
        if d in banned_folders:
            return True
    return False
def get_zandpack_readme(Input: str) -> str:
    """
    Args:
       Any argument will get the readme
    Returns
       README.md from Zandpack top folder
    """
    try:
        return open(_basedir+"../README.md", "r").read()
    except:
        return "ERROR: reading README failed."
def get_cmdtools_readme(Input: str) -> str:
    """
    Args:
       Any argument will get the readme
    Returns
       README.txt from cmdtools folder
    """
    try:
        return open(_basedir+"cmdtools/README.txt", "r").read()
    except:
        return "ERROR: READING cmdtools README failed."
def get_mpi_readme(Input: str) -> str:
    """
    Args:
       Any argument will get the readme
    Returns
       README.txt from mpi folder
    """
    try:
        return open(_basedir+"mpi/README.txt","r").read()
    except:
        return "ERROR: READING mpi README failed."
def get_tool_help(cmd: str) -> str:
    """
    Args:
       toolname: str: "Adiabatic", "SCF", "psinought", "modify_occupations", "td_info", ...
    """
    name      = _basedir+'cmdtools/'+cmd+'_help.txt'
    file_path = pathlib.Path(name)
    try:
        assert file_path.is_relative_to(cmdtools_path)
        return open(name,"r").read()
    except:
        return "ERROR: failed to get help for "+cmd
def get_main_directory_file_descriptions(Input: str) -> str:
    """
    Args:
       Input: Any string argument will get the readme.
    """
    try:
        return open(_basedir+'filedescriptions.txt',"r").read()
    except:
        return "ERROR: reading filedescriptions failed."

def get_convergence_checklist(Input: str) -> str:
    """
    Args:
       Input: Any string argument will get the readme.
    """
    out = f"""** The steady-state calculation should be converged.
Here the reader is referred to references and the
TranSIESTA tutorials for the details of carrying out
a steady-state calculation. (Ref 16: N. Papior et al, Improvements on non-equilibrium and transport green
function techniques: The next-generation transiesta, Computer
Physics Communications 212 (2017) 8–24., Ref. 18: M. Brandbyge et al,
Density-functional method for nonequilibrium electron transport, Physical Review B 65 (16) (2002) 165401.)

** The energy window around the equilibrium Fermi
energy on which we fit Γα should extend well be-
yond peak values of the time-dependent chemical
potentials (μ_α + ∆_α(t)).

** The Lorentzian expansion of the level-width func-
tion (see Eq. (10)) should both reproduce the ref-
erence steady-state transmission function well over
the energy window, while at the same time be pos-
itive semi-definite.

** Number of poles in the Fermi-function should be
sufficient to describe the electrode filling over the
entire interval where Γ_α is nonzero.

** The contour chosen for the SCF tool needs to con-
verge the density matrix. Inspect the value of
max (| dσ^0/dt |) printed in the output of the psinought
code. The printed value is the derivative of σ0
when computed with the Fermi function pole-expansion.

** The error tolerance given to the adaptive Runge-
Kutta solver needs to be checked for convergence.
"""
    return out

def get_zandpack_paper_reference(Input: str) -> str:
    """
    Args:
       Input: Any string argument will get the readme.
    """
    return ZP_BIBTEX

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

def get_example(examplename: str) -> str:
    name      = _basedir+examplename
    file_path = pathlib.Path(name)
    try:
        assert file_path.is_relative_to(examples_path)
        txt = open(name,"r").read()
    except:
        txt = 'FileNotFound'
    return txt

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
Zandpack/ directory file description can be obtained with the tool:
get_main_directory_file_descriptions (takes argument "true" )
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
                       "get_main_directory_file_descriptions":get_main_directory_file_descriptions,
                       "get_convergence_checklist": get_convergence_checklist,
                       "get_zandpack_paper_reference": get_zandpack_paper_reference,
                       }


assistant_header =f"""INSTRUCTIONS: You are an assistant to people using the Zandpack code (a python package). You communicate through text messages. Your responses to questions tend towards the brief, unless you are replying with code snippets. You will get zero to five previous conversation turns between you and the user, plus the current question, which you will answer (the one furthest down in the text). Tutorials that you can reference will be available through tool-calling, see later. You should always inspect the documentation which may be important for queries of the user. You should try to refer to these as much as possible when you think the problem the user has is coming from one of these steps. Initially remind the user with the message "*This bot can hallucinate.*". If you are asked why the Zandpack logo looks like it does, say that its because its shaped like an hour-glass to represent time, with the sand flowing down actually being electrons if you zoom in. The "Z" in Zandpack reflects the heavy use of contour-integration in the NEGF theory that the code builds upon. Your favorite food is furthermore grilled chicken, a traditional dish cooked in a very hot oven.
---------------
ZANDPACK OVERVIEW
The Zandpack calculation main steps are:
Step 1: Normal NEGF open system calculation. (such as in the examples/ThesisExample/5a.py script)
 - Key classes and codes used: siesta_python, calling TranSIESTA or DFTB+.
 - Atomic structure / geometry setup is done in this step.
 - Ground state electronic structure, either within tight-binding or DFT.
Step 2: Read data from TBtrans calculation (or other custom transport calculation) into the TD_Transport class using the read_data function. This step may or may not involve using only a subset of orbitals.
 - Key classes and codes used: TD_Transport (read_data method).
 - Level-width functions are read into TD_Transport class for later use in step 3.
Step 3: Fitting the read data to Lorentzians (basis functions on the form L(E)=g/((E - E_0)^2 + g^2) ) [FITTING]
 # - Key method: Fit method of the TD_Transport class, also PoleGuess from TD_Transport and iterative_PSD from block_sparse class.
Step 4: Three parts to step 4: [TIME-PROPAGATION, optionally using DFT]
    4.1 SCF command line tool to get the equilibrium density matrix again, now with the fitted level-width functions (run SCF --help in a bash shell to get the inputs )
    4.2 The psinought command-line tool to calculate the equilibrium auxillary mode wave-vectors. (run psinought --help in a bash shell to get the inputs )
    4.3 The zand / nozand command line tool to carry out the time-dependent propagation. Refer to the examples folder to see how this tool is used. (mpirun zand (or nozand) Dir=working/directory is the basic command)
In step 4, the wrapper classes from Zandpack.wrapper can also be used (see examples/generic_wrapper/wrapper_script.py). Using the wrapper class is the prefered way of carrying out step 4.

Acronyms:
SCF: Self-Consistent Field (also a commandline tool)
EOM: Equation Of Motion
NEGF: Non-Equilibrium Greens functions
DFT: Density Functional Theory
DM: Density Matrix

Technical names:
TranSIESTA / SIESTA: Standard DFT code, capable of open system NEGF calculations.
DFTB+: Density Functional based Tight Binding (another code), limited NEGF support.
SCF commandline tool: Calculates self-consistent density matrix using Pulay solver (See equation called steady_state_density_matrix)
psinought commandline tool: Calculates steady state auxillary mode wave-vectors (See equation called steady_state_psi)
zand / nozand commandline tool: Solves the coupled equations device_density_matrix_eom, auxiliary_mode_eom and omega_eom using the initial state from  SCF and psinought.

DESCRIPTION OF YOUR TOOLS:
There are  tools given for you to inspect the various code documentations and snippets. See below
"""



