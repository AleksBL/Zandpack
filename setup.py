import os
IL = "INSTALL_LOG.txt"
WD = os.getcwd()
#os.chdir('Zandpack')
with open(IL, "w") as f:
    print('-------',file=f)
    print('We are working in folder: ',file=f)
    print(WD,file=f)
    print('-------',file=f)
    print('Executable files should have status (755) or more.', file=f)
#for f in os.listdir('mpi'):
mask = oct(os.stat('Zandpack/mpi/zand').st_mode)[-3:]
with open(IL, "a") as f:
    print('-------',file=f)
    print('zand executable status: ' + str(mask) ,file=f)

mask = oct(os.stat('Zandpack/cmdtools/psinought').st_mode)[-3:]
with open(IL, "a") as f:
    print('-------',file=f)
    print('psinought executable status: ' + str(mask) ,file=f)

mask = oct(os.stat('Zandpack/cmdtools/SCF').st_mode)[-3:]
with open(IL, "a") as f:
    print('-------',file=f)
    print('SCF executable status: ' + str(mask) ,file=f)

with open(IL, "a") as f:
    print('-------',file=f)
    print('If any mistakes are seen in this file, manually set the permissions with chmod. ' ,file=f)
    print('The ones printed here are the important ones, but you might encounter', file=f)
    print('permission problems on the other files in the cmdtools folder. Fix it with chmod! ' ,file=f)
    print('Examples of adding to your PATH variable for easy execution: ', file=f)
    print('    export PATH=$PATH:'+WD+'/Zandpack/mpi',file=f)
    print('    export PATH=$PATH:'+WD+'/Zandpack/cmdtools',file=f)
    print('Copy these to your .bashrc file (on linux).',file=f)
    




from setuptools import setup
setup(name='Zandpack',
      version='1.0',
      description='Module for calculating timedependent Charge transport in open quantum systems using LCAO models from DFT or tight-binding.',
      url='',
      author='Aleksander Bach Lorentzen',
      author_email='aleksander.bl.mail@gmail.com',
      license='MPL-2.0',
      packages=['Zandpack'],
      zip_safe=False,
      install_requires = ["numpy", 
                          "numba", 
                          "sisl", 
                          "matplotlib", 
                          "scipy", 
                          "siesta_python",
                          "Block_matrices",
                          "Gf_Module",
                          "psutil",
                          "joblib",
       ])

