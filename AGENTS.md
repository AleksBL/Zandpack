### Task
Your main tasks when working on is project is:
  - Do code review on the main codes in the project. This includes
     * TimedependentTransport.py and the TD_Transport class
     * The programs in the cmdtools and mpi folders. The nozand is the main work-horse of this project.
        - These tools have some dependencies you can figure out from the imports in each.
     * The wrapper.py and recipe.py modules

DO NOT modify or add any code yourself unless explicitly prompted. Your default mode of work is coding guidance, review and idea-generation, not writing it yourself. 

The AI_code folder is a folder dedicated to AI generated code (from elsewhere, you cannot add or modify code ANYWHERE). As with all the code, changes should be scrutinized and sufficient comments should be present. 

Use the .gitignore as a baseline for files that can safely be ignored, unless explicitly asked to work on these. 

### Open problems / areas of interest
1. nozand performance.
  - Use of blas functions zgemv, zgemm directly in the code instead of using the 
    numpy matmul wrapper. Several places have 
2. Getting the tests to run. 
  - Currently only three of the tests are online. 
  - The rest needs some modification in order to run as they are outdated and the code API has gone through some changes since they were made. 
  - Furthermore, the examples given in the examples folder can also be used as tests, if adapted properly (Made to work with the scipy_ode wrapper) for independent testing.
  
Also see the issues.txt file to see the issues pulled from GitHub.

### Coding guidelines
- Code related to numerical calculations should be decomposed in functions, whose scope is well defined (has a specific job in some larger script or context) and should preferably not exceed 60 lines including docstring. Furthermore, these should prefer readability over conciseness. Similarly with class methods, these should also have a welldefined purpose, be readable, and preferably be well below 60 lines each. 
- Any time data code is made that writes data to the harddisk, unless there are some good reason not to, the data should be compressed eg. using the numpy.savez_compressed function or the xz tool the situation allows. 
- If at any point you are asked to  modify a file / files, there should be a validation stage where whatever new code is validated against the old. Exceptions to this is if there actually is a mistake which needs correction, in which case the user has to be explicitly notified and a before - after bugfix process. Be sure the user is made aware of any bugs / mistakes.  
- Remember the comments and make sure the code-base is understandable in reasonable time by a human reader. Initial iterations / experimentation of new code can be longer, more ad-hoc, but final versions should have this bullet-point in mind.
