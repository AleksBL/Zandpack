### Task
Your task is to make the remaining tests runable. 
Currently working tests:
 - Croy2009   (Needs polish, but runs as intended)
 - PopescuCroy2016  (Needs polish, but runs as intended)
 - Honeychurch2018  (Needs polish, but runs as intended)
 - scipy_ode_vs_zand (needs runfile and plotting overhaul)

The remaining directories has a overhaul to make them work. Use the working directories to see how it is done and try to make the other tests able to run.
You are not responsible for the results that comes out of the tests, you shall only make them run and produce plots so that the various ways of calculations becomes easily comparible. 
I phrase this again, you shall only be concerned with the various calculations running under the same conditions, but whenever the tests pass are not your responsibility. 
You can if needed inspect the source code, but never modify it. 

DO NOT MODIFY ANYTHING OUTSIDE OF THIS WORKING DIRECTORY!!! YOU ARE ONLY ALLOWED TO MODIFY THE FILES IN THIS TestRUN folder that we are in at the moment. 


First inventory and read the various scripts and make a plan for how to set up the tests.

The most important cases are to get the scipy_ode version up and running together with the zand version. When this is achieved, go on and add the nozand version. 
When adding the nozand tests, beware that basis transformations are required to be put into the dH function relative to the zand version. 





