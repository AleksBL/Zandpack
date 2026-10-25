### ERROR REPORT.
An error in the Runge-Kutta stepping method was discovered and fixed in commit 688375d and the fix is clear from 
this commit. While h_old should have been used to step forward the system state, h was used. See the comments in
in the nozand / zand codes for details, or inspect the commit on github. It is among some other modifications that
happened in the same commit. 

### COMMENTS
After testing, the calculations I've been running have only changed in the decimal places, a very small change in 
response, currents etc. Mostly what it seems to have fixed is the error under RK propagation being much more 
stable. Before this fix, it tended to fluctuate within the error tolerance, but quite erratically. At the moment
of writing this, I havent seen any qualitative changes in any results as a consequence of fixing this.


Date of disovery: Aug 25 2026
Date of fix:      Aug 25 2026 
