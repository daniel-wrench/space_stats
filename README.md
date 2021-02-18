# Daniel Wrench's Awesome Space Stats Repo

Log-in to Raapoi. 


*data_import_funcs.py*  
*calculate_stats.py* contains a function called plot_results() which is used for visualising the predictions of the neural network on the test data

**PROCESS**
1. Set-up singularity tensorflow container in Linux on local computer. I had to do this using Linux on a Virtual Machine, following the instructions from https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example  
2. Log into Raapoi (or relevant HPC cluster).
3. (Set up bin/python_env file to source for running commands in ipython shell.)
4. Use sftp to copy across tensor.sif file
5. Use lftp to download NASA datasets from SPDF into data/spacecraft folder.
6. Run *terminal_pre-processing.py* code (currently, this is copied and pasted into the ipython --pylab shell).  
This code takes the raw data (currently majority PSP) and outputs the following arrays:  
-PSP clean inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with no gaps  
-PSP gapped inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with artificial gaps of 20% removed in three chunks. *The gapped datasets are chosen to be the final 20% of the original datasets*. These are output in both filled (0 (mean) imputed) and unfilled versions.  
-PSP clean and gapped outputs: expected structure functions corresponding to each (original, ungapped) dataset  
-PSP (gapped) math outputs: structure functions for the gapped datasets, calculated *after* gapping took place.  
-Voyager gappy inputs: a small number of normalised subsets of Voyager data. *These are already very gappy.* Voyager data is just for testing - there is no "expected" output because we do not have the complete datasets  
-Voyager (gapped) math outputs: structure functions for the gapped datasets  

7. Run *ml_program.py* by submitting it to the cluster inside the *singularity_submit.sh* file. This also contains the CPU, memory, and time requests for the job. Currently these are set to 6 cpus-per-task and 6GB of memory.  
This file does the following:  
-Reads in the arrays above  
