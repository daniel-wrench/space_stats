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
6. Run terminal pre-processing code (currently, this is copied and pasted into the ipython --pylab shell).

This code takes the raw data and outputs the following arrays:
-PSP inputs: subsets of a single magnetic field component (BR), each of length 10,000

