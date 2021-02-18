# Daniel Wrench's Awesome Space Stats Repo

### CODES
**data_import_funcs.py**  Contains functions for importing CDF files and outputting dated pandas DataFrames.  
**remove_obs_funcs.py** Contains functions to remove data from dataframes    
**calculate_stats.py** Contains a function called plot_results() which is used for visualising the predictions of the neural network on the test data  


### PROCESS
1. Set-up singularity tensorflow container in Linux on local computer. I had to do this using Linux on a Virtual Machine, following the instructions from https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example  
2. Log into Raapoi (or relevant HPC cluster).
3. (Set up bin/python_env file to source for running commands in ipython shell.)
4. Use sftp to copy across tensor.sif file
5. Use lftp to download NASA datasets from SPDF into data/spacecraft folder.
6. Run **terminal_pre-processing.py** code. (Currently, this is copied and pasted into the ipython --pylab shell. This is because we need access to the strfn() function from the TurbAn repository.)
This code takes the raw data (currently majority PSP) and outputs the following arrays:  
-PSP clean inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with no gaps  
-PSP gapped inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with artificial gaps of between 10% and 40% removed in between 3 and 5 chunks. *The gapped datasets are chosen to be the final 20% of the original datasets*. These are output in both filled (0 (mean) imputed) and unfilled versions.  
-PSP clean and gapped outputs: expected structure functions corresponding to each (original, ungapped) dataset. These are the second-order structure functions, and are calculated up to the lag equal to 20% of the input data length (2000 points). **Calculating these for all subsets takes around 15min.**
-PSP (gapped) math outputs: structure functions for the gapped datasets, calculated *after* gapping took place.  
-Voyager gappy inputs: a small number of normalised subsets of Voyager data. *These are already very gappy.* Voyager data is just for testing - there is no "expected" output because we do not have the complete datasets  
-Voyager (gapped) math outputs: structure functions for the gapped datasets  

7. Run **ml_program.py** by submitting it to the cluster inside the **singularity_submit.sh** file. This also contains the CPU, memory, and time requests for the job. Currently these are set to 6 cpus-per-task and 6GB of memory.  
This file does the following:  
-Reads in the arrays above.  
-Concatenates the clean and gapped datasets. These constitute the full data which will be used for training and testing the model. We create a *gapped_key* to keep track of which of the datasets are gapped. We then randomly select 10% of the data to be used for testing, with the remaining 90% being used for training.  
-Builds and trains a feed-forward back-progragation neural network, or multi-layer perceptron (MLP). The number of nodes in the output layer is equal to the length each structure function. As well as the number of nodes, we specify the dropout layers, optimizer, learning rate, loss function, validation split, and number of epochs to train for.  
-Evaluates the model by calculating the loss function on the test data.  
-Outputs the following arrays:  
--test_gapped_indices: Array with a 1 if the corresponding test subset is gapped, 0 otherwise.  
--inputs_train and inputs_test: Full training and testing input data  
--ann_test_predictions and ann_test_observed: Expected and predicted structure functions for the test data  
--voyager_gapped_predictions: Predicted structure functions for the gappy Voyager data  
8. Plot a series of expected and observed functions using the calc.plot_results() function in the ipython terminal
9. Visualise a select few prediction curves against the expected and mathematical curves, using the **plotting.py** code
