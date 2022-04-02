# Neural Nets for Space Stats

This code was used to explore solar wind data and implement a study of using neural networks (as well as simpler methods) to estimate structure functions of the solar wind from time series with gaps.

### HELPER FUNCTIONS

- **data_import_funcs.py**  Contains functions for importing CDF files and outputting dated pandas DataFrames.
- **remove_obs_funcs.py** Contains functions to remove data from dataframes
- **calculate_stats.py** Contains a function called plot_results() which is used for visualising the predictions of the neural network on the test data

### SET-UP

1. Set-up a Singularity Tensorflow container in Linux on local computer. I had to do this using Linux on a Virtual Machine, following the instructions from [https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example](https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example).
2. Log into Raapoi (or relevant HPC cluster).
3. (Set up `bin/python_env` file to source for running commands in ipython shell.)
4. Use `sftp` to copy across `tensor.sif` file
5. Use `lftp` to download NASA datasets from SPDF into data/spacecraft folder.

### SCRIPT EXECUTION PROCESS & PSEUDO-CODE

The scripts can be run inside or outside of Rapoi. Currently scripts 1-3 have corresponding batch submission files (.sh) that allows them to be submitted as jobs to the Rapoi cluster.

All of the following is to be run in the Rāpoi terminal.
View that state of cluster jobs with `vuw-myjobs`.

0. `source ~/bin/python_env` *if you are running anything in the terminal*
1. `sbatch 1_batch_job.sh` *Only have to do this once, depending on the size, number, and frequency of original unique intervals you want*

    `1_process_data.py` does the following:
    1. Load data from CDF files, get time and date column formatted correctly
    2. Re-sample dataframe to correct freq (dataframe)
    3. Return the amount of missing data, both before and after re-sampling
    4. Output the final tidy data as pkl files

2. `python 2_batch_job.sh` *Only have to do this once, depending on the number of duplicate intervals to make, how much to gap each one, and what proportion of data to use for test set.* **Takes about 20min.**
    
    This should be run in the cluster using a bash script. For now using `srun` produces a `MemoryError`. The next step will be to try either converting from float64 to float32 in the python script, *or* writing a `.sh` file according to Tulasi’s example to submit to the cluster. Using the `.sh` job I was able to produce 5 x 156 copies. This only took 1.5 seconds per interval, and would not work when running in interactive mode. At minimum, 8 is too much for the `2_singularity_submit.sh` job, with 64 cpus per task and 3G mem per CPU

    `2_process_data.py` takes the .pkl data (currently majority PSP) and outputs the following arrays:
    - PSP clean inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with no gaps
    - PSP gapped inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with artificial gaps of between 10% and 40% removed in between 3 and 5 chunks. *The gapped datasets are chosen to be the final 20% of the original datasets*. These are output in both filled (0 (mean) imputed) and unfilled versions.
    - PSP clean and gapped outputs: expected structure functions corresponding to each (original, ungapped) dataset. These are the second-order structure functions, and are calculated up to the lag equal to 20% of the input data length (2000 points). **Calculating these for all subsets takes around 15min.**
    - PSP (gapped) math outputs: structure functions for the gapped datasets, calculated *after* gapping took place.

    This pipeline consists of two major functions:

    **`mag_interval_pipeline_split()`**, specifying the length of the data and the number of intervals to split it into, and the proportion to set aside for testing. This splits the dataset into standardised intervals, then groups them into a training and test set, both of which are lists of dataframes. The intermediate outputs are a plot and the summary statistics of the first interval, before and after standardisation, and the dimensions of the final outputs
   
    The arguments I have specified to this function are to separate out 80% of input-output pairs for training, 20% for testing, for PSP. For MMS, use 100% of intervals for testing.

    To each of these sets, the second major function **`mag_interval_pipeline_gap()`** is applied separately, specifying the number of copies to make of each interval, the re-sampling frequency applied in `1_read_data.py`, and the minimum and maximum proportion of data to remove from each artificially gapped interval.

        1. Make multiple copies of inputs (list of dataframes)
            1. Transform inputs for outputting (**array of arrays**)
        2. Make multiple copies of outputs (list of arrays)
            1. Transform outputs for outputting (**array of arrays**)
        3. Gap each input interval, adding a missing indicator column (list of dataframes)
        4. Re-standardise each component of each input interval (list of dataframes)
        5. Fix standardised missing indicator column for each input interval (list of dataframes)
            1. Transform gapped inputs for outputting (**array of arrays**)
            2. Output proportion missing for each interval
        6. Calculate structure functions for gapped inputs (list of arrays)
            1. Transform gapped outputs for outputting (**array of arrays**)
        7. Fill missing values with zeroes in a copy of the gapped intervals (list of dataframes)
            1. Transform filled inputs for outputting (**array of arrays + array of flat arrays**)
        8. Calculate structure functions for filled inputs (list of arrays)
            1. Transform filled outputs for outputting (**array of arrays**)
        9. Linearly interpolate missing values in a copy of the gapped intervals (list of dataframes)
            1. Transform inputs for outputting (**array of arrays + array of flat arrays**)
        10. Calculate structure functions for interpolated inputs (list of arrays)
            1. Transform interpolated outputs for outputting (**array of arrays**)

2.1. Review plots
    - `results/…example_input_raw.png`
    - `results/…example_input_std.png`
    - `results/…test_preprocessed_plots.png`
2.2.. Create a folder for the results of this model: `mkdir results/date/mod_#`
3.1. Update `3_train_neural_net.py` with the new model number and adjust the hyperparameters as needed
3.2. Update `4_plot_predictions.py` with the new model number
3.3. `sbatch 3_singularity_submit.sh` *Do not run this with fewer than 3GB of memory requested or when in the `galaxenv` environment*
    
3.4. Review `3_train_neural_net.out`

    `3_train_neural_net.py` does the following:
    1. Load n x 40,000 training and test inputs, including MMS test
    2. Load n x 2000 training and test outputs, including MMS test
    3. Train model (feed-forward ANN/MLP). The number of nodes in the output layer is equal to the length each structure function. As well as the number of nodes, we specify the dropout layers, optimizer, learning rate, loss function, validation split, and number of epochs to train for.
    4. Output test predictions
    5. Output training and validation loss curve
    6. Output training and validation losses, and test loss
        - `psp_outputs_test_predictions`
        - `psp_outputs_test_predictions_loss`

4. Produce plots of a sample of true vs. predicted test **SHOULD BE VALIDATION** outputs with `python 4_plot_predictions.py`
4.1. Review plots in `results/date/mod_#` to see how well the model is performing on unseen data
4.2. Add model statistics (and plots if needed) to Results word doc
4.3. Repeat 5-12 until a good model is found
4.4. Download test data files and model results to local computer
5. Run `05_results.py` to produce final plots and statistics
    - For PSP and MMS:
        1. Table with one row for every interval:
            1. Amount missing
            2. MSE and MAPE for each of original, gapped, filled, lint, and predicted curves, compared with original curve
        2. Regression analysis using above table:
            1. Correlations between missingness against all other columns
            2. Regression outputs of missingness against all other columns
    - Scatterplots of missingness against all other columns