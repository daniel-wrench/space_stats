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
1. `sbatch 1_batch_job.sh` *Only have to do this once, depending on the size, number, and frequency of original unique intervals you want* **Takes about 4min with 20 CPUs and 10G per CPU.**

    `1_process_data.py` does the following:
    1. Load data from CDF files, get time and date column formatted correctly
    2. Re-sample dataframe to correct freq (dataframe)
    3. Return the amount of missing data, both before and after re-sampling
    4. Output the final tidy data as pkl files

2. `python 2_batch_job.sh` *Only have to do this once, depending on the number of duplicate intervals to make, how much to gap each one, and what proportion of data to use for test set.* **Takes about 20min with 10 CPUs and 50G per CPU.**
    
    This should be run in the cluster using a bash script. For now using `srun` produces a `MemoryError`. The next step will be to try either converting from float64 to float32 in the python script, *or* writing a `.sh` file according to Tulasi’s example to submit to the cluster. Using the `.sh` job I was able to produce 5 x 156 copies. This only took 1.5 seconds per interval, and would not work when running in interactive mode. At minimum, 8 is too much for the `2_singularity_submit.sh` job, with 64 cpus per task and 3G mem per CPU

    `2_process_data.py` takes the .pkl data (currently majority PSP) applies two major functions. **`mag_interval_pipeline_split()`** specifies the length of the data and the number of intervals to split it into, and the proportion to set aside for testing. This function splits the dataset into standardised intervals, then groups them into a training and test set, both of which are lists of dataframes. The intermediate outputs are a plot and the summary statistics of the first interval, before and after standardisation, and the dimensions of the final outputs. *The arguments I have specified to this function are to separate out 80% of input-output pairs for training, 20% for testing, for PSP. For MMS, use 100% of intervals for testing.*

    To each of these sets, the second major function **`mag_interval_pipeline_gap()`** is applied separately, specifying the number of copies to make of each interval, the re-sampling frequency applied in `1_read_data.py`, and the minimum and maximum proportion of data to remove from each artificially gapped interval. 

    This function copies the inputs the specified number of items to create `clean_inputs_list`, then initialises several empty lists. It then loops over every interval in this list and
    - removes 3-20 gaps according to the % specified, normalising the result and saving it to `gapped_inputs_list`.
    - mean (0) imputes the gapped input, saving it to `filled_inputs_list`.
    - linearly interpolates the gapped input, saving it to `lint_inputs_list`.

    The function then calculates the structure functions for each input interval in each of these lists, saving the results to equivalent `..._outputs_lists`.

    After this, an intermediate output of a plot of each input and output version of an interval and one of its copies is produced.

    Next, the function `prepare_array_for_output()` is called on each input list, which converts the lists of dataframes into three arrays of vectors: 4-dimensional vectors, 1-d vectors including the missing indicator vector, and 1-d vectors excluding the missing indicator vector. (Only the first of these are output for un-gapped and un-filled input lists.)'

    The simple 1-d outputs are prepared for output simply using the function `np.array(list)`.

    The dimensions of these final arrays are output as intermediate output, and then these arrays are saved.

3. Review plots
    - `results/…example_input_raw.png`
    - `results/…example_input_std.png`
    - `results/…test_preprocessed_plots.png`

4. Create a folder for the results of this model: `mkdir results/date/mod_#`

5. Update `3_train_neural_net.py` with the new model number and adjust the hyperparameters as needed

6. Update `4_plot_predictions.py` with the new model number

7. `sbatch 3_singularity_submit.sh` *Do not run this with fewer than 3GB of memory requested or when in the `galaxenv` environment*
    
8. Review `3_train_neural_net.out`

    `3_train_neural_net.py` does the following:
        1. Load n x 40,000 training and test inputs, including MMS test
        2. Load n x 2000 training and test outputs, including MMS test
        3. Train model (feed-forward ANN/MLP). The number of nodes in the output layer is equal to the length each structure function (2000). As well as the number of nodes, we specify the dropout layers, optimizer, learning rate, loss function, validation split, and number of epochs to train for.
        4. Output test predictions
        5. Output training and validation loss curve
        6. Output training and validation losses, and test loss
            - `psp_outputs_test_predictions`
            - `psp_outputs_test_predictions_loss`.

4. Produce plots of a sample of true vs. predicted test **SHOULD BE VALIDATION** outputs with `python 4_plot_predictions.py`
5. Review plots in `results/date/mod_#` to see how well the model is performing on unseen data
6. Add model statistics (and plots if needed) to Results word doc
7. Repeat 5-12 until a good model is found
8. Download test data files and model results to local computer
9. Run `05_results.py` to produce final plots and statistics
    - For PSP and MMS:
        1. Table with one row for every interval:
            1. Amount missing
            2. MSE and MAPE for each of original, gapped, filled, lint, and predicted curves, compared with original curve
        2. Regression analysis using above table:
            1. Correlations between missingness against all other columns
            2. Regression outputs of missingness against all other columns
    - Scatterplots of missingness against all other columns