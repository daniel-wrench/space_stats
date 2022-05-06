# Neural Nets for Space Stats

This code was used to explore solar wind data and implement a study of using neural networks (as well as simpler methods) to estimate structure functions of the solar wind from time series with gaps. Compare neural network predictions with mathematical structure function calculated from:
- Gapped interval
- Mean-imputed interval
- Linearly interpolated interval

## AIM

Aiming to predict structure functions directly from solar wind magnetic field intervals with gaps using a neural network trained on many input-output pairs with artificial gaps in order to attempt to make the network robust to these data gaps (‘noise’).

## NEXT STEPS

Put `3_train_neural_net.py` and `keras_tuner.py` side-by-side and implement hyperparameter tuning. Then, run this locally, before beefing it up for Raapoi.

## CURRENT DATA USED

2 intervals from PSP (`psp_fld_l2_mag_rtn_...cdf`)
- 2018-11-01 - 2018-11-17 = 1,950,000 continuous observations
- 2018-11-21 - 2018-11-30 = 1,150,000 continuous observations

**= 310 intervals of length 10,000**

2 intervals from MMS-1 (`mms1_fgm_brst_l2_...cdf`
- 2017-12-26 06:12:43 - 06:49:53 = 290,000 continuous observations
- 2018-01-08 04:00:03 - 04:57:33 = 440,000 continuous observations

**= 29 + 44 = 73 intervals of length 10,000**

## BUGS

- `1_read_data.py`, line 268: The version of `pandas` in Raapoi does not like this line, nor will it read the pkl that this has been applied to if I upload the file from my local machine. For now I am just not using this file (`mms4_df_1.pkl`) in `2_process_data.py`.
- `2_process_data.py`: The MMS_example_input... plots are not correct. Something to do with `plt.show()` or equivalent functions. 

### HELPER FUNCTIONS

- `data_import_funcs.py`:  Contains functions for importing CDF files and outputting dated pandas DataFrames.
- `remove_obs_funcs.py`: Contains functions to remove data from dataframes
- `calculate_stats.py`: Contains a function called plot_results() which is used for visualising the predictions of the neural network on the test data
- `TurbAn`: contains fast Fortran code for calculating the structure function

### SET-UP
See also: 
- `Research/Computers and ML/hpc_cheat_sheet.pdf`
- [Rāpoi documentation](https://vuw-research-computing.github.io/raapoi-docs/)

1. Install Oracle VM VirtualBox and create a Linux VM with 20GB disk space. Also install Singularity according to the code provided [here](https://sylabs.io/guides/3.0/user-guide/installation.html). _At the point where installing dep for Go, this would not work, so I used the sudo command found [here](https://github.com/golang/dep/cmd/dep)_.
2. Build a Singularity Tensorflow container in Linux on local computer. I had to do this using Linux on a Virtual Machine, following the instructions from [the documentation](https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example). It is a large file (1.2GB) that takes a few minutes to build. (There are more tips [here](https://clusterdeiguide.readthedocs.io/en/latest/SingularityExamples.html))
3. Connect to the cluster with ssh username@clustername (raapoi.vuw.ac.nz). _This may require connecting to the VUW vpn (`vpn.wgtn.ac.nz`) through Cisco. The best interface on a Windows computer is MobaXTerm - if you are on Linux simply type sftp://username@raapoi.vuw.ac.nz into the Dolphin file explorer._
4. Use `sftp` to copy across `tensor.sif`, as well as the `bin` and `galaxenv` repositories (these allow you to access a virtual environment).
5. Use `lftp` to download NASA datasets from SPDF (`https://spdf.gsfc.nasa.gov/pub/data`) into the `data_raw` folder.

### SCRIPT EXECUTION PROCESS & PSEUDO-CODE

The scripts can be run inside or outside of Rapoi. Currently scripts 1-3 have corresponding batch submission files (.sh) that allows them to be submitted as jobs to the Rapoi cluster.

All of the following is to be run in the Rāpoi terminal.
View that state of cluster jobs with `vuw-myjobs`.

0. `source ~/bin/python_env` *if you are running anything in the terminal using `ipython --pylab`*
1. `sbatch 1_batch_job.sh` *Only have to do this once, depending on the size, number, and frequency of original unique intervals you want* **Takes about 8min with 6 CPUs and 15G per CPU. Requires > 20GB total and/or > 10GB per CPU** 

    `1_process_data.py` does the following:
    1. Load data from CDF files and get the time and date column formatted correctly: returns time series dataframe of magnetic field components
    2. Re-sample dataframe to correct freq to get a consistent 13-14 correlation lengths across spacecraft/physical systems
        - For PSP data, resample to 0.75s frequency (correlation time = 500s)
        - For MMS, resample to 0.008s (correlation time = 6s)
        - (For Wind, resample to 5s (correlation time = 1 hour)
    4. Return the amount of missing data, both before and after re-sampling
    5. Output the final tidy data as pkl files

2. `sbatch 2_batch_job.sh` *Only have to do this once, depending on the number of duplicate intervals to make, how much to gap each one, and what proportion of data to use for test set.* **Takes about 30min with 10 CPUs and 5G per CPU.**

    `2_process_data.py` takes the .pkl data (currently majority PSP) applies two major functions. **`mag_interval_pipeline_split()`** specifies the length of the data and the number of intervals to split it into, and the proportion to set aside for testing. This function splits the dataset into standardised intervals (mean 0 and standard deviation 1), then groups them into a training and test set, both of which are lists of dataframes. The intermediate outputs are a plot and the summary statistics of the first interval, before and after standardisation, and the dimensions of the final outputs. *The arguments I have specified to this function are to separate out 80% of input-output pairs for training, 20% for testing, for PSP. For MMS, use 100% of intervals for testing.*

    To each of these sets, the second major function **`mag_interval_pipeline_gap()`** is applied separately, specifying the number of copies to make of each interval, the re-sampling frequency applied in `1_read_data.py`, and the minimum and maximum proportion of data to remove from each artificially gapped interval. 

    This function copies the inputs the specified number of items to create `clean_inputs_list`, then initialises several empty lists. It then loops over every interval in this list and
    - removes 3-20 gaps according to the % specified, normalising the result and saving it to `gapped_inputs_list`.
    - mean (0) imputes the gapped input, saving it to `filled_inputs_list`.
    - linearly interpolates the gapped input, saving it to `lint_inputs_list`.

    The function then calculates the structure functions for each input interval in each of these lists, saving the results to equivalent `..._outputs_lists`.

    After this, an intermediate output of a plot of each input and output version of an interval and one of its copies is produced.

    Next, the function `prepare_array_for_output()` is called on each input list, which converts the lists of dataframes into three arrays of vectors: 4-dimensional vectors, 1-d vectors including the missing indicator vector, and 1-d vectors excluding the missing indicator vector. (Only the first of these are output for un-gapped and un-filled input lists.)'

    The simple 1-d outputs are prepared for output simply using the function `np.array(list)`.

    The dimensions of these final numpy arrays are output as intermediate output, and then these arrays are saved.

3. Review plots
    - `results/…example_input_raw.png`
    - `results/…example_input_std.png`
    - `results/…test_preprocessed_plots.png`

4. Create a folder for the results of this model: `mkdir results/date/mod_#`

5. Update `3_train_neural_net.py` with the new model number and adjust the hyperparameters as needed

6. Update `4_plot_predictions.py` with the new model number

7. `sbatch 3_batch_job.sh` *Requires > 2 x 8GB CPUs. Do not run when in the `galaxenv` environment.* `3_train_neural_net.py` does the following:
   1. Load n x 40,000 training and test inputs, including MMS test
   2. Load n x 2000 training and test outputs, including MMS test
   3. Define these as Tensorflow objects
   4. Train model (feed-forward ANN/MLP). The number of nodes in the output layer is equal to the length each structure function (2000). As well as the number of nodes, we specify the dropout layers, optimizer, learning rate, loss function, validation split, and number of epochs to train for.
   5. Output test predictions
   6. Output training and validation loss curve
   7. Output training and validation losses, and test loss
        - `psp_outputs_test_predictions`
        - `psp_outputs_test_predictions_loss`.

8. Review `3_train_neural_net.out`
4. Produce plots of a sample of true vs. predicted test **SHOULD BE VALIDATION** outputs with `python 4_plot_predictions.py`
5. Review plots in `results/date/mod_#` to see how well the model is performing on unseen data
6. Add model statistics (and plots if needed) to Results word doc
7. Repeat 5-11 until a good model is found
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
