# Neural Nets for Space Stats

This code was used to explore solar wind data and implement a study of using neural networks (as well as simpler methods) to estimate structure functions of the solar wind from time series with gaps.

### CODE DESCRIPTIONS

- **data_import_funcs.py**  Contains functions for importing CDF files and outputting dated pandas DataFrames.
- **remove_obs_funcs.py** Contains functions to remove data from dataframes
- **calculate_stats.py** Contains a function called plot_results() which is used for visualising the predictions of the neural network on the test data

### SET-UP

1. Set-up a Singularity Tensorflow container in Linux on local computer. I had to do this using Linux on a Virtual Machine, following the instructions from [https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example](https://vuw-research-computing.github.io/raapoi-docs/examples/#singularitytensorflow-example)
2. Log into Raapoi (or relevant HPC cluster).
3. (Set up `bin/python_env` file to source for running commands in ipython shell.)
4. Use `sftp`to copy across `tensor.sif` file
5. Use `lftp`to download NASA datasets from SPDF into data/spacecraft folder.

### SCRIPT EXECUTION PROCESS

All of the following is to be run in the Rāpoi terminal

1. `python 1_read_data.py` *Only have to do this once, depending on the size, number, and frequency of original unique intervals you want*
2. Review plots
    1. `…example_input_raw.png`
    2. `…example_input_std.png`
3. `python 2_process_data.py` *Only have to do this once, depending on the number of duplicate intervals to make, how much to gap each one, and what proportion of data to use for test set.* 
    
    This should be run in the cluster using a bash script. For now using `srun` produces a `MemoryError`. The next step will be to try either converting from float64 to float32 in the python script, *or* writing a `.sh` file according to Tulasi’s example to submit to the cluster. Using the `.sh` job I was able to produce 5 x 156 copies. This only took 1.5 seconds per interval, and would not work when running in interactive mode. At minimum, 8 is too much for the [bash.sh](http://bash.sh/) job, with 64 cpus per task and 3G mem per CPU
    
4. Review plot:`…test_preprocessed_plots.png`
5. Create a folder for the results of this model:`mkdir results/date/mod_#`
6. Update `3_train_neural_net.py` with the new model number and adjust the hyperparameters as needed
7. Update `4_plot_predictions.py` with the new model number
8. `sbatch singularity_submit.sh`
    
    Do not run this with fewer than 3GB of memory requested or when in `galaxenv` mode
    
9. Review `3_train_neural_net.out`
10. Produce plots of a sample of true vs. predicted test outputs with`python 4_plot_predictions.py`
11. Review plots in `results/date/mod_#` to see how well the model is performing on unseen data
12. Add model statistics (and plots if needed) to Results word doc
13. Repeat 6-12 until a good model is found
14. Download test data files and model results to local computer
15. Run `05_results.py` to produce final plots and statistics

# Pseudo-code

## Pre-processing

**terminal_preprocessing.py,** to run in Rāpoi terminal

1. Load data from CDF files, get time and date column formatted correctly
2. Re-sample dataframe to correct freq (dataframe)

*Steps 1-2 take 7 minutes in the Rāpoi terminal*

1. Check for missing data
2. Run **mag_interval_pipeline_split()**

psp_inputs_train_list,

psp_inputs_test_list =

def **mag_interval_pipeline_split**(

df = psp_df,

n_values = 1950000

n_subsets = 1950000/10000

delta = 0.75,

test_size = 0.2)

1. Split dataframe into intervals of length 10,000 (list of dataframes)
2. Standardise each interval (list of dataframes)
3. Shuffle order of intervals (list of dataframes)
4. Calculate structure function for each interval (list of arrays)
5. Separate out 80% of input-output pairs for training, 20% for testing, for PSP. For MMS, use 100% of intervals for testing. Apply steps 7-14 to each set separately.

*Steps 3-4 take 1.5 minutes for 40 intervals. 3 minutes for 80 intervals. 8.5min for 200 intervals = ~ 2.5-3 seconds per interval.*

Now running for MMS too  = total of 395 intervals= 16min.

For our full set of 3460 intervals, we would expect 2.5 hours to finish.

1. ***Run **mag_interval_pipeline_gap()**

psp_clean_inputs_train,

psp_clean_outputs_train,

psp_gapped_inputs_train,

psp_gapped_outputs_train,

psp_filled_inputs_train,

psp_filled_inputs_train_flat,

psp_filled_outputs_train,

psp_lint_inputs_train,

psp_lint_inputs_train_flat,

psp_lint_outputs_train =

def **mag_interval_pipeline_gap**(

inputs_list = psp_inputs_train_list,

n_values = 1950000,

n_subsets = 195000/10000,

n_copies = 20

delta = 0.75,

freq = ‘0.75S’

)

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

## Training model

**3_train_neural_net.py**, to submit as job to Rāpoi cluster

1. Load n x 40,000 training and test inputs, including MMS test
2. Load n x 2000 training and test outputs, including MMS test
3. Train model
4. Output test predictions
5. Output training and validation loss curve
6. Output training and validation losses, and test loss

psp_outputs_test_predictions

psp_outputs_test_predictions_loss

## Evaluating model

1. With each model training, plot 9 model predictions against true curves.

## Plotting final results

1. For PSP and MMS:
    1. **Table with one row for every interval:**
        1. Amount missing
        2. MSE and MAPE for each of original, gapped, filled, lint, and predicted curves, compared with original curve
    2. Regression analysis using above table:
        1. Correlations between missingness against all other columns
        2. Regression outputs of missingness against all other columns
- Scatterplots of missingness against all other columns

---

# Old documentation (could still be relevant)

1. Run **terminal_pre-processing.py** code. (Currently, this is copied and pasted into the ipython --pylab shell. This is because we need access to the strfn() function from the TurbAn repository.)
This code takes the raw data (currently majority PSP) and outputs the following arrays:
- PSP clean inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with no gaps
- PSP gapped inputs: normalised subsets of a single magnetic field component (BR), each of length 10,000, with artificial gaps of between 10% and 40% removed in between 3 and 5 chunks. *The gapped datasets are chosen to be the final 20% of the original datasets*. These are output in both filled (0 (mean) imputed) and unfilled versions.
- PSP clean and gapped outputs: expected structure functions corresponding to each (original, ungapped) dataset. These are the second-order structure functions, and are calculated up to the lag equal to 20% of the input data length (2000 points). **Calculating these for all subsets takes around 15min.**
- PSP (gapped) math outputs: structure functions for the gapped datasets, calculated *after* gapping took place.
- Voyager gappy inputs: a small number of normalised subsets of Voyager data. *These are already very gappy.* Voyager data is just for testing - there is no "expected" output because we do not have the complete datasets
- Voyager (gapped) math outputs: structure functions for the gapped datasets
1. Run **ml_program.py** by submitting it to the cluster inside the **singularity_submit.sh** file. This also contains the CPU, memory, and time requests for the job. Currently these are set to 6 cpus-per-task and 6GB of memory.
This file does the following:
- Reads in the arrays above.
- Concatenates the clean and gapped datasets. These constitute the full data which will be used for training and testing the model. We create a *gapped_key* to keep track of which of the datasets are gapped. We then randomly select 10% of the data to be used for testing, with the remaining 90% being used for training.
- Builds and trains a feed-forward back-progragation neural network, or multi-layer perceptron (MLP). The number of nodes in the output layer is equal to the length each structure function. As well as the number of nodes, we specify the dropout layers, optimizer, learning rate, loss function, validation split, and number of epochs to train for.
- Evaluates the model by calculating the loss function on the test data.
- Outputs the following arrays:
-- test_gapped_indices: Array with a 1 if the corresponding test subset is gapped, 0 otherwise.
-- inputs_train and inputs_test: Full training and testing input data
-- ann_test_predictions and ann_test_observed: Expected and predicted structure functions for the test data
-- voyager_gapped_predictions: Predicted structure functions for the gappy Voyager data
1. Plot a series of expected and observed functions using the calc.plot_results() function in the ipython terminal
2. Visualise a select few prediction curves against the expected and mathematical curves, using the **[plotting.py](http://plotting.py/)** code
