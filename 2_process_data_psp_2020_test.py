
##############################################################################

# DATA PROCESSING PART 2: PREPARING INTERVALS FOR NEURAL NETWORK APPROXIMATION

###################### Daniel Wrench, September 2021 #########################

# This is being run in the Raapoi terminal
# The outputs are fed into a Tensorflow script to train a neural network model

##############################################################################

# Loading packages, including those on Git in ~bin/python_env

import warnings
import space_stats.data_import_funcs as data_import
import space_stats.calculate_stats as calcs
import space_stats.remove_obs_funcs as removal
import TurbAn.Analysis.TimeSeries.OLLibs.F90.ftsa as ftsa
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##############################################################################

# Helper functions

def calc_strfn(input_intervals, dt):
    sfs = []

    for interval in input_intervals:
        r,s = ftsa.strfn_vec(ax = interval.iloc[:,0], 
                            ay = interval.iloc[:,1], 
                            az = interval.iloc[:,2], 
                            lags = range(0, round(0.2*len(input_intervals[1]))), 
                            orders = [2], 
                            dt = dt) #Second order structure function
        sfs.append(s[0])
    return sfs

#def prepare_array_for_output(dataset):
#            list_of_vectors = []
#            list_of_flat_vectors = []
#            for i in range(len(dataset)):
#                vector = np.array(dataset)[i].transpose()
#                list_of_vectors.append(vector)
#                list_of_flat_vectors.append(vector.flatten())
#            array_of_vectors = np.array(list_of_vectors)
#            array_of_flat_vectors = np.array(list_of_flat_vectors)
#            return(array_of_vectors, array_of_flat_vectors)

def prepare_array_for_output(dataset):
	list_of_vectors = []
	for i in range(len(dataset)):
		vector = []
		for j in range(dataset[0].shape[1]):
			vec_comp = dataset[i].iloc[:,j].to_numpy()
			vector.append(vec_comp)
		list_of_vectors.append(vector)
	array_of_vectors = np.array(list_of_vectors)

	list_of_flat_vectors = []
	list_of_flat_vectors_no_ind = []
	for i in range(len(array_of_vectors)):
		list_of_flat_vectors.append(array_of_vectors[i].flatten())
		list_of_flat_vectors_no_ind.append(array_of_vectors[i][0:3].flatten())

	array_of_flat_vectors = np.array(list_of_flat_vectors)
	array_of_flat_vectors_no_ind = np.array(list_of_flat_vectors_no_ind)
  
	return(array_of_vectors, array_of_flat_vectors, array_of_flat_vectors_no_ind)

################################################################################

# First of two major pipeline functions
# This one takes a time series and splits it into many intervals, then groups them into a training a test set

def mag_interval_pipeline_split(df, dataset_name, n_values, n_subsets, test_size):

    inputs_list_raw = np.split(df[:n_values], n_subsets) 

    plt.plot(inputs_list_raw[0])
    plt.title("Single clean vector interval pre-standardisation")
    plt.savefig(dataset_name + "_example_input_raw.png")
    plt.clf() #Try plt.cla() if this doesn't work

    inputs_list = [calcs.normalize(i) for i in inputs_list_raw]

    plt.plot(inputs_list[0])
    plt.title("Single clean vector interval post-standardisation")
    plt.savefig(dataset_name + "_example_input_std.png")
    plt.clf()

    print("\n Means of a clean interval post-standardisation")
    print(inputs_list[0].mean())
    print("\n Standard deviations of a clean interval post-standardisation")
    print(inputs_list[0].std())

    random.seed(5)

    shuffler = np.random.permutation(len(inputs_list))
    inputs_list_shuffled = [inputs_list[i] for i in shuffler]

    train_test_boundary = int((1-test_size)*len(inputs_list_shuffled))
    inputs_train_list = inputs_list_shuffled[:train_test_boundary]
    inputs_test_list = inputs_list_shuffled[train_test_boundary:]

    if test_size < 1:
        print("\nInput training data dimensions:", 
        len(inputs_train_list), 'x', inputs_train_list[0].shape)

    print("\nInput test data dimensions:", 
    len(inputs_test_list), 'x', inputs_test_list[0].shape)

    return (inputs_train_list, inputs_test_list)

# Second of two major pipeline functions

# This one takes a set of intervals and copies each interval multiple times
# Each interval in the new larger set has chunks removed, which is then both filled in with zeroes and linearly interpolated
# The structure function is then calculated for each version of each interval 

def mag_interval_pipeline_gap(inputs_list, dataset_name, n_copies, freq, dt, min_removal_percent, max_removal_percent):
    random.seed(5)

    clean_inputs_list = inputs_list * n_copies

    gapped_inputs_list = []
    gapped_inputs_list = []
    filled_inputs_list = []
    lint_inputs_list = []
    prop_removed = np.array([])

    for input in clean_inputs_list:
        
        gapped_input_raw = removal.remove_chunks_df(input, random.randint(min_removal_percent,max_removal_percent)/100, random.randint(3,20), 0.1)
        gapped_input = gapped_input_raw.resample(freq).mean()
        prop_removed = np.append(prop_removed, gapped_input_raw.missing.mean())

        gapped_input_std = calcs.normalize(gapped_input)
        
        # Correcting for standardisation of missing column
        gapped_input_std.iloc[:,3] = gapped_input.iloc[:,3]

        # Save standardised, gapped inputs to a list for outputting
        gapped_inputs_list.append(gapped_input_std)

        # Mean (0) imputing the artificially gapped inputs. This is because we need placeholder values for the network, and as an alternative to using a network in the first place 
        filled_input = gapped_input_std.fillna(0) 
        filled_inputs_list.append(filled_input)

        # Linear interpolating the artificially gapped inputs, as another alternative
        lint_input = gapped_input_std.interpolate() 
        lint_inputs_list.append(lint_input)

    clean_outputs_list = calc_strfn(clean_inputs_list, dt)
    gapped_outputs_list = calc_strfn(gapped_inputs_list, dt)
    filled_outputs_list = calc_strfn(filled_inputs_list, dt)
    lint_outputs_list = calc_strfn(lint_inputs_list, dt)

    ind = len(inputs_list)

    fig, axs = plt.subplots(8, 2, figsize = (10,30))

    axs[0,0].plot(clean_inputs_list[0])
    axs[1,0].plot(clean_inputs_list[ind])
    axs[0,1].plot(clean_outputs_list[0])
    axs[1,1].plot(clean_outputs_list[ind])
    axs[0,0].set_title("Copies of clean interval")
    axs[0,1].set_title("Corresponding structure functions")
    axs[1,0].set_xticks([])

    axs[2,0].plot(gapped_inputs_list[0])
    axs[3,0].plot(gapped_inputs_list[ind])
    axs[2,1].plot(gapped_outputs_list[0])
    axs[3,1].plot(gapped_outputs_list[ind])
    axs[2,0].set_title("Gapped copies of interval")
    axs[3,0].set_xticks([])

    axs[4,0].plot(filled_inputs_list[0])
    axs[5,0].plot(filled_inputs_list[ind])
    axs[4,1].plot(filled_outputs_list[0])
    axs[5,1].plot(filled_outputs_list[ind])
    axs[4,0].set_title("Filled copies of interval")
    axs[5,0].set_xticks([])

    axs[6,0].plot(lint_inputs_list[0])
    axs[7,0].plot(lint_inputs_list[ind])
    axs[6,1].plot(lint_outputs_list[0])
    axs[7,1].plot(lint_outputs_list[ind])
    axs[6,0].set_title("Interpolated copies of interval")
    axs[7,0].set_xticks([])

    fig.suptitle('Validating pre-processing')
    plt.savefig(dataset_name + "_preprocessed_plots.png")
    plt.show()

    clean_inputs = prepare_array_for_output(clean_inputs_list)[0]
    #cis = clean_inputs.shape
	 #np.save(file = 'data_processed/' + dataset_name + 'clean_inputs', arr = psp_clean_inputs_train)    
	 #del(clean_inputs)
    
    gapped_inputs = prepare_array_for_output(gapped_inputs_list)[0]
    filled_inputs, filled_inputs_flat, filled_inputs_flat_no_ind = prepare_array_for_output(filled_inputs_list)
    lint_inputs, lint_inputs_flat, lint_inputs_flat_no_ind = prepare_array_for_output(lint_inputs_list)

    clean_outputs = np.array(clean_outputs_list)
    gapped_outputs = np.array(gapped_outputs_list)
    filled_outputs = np.array(filled_outputs_list)
    lint_outputs = np.array(lint_outputs_list)

    print("\nUnique input dimensions: \n", 
    len(inputs_list), 'x', inputs_list[0].shape)

    print("\nFinal input dimensions:", 
    "\n  Clean:", clean_inputs.shape,
    "\n  Gapped:", gapped_inputs.shape, 
    "\n  Filled:", filled_inputs.shape,
    "\n  Lint:", lint_inputs.shape)

    print("\nFinal flat input dimensions:", 
    "\n  Filled:", filled_inputs_flat.shape,
    "\n  Filled flat, no indicator vector:", filled_inputs_flat_no_ind.shape,
    "\n  Lint:", lint_inputs_flat.shape,
    "\n  Lint flat, no indicator vector:", lint_inputs_flat_no_ind.shape)

    print("\nFinal output dimensions:", 
    "\n  Clean:", clean_outputs.shape, 
    "\n  Gapped:", gapped_outputs.shape, 
    "\n  Filled:", filled_outputs.shape,
    "\n  Lint:", lint_outputs.shape)

    return (clean_inputs, 
    clean_outputs,
    gapped_inputs,
    gapped_outputs,
    filled_inputs,
    filled_inputs_flat,
    filled_inputs_flat_no_ind,
    filled_outputs,
    lint_inputs,
    lint_inputs_flat,
    lint_inputs_flat_no_ind,
    lint_outputs,
    prop_removed
    )

##############################


## PSP 2020 (only used for testing final neural net)

print("\nTIME: ", datetime.datetime.now())

# Loading the data

psp_df_2020_test = pd.read_pickle("data_processed/psp/psp_df_2020_test.pkl")

# Splitting test set

(psp_df_2020_train_list, 
psp_df_2020_test_list) = mag_interval_pipeline_split(
psp_df_2020_test,
dataset_name="psp", 
n_values = 270000, 
n_subsets = 270000/10000,  
test_size = 1) 

# Duplicating, gapping, and calculating structure functions for MMS test set

print("\n\nPROCESSING PSP 2020 (TEST) DATA \n")

(psp_clean_inputs_test_2020, 
psp_clean_outputs_test_2020,
psp_gapped_inputs_test_2020,
psp_gapped_outputs_test_2020,
psp_filled_inputs_test_2020,
psp_filled_inputs_test_flat_2020,
psp_filled_inputs_test_flat_no_ind_2020,
psp_filled_outputs_test_2020,
psp_lint_inputs_test_2020,
psp_lint_inputs_test_flat_2020,
psp_lint_inputs_test_flat_no_ind_2020,
psp_lint_outputs_test_2020,
psp_gapped_inputs_test_prop_removed_2020
) = mag_interval_pipeline_gap(
    psp_df_2020_test_list, 
    "psp_test_2020",
    n_copies = 5, 
    freq = '0.75S', 
    dt = 0.75, 
    min_removal_percent = 0, 
    max_removal_percent = 95)

# Saving psp test outputs

np.save(file = 'data_processed/psp/psp_clean_inputs_test_2020', arr = psp_clean_inputs_test_2020)
np.save(file = 'data_processed/psp/psp_clean_outputs_test_2020', arr = psp_clean_outputs_test_2020)
np.save(file = 'data_processed/psp/psp_gapped_inputs_test_2020', arr = psp_gapped_inputs_test_2020)
np.save(file = 'data_processed/psp/psp_gapped_outputs_test_2020', arr = psp_gapped_outputs_test_2020)

np.save(file = 'data_processed/psp/psp_filled_inputs_test_2020', arr = psp_filled_inputs_test_2020)
np.save(file = 'data_processed/psp/psp_filled_inputs_test_flat_2020', arr = psp_filled_inputs_test_flat_2020)
np.save(file = 'data_processed/psp/psp_filled_inputs_test_flat_no_ind_2020', arr = psp_filled_inputs_test_flat_no_ind_2020)
np.save(file = 'data_processed/psp/psp_filled_outputs_test_2020', arr = psp_filled_outputs_test_2020)

np.save(file = 'data_processed/psp/psp_lint_inputs_test_2020', arr = psp_lint_inputs_test_2020)
np.save(file = 'data_processed/psp/psp_lint_inputs_test_flat_2020', arr = psp_lint_inputs_test_flat_2020)
np.save(file = 'data_processed/psp/psp_lint_inputs_test_flat_no_ind_2020', arr = psp_lint_inputs_test_flat_no_ind_2020)
np.save(file = 'data_processed/psp/psp_lint_outputs_test_2020', arr = psp_lint_outputs_test_2020)

np.save(file = 'data_processed/psp/psp_gapped_inputs_test_prop_removed_2020', arr = psp_gapped_inputs_test_prop_removed_2020)

print("\nTIME: ", datetime.datetime.now())

print("\nDONE")
