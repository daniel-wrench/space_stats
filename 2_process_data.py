
##############################################################################

# DATA PROCESSING PART 2: PREPARING INTERVALS FOR NEURAL NETWORK APPROXIMATION

###################### Daniel Wrench, April 2022 #########################

# NEXT STEPS: Adjust variable names used when applying convert_df_list_to_arrays(): no longer returning 3 objects

# This is submitted as a job to the Raapoi cluster via a .sh script
# The outputs are fed into Part 3: 3_train_neural_net.py, a Tensorflow script to train a neural network model

##############################################################################

# Loading packages, including those on Git in ~bin/python_env

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_import_funcs as data_import
import calculate_stats as calcs
import remove_obs_funcs as removal
import TurbAn.Analysis.TimeSeries.OLLibs.F90.ftsa as ftsa
import random
import datetime
import matplotlib
matplotlib.use('Agg')

##############################################################################

# Helper functions


def calc_strfn(input_intervals, dt):
    """Calculate the 3D second-order structure function up to a lag of 20% of the input interval length
    
    Inputs:
    - input_intervals: list of dataframes, each with 3 columns, one for each vector component
    - dt: time between observations, in seconds"""
    sfs = []

    for interval in input_intervals:
        r, s = ftsa.strfn_vec(ax=interval.iloc[:, 0],
                              ay=interval.iloc[:, 1],
                              az=interval.iloc[:, 2],
                              lags=range(
                                  0, round(0.2*len(input_intervals[1]))),
                              orders=[2],  # Second order structure function
                              dt=dt)  
        sfs.append(s[0])
    return sfs


def convert_df_list_to_arrays(dataset):
    """Take a list of dataframes, each with a time index and four columns (3 vector components and missing indicator), and converts them into an array of arrays 

    Returns a 3D array (3 components)"""

    list_of_vectors = []
    for i in range(len(dataset)):
        vector = []
        for j in range(dataset[0].shape[1]):
            vec_comp = dataset[i].iloc[:, j].to_numpy()
            vector.append(vec_comp)
        list_of_vectors.append(vector)
    array_of_vectors = np.array(list_of_vectors)

    return array_of_vectors

################################################################################

# First of two major pipeline functions


def mag_interval_pipeline_split(df_list, dataset_name, n_values_list, n_subsets_list, validate_size, test_size):
    """Takes a time series and splits it into many intervals, normalises them,
    then groups them into a training, validation and test set"""

    print("SPLITTING, NORMALISING AND SHUFFLING " + dataset_name + " MAG FIELD DATA\n")

    inputs_list_raw = np.split(
        df_list[0][:n_values_list[0]], n_subsets_list[0])
    # Recent additional feature that allows multiple dfs to be imported, useful for when using
    # multiple non-adjacent intervals
    if len(df_list) > 1:
        for i in range(1, len(df_list)):
            inputs_list_raw_next = np.split(
                df_list[i][:n_values_list[i]], n_subsets_list[i])
            inputs_list_raw.extend(inputs_list_raw_next)

    # List comprehension! Better than for loops!
    inputs_list = [calcs.normalize(i) for i in inputs_list_raw]

    # Intermediate output: check standardisation procedure has worked
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axs[0].plot(inputs_list_raw[0])
    axs[0].set_title("Clean interval pre-normalisation")
    axs[0].set_xticks([])
    axs[1].plot(inputs_list[0])
    axs[1].set_title("Clean interval post-normalisation")
    axs[1].set_xticks([])
    plt.savefig("results/" + dataset_name + "_std_check_plot.png")
    plt.clf()

    random.seed(5)

    shuffler = np.random.permutation(len(inputs_list))
    inputs_list_shuffled = [inputs_list[i] for i in shuffler]

    if test_size == 1:

        inputs_test = inputs_list_shuffled
        print("\nInput test data dimensions:", len(inputs_test), 'x', inputs_test[0].shape)
        return inputs_test

    else: 
        train_size = 1 - validate_size - test_size

        train_val_boundary = int(train_size*len(inputs_list_shuffled))
        val_test_boundary = int((train_size+validate_size)*len(inputs_list_shuffled))
        inputs_train = inputs_list_shuffled[:train_val_boundary]
        inputs_validate = inputs_list_shuffled[train_val_boundary:val_test_boundary]
        inputs_test = inputs_list_shuffled[val_test_boundary:]

        print("Means of first interval pre-normalisation:" + str(round(inputs_list_raw[0].mean(), 2)))
        print("\nStandard deviations of first interval pre-normalisation:"+ str(round(inputs_list_raw[0].std(), 2)))

        print("\n\nMeans of first interval post-normalisation:" + str(round(inputs_list[0].mean(), 2)))
        print("\nStandard deviations of post interval pre-normalisation:" + str(round(inputs_list[0].std(), 2)))

        print("\n\nInput training data dimensions:", len(inputs_train), 'x', inputs_train[0].shape)
        print("\nInput validation data dimensions:", len(inputs_validate), 'x', inputs_validate[0].shape)
        print("\nInput test data dimensions:", len(inputs_test), 'x', inputs_test[0].shape)

        return (inputs_train, inputs_validate, inputs_test)


# Second of two major pipeline functions

def mag_interval_pipeline_gap(
        inputs_list,
        dataset_name,
        n_copies,
        freq,
        dt,
        min_removal_percent,
        max_removal_percent):
    """Takes a set of intervals and copies each interval multiple times.
    Each interval in the new larger set has chunks removed, which is then both filled in with zeroes and linearly interpolated.
    The structure function is then calculated for each version of each interval"""

    random.seed(5)

    clean_inputs_list = inputs_list * n_copies

    gapped_inputs_list = []
    filled_inputs_0_list = []
    filled_inputs_9_list = []
    lint_inputs_list = []
    prop_removed = np.array([])

    for input in clean_inputs_list:

        gapped_input_raw = removal.remove_chunks_df(input,
                                                    random.randint(
                                                        min_removal_percent,
                                                        max_removal_percent)/100,
                                                    random.randint(3, 20),
                                                    0.1,
                                                    missing_ind_col=False)

        # The data is already at the right frequency, but this line resamples to give us the NAs we want 
        gapped_input = gapped_input_raw.resample(freq).mean()

        # Saving the proportion removed to an array
        prop_removed = np.append(prop_removed, gapped_input.iloc[:,0].isnull().sum()/len(gapped_input))

        # Re-normalising the gapped interval
        gapped_input_std = calcs.normalize(gapped_input)

        # IF USING MASK: 
        # Correcting for standardisation of missing column
        #gapped_input_std.iloc[:, 3] = gapped_input.iloc[:, 3]

        # Save standardised, gapped inputs to a list for outputting
        gapped_inputs_list.append(gapped_input_std)

        # Mean (0) imputing the artificially gapped inputs. This is because we need placeholder values for the network, and as an alternative to using a network in the first place
        filled_input_0 = gapped_input_std.fillna(0)
        filled_inputs_0_list.append(filled_input_0)

        # Alternatively, using 99990000000 as a placeholder value. We don't calculate the sfn mathematically from these, as we do for the mean-imputed version above
        filled_input_9 = gapped_input_std.fillna(999999)
        filled_inputs_9_list.append(filled_input_9)

        # Linear interpolating the artificially gapped inputs, as another alternative
        lint_input = gapped_input_std.interpolate()
        lint_inputs_list.append(lint_input)

    clean_outputs_list = calc_strfn(clean_inputs_list, dt)
    gapped_outputs_list = calc_strfn(gapped_inputs_list, dt)
    filled_outputs_0_list = calc_strfn(filled_inputs_0_list, dt)
    lint_outputs_list = calc_strfn(lint_inputs_list, dt)

    # Intermediate output: plot of each input and output version of an interval
    # and of its copies

    ind = len(inputs_list)

    fig, axs = plt.subplots(10, 2, figsize=(10, 32))

    axs[0, 0].plot(clean_inputs_list[0])
    axs[1, 0].plot(clean_inputs_list[ind])
    axs[0, 1].plot(clean_outputs_list[0])
    axs[1, 1].plot(clean_outputs_list[ind])
    axs[0, 0].set_title("Copies of clean interval")
    axs[0, 1].set_title("Corresponding structure functions")
    axs[0, 0].set_xticks([])
    axs[1, 0].set_xticks([])

    axs[2, 0].plot(gapped_inputs_list[0])
    axs[3, 0].plot(gapped_inputs_list[ind])
    axs[2, 1].plot(gapped_outputs_list[0])
    axs[3, 1].plot(gapped_outputs_list[ind])
    axs[2, 0].set_title("Gapped copies of interval")
    axs[2, 0].set_xticks([])
    axs[3, 0].set_xticks([])

    axs[4, 0].plot(filled_inputs_0_list[0])
    axs[5, 0].plot(filled_inputs_0_list[ind])
    axs[4, 1].plot(filled_outputs_0_list[0])
    axs[5, 1].plot(filled_outputs_0_list[ind])
    axs[4, 0].set_title("Zero-filled copies of interval")
    axs[4, 0].set_xticks([])
    axs[5, 0].set_xticks([])

    axs[6, 0].plot(filled_inputs_9_list[0])
    axs[7, 0].plot(filled_inputs_9_list[ind])
    axs[6, 0].set_title("Huge-filled copies of interval (sfns not calculated)")
    axs[6, 0].set_xticks([])
    axs[7, 0].set_xticks([])

    axs[8, 0].plot(lint_inputs_list[0])
    axs[9, 0].plot(lint_inputs_list[ind])
    axs[8, 1].plot(lint_outputs_list[0])
    axs[9, 1].plot(lint_outputs_list[ind])
    axs[8, 0].set_title("Interpolated copies of interval")
    axs[8, 0].set_xticks([])
    axs[9, 0].set_xticks([])

    #fig.suptitle('Validating pre-processing')
    plt.savefig("results/" + dataset_name + "_preprocessed_plots.png")
    plt.clf()

    clean_inputs = convert_df_list_to_arrays(clean_inputs_list)
    # cis = clean_inputs.shape
    # np.save(file = 'data_processed/' + dataset_name + 'clean_inputs', arr = psp_clean_inputs_train)
    # del(clean_inputs)

    gapped_inputs = convert_df_list_to_arrays(gapped_inputs_list)
    filled_inputs_0 = convert_df_list_to_arrays(filled_inputs_0_list)
    filled_inputs_9 = convert_df_list_to_arrays(filled_inputs_9_list)
    lint_inputs = convert_df_list_to_arrays(lint_inputs_list)

    clean_outputs = np.array(clean_outputs_list)
    gapped_outputs = np.array(gapped_outputs_list)
    filled_outputs_0 = np.array(filled_outputs_0_list)
    lint_outputs = np.array(lint_outputs_list)

    print("\nUnique input dimensions: \n",
          len(inputs_list), 'x', inputs_list[0].shape)

    print("\nFinal input dimensions:",
          "\n  Clean:", clean_inputs.shape,
          "\n  Gapped:", gapped_inputs.shape,
          "\n  Filled with 0s:", filled_inputs_0.shape,
          "\n  Filled with 9s:", filled_inputs_9.shape,
          "\n  Lint:", lint_inputs.shape)

    print("\nFinal output dimensions:",
          "\n  Clean:", clean_outputs.shape,
          "\n  Gapped:", gapped_outputs.shape,
          "\n  Filled:", filled_outputs_0.shape,
          "\n  Lint:", lint_outputs.shape)

    return (clean_inputs,
            clean_outputs,
            gapped_inputs,
            gapped_outputs,
            filled_inputs_0,
            filled_inputs_9,
            filled_outputs_0,
            lint_inputs,
            lint_outputs,
            prop_removed
            )

##############################

# PSP (used for training and testing the neural net)


print("\nTIME: ", datetime.datetime.now())

# Loading the data
psp_df_1 = pd.read_pickle("data_processed/psp/psp_df_1.pkl")
psp_df_2 = pd.read_pickle("data_processed/psp/psp_df_2.pkl")

# Splitting into training and test sets
(psp_inputs_train_list, psp_inputs_validate_list, psp_inputs_test_list) = mag_interval_pipeline_split(
                                                     df_list=[psp_df_1, psp_df_2],
                                                     dataset_name="psp",
                                                     # CORRECT TO 1950000, 1150000 FOR PROPER RUN
                                                     n_values_list=[1950000, 1150000], 
                                                     n_subsets_list=[1950000/10000, 1150000/10000],
                                                     validate_size=0.1,
                                                     test_size=0.2)

# Duplicating, gapping, and calculating structure functions for training set

print("\n\nPROCESSING PSP TRAINING DATA \n")

(psp_clean_inputs_train,
 psp_clean_outputs_train,
 psp_gapped_inputs_train,
 psp_gapped_outputs_train,
 psp_filled_inputs_0_train,
 psp_filled_inputs_9_train,
 psp_filled_outputs_0_train,
 psp_lint_inputs_train,
 psp_lint_outputs_train,
 psp_gapped_inputs_train_prop_removed
 ) = mag_interval_pipeline_gap(
    psp_inputs_train_list,
    "psp_train",
    # CORRECT TO 10 FOR PROPER RUN
    n_copies=10,
    freq='0.75S',
    dt=0.75,
    min_removal_percent=0,
    max_removal_percent=50)

# Saving PSP training outputs
# CHANGE SO AS TO USE FEWER LINES OF CODE
np.save(file='data_processed/psp/psp_clean_inputs_train',
        arr=psp_clean_inputs_train)
np.save(file='data_processed/psp/psp_clean_outputs_train',
        arr=psp_clean_outputs_train)
np.save(file='data_processed/psp/psp_gapped_inputs_train',
        arr=psp_gapped_inputs_train)
np.save(file='data_processed/psp/psp_gapped_outputs_train',
        arr=psp_gapped_outputs_train)
np.save(file='data_processed/psp/psp_filled_inputs_0_train',
        arr=psp_filled_inputs_0_train)
np.save(file='data_processed/psp/psp_filled_outputs_0_train',
        arr=psp_filled_outputs_0_train)
np.save(file='data_processed/psp/psp_filled_inputs_9_train',
        arr=psp_filled_inputs_9_train)
np.save(file='data_processed/psp/psp_lint_inputs_train',
        arr=psp_lint_inputs_train)
np.save(file='data_processed/psp/psp_lint_outputs_train',
        arr=psp_lint_outputs_train)
np.save(file='data_processed/psp/psp_gapped_inputs_train_prop_removed',
        arr=psp_gapped_inputs_train_prop_removed)

print("\nFINISHED PROCESSING PSP TRAINING DATA \n")

print("\nTIME: ", datetime.datetime.now())


print("\n\nPROCESSING PSP VALIDATION DATA \n")

(psp_clean_inputs_validate,
 psp_clean_outputs_validate,
 psp_gapped_inputs_validate,
 psp_gapped_outputs_validate,
 psp_filled_inputs_0_validate,
 psp_filled_inputs_9_validate,
 psp_filled_outputs_0_validate,
 psp_lint_inputs_validate,
 psp_lint_outputs_validate,
 psp_gapped_inputs_validate_prop_removed
 ) = mag_interval_pipeline_gap(
    psp_inputs_validate_list,
    "psp_validate",
    n_copies=5,
    freq='0.75S',
    dt=0.75,
    min_removal_percent=0,
    max_removal_percent=50)

# Saving PSP validation outputs
np.save(file='data_processed/psp/psp_clean_inputs_validate',
        arr=psp_clean_inputs_validate)
np.save(file='data_processed/psp/psp_clean_outputs_validate',
        arr=psp_clean_outputs_validate)
np.save(file='data_processed/psp/psp_gapped_inputs_validate',
        arr=psp_gapped_inputs_validate)
np.save(file='data_processed/psp/psp_gapped_outputs_validate',
        arr=psp_gapped_outputs_validate)
np.save(file='data_processed/psp/psp_filled_inputs_0_validate',
        arr=psp_filled_inputs_0_validate)
np.save(file='data_processed/psp/psp_filled_outputs_0_validate',
        arr=psp_filled_outputs_0_validate)
np.save(file='data_processed/psp/psp_filled_inputs_9_validate',
        arr=psp_filled_inputs_9_validate)
np.save(file='data_processed/psp/psp_lint_inputs_validate',
        arr=psp_lint_inputs_validate)
np.save(file='data_processed/psp/psp_lint_outputs_validate',
        arr=psp_lint_outputs_validate)
np.save(file='data_processed/psp/psp_gapped_inputs_validate_prop_removed',
        arr=psp_gapped_inputs_validate_prop_removed)

print("\nFINISHED PROCESSING PSP VALIDATION DATA \n")

print("\nTIME: ", datetime.datetime.now())


# Duplicating, gapping, and calculating structure functions for test set

print("\n\nPROCESSING PSP TEST DATA \n")

(psp_clean_inputs_test,
 psp_clean_outputs_test,
 psp_gapped_inputs_test,
 psp_gapped_outputs_test,
 psp_filled_inputs_0_test,
 psp_filled_inputs_9_test,
 psp_filled_outputs_0_test,
 psp_lint_inputs_test,
 psp_lint_outputs_test,
 psp_gapped_inputs_test_prop_removed
 ) = mag_interval_pipeline_gap(
    psp_inputs_test_list,
    "psp_test",
    n_copies=5,
    freq='0.75S',
    dt=0.75,
    min_removal_percent=0,
    max_removal_percent=95)

# Saving PSP testing outputs
# CHANGE SO AS TO USE FEWER LINES OF CODE
np.save(file='data_processed/psp/psp_clean_inputs_test',
        arr=psp_clean_inputs_test)
np.save(file='data_processed/psp/psp_clean_outputs_test',
        arr=psp_clean_outputs_test)
np.save(file='data_processed/psp/psp_gapped_inputs_test',
        arr=psp_gapped_inputs_test)
np.save(file='data_processed/psp/psp_gapped_outputs_test',
        arr=psp_gapped_outputs_test)
np.save(file='data_processed/psp/psp_filled_inputs_0_test',
        arr=psp_filled_inputs_0_test)
np.save(file='data_processed/psp/psp_filled_outputs_0_test',
        arr=psp_filled_outputs_0_test)
np.save(file='data_processed/psp/psp_filled_inputs_9_test',
        arr=psp_filled_inputs_9_test)
np.save(file='data_processed/psp/psp_lint_inputs_test',
        arr=psp_lint_inputs_test)
np.save(file='data_processed/psp/psp_lint_outputs_test',
        arr=psp_lint_outputs_test)
np.save(file='data_processed/psp/psp_gapped_inputs_test_prop_removed',
        arr=psp_gapped_inputs_test_prop_removed)


print("\nFINISHED PROCESSING PSP TEST DATA \n")

##################################################################################################

# MMS

print("\nTIME: ", datetime.datetime.now())

# Loading the data

mms1_df_1 = pd.read_pickle("data_processed/mms/mms1_df_1.pkl")
mms1_df_2 = pd.read_pickle("data_processed/mms/mms1_df_2.pkl")

# mms2_df_1 = pd.read_pickle("data_processed/mms/mms2_df_1.pkl")
# mms2_df_2 = pd.read_pickle("data_processed/mms/mms2_df_2.pkl")

# mms3_df_1 = pd.read_pickle("data_processed/mms/mms3_df_1.pkl")
# mms3_df_2 = pd.read_pickle("data_processed/mms/mms3_df_2.pkl")

# MISSING DATA HERE
# mms4_df_1 = pd.read_pickle("data_processed/mms/mms4_df_1.pkl") 

# mms4_df_2 = pd.read_pickle("data_processed/mms/mms4_df_2.pkl")

#################

# GET 100% OF MMS DATA TO TEST ON

mms_inputs_test_list = mag_interval_pipeline_split(
    df_list=[
            mms1_df_1, 
            mms1_df_2
            ],
    dataset_name="mms",
    # CORRECT TO [290000, 440000] FOR PROPER RUN
    n_values_list=[290000, 440000],
    n_subsets_list=[290000/10000, 440000/10000],
    validate_size=0,
    test_size=1)

#print("\n\nPROCESSING MMS TRAINING DATA \n")

# (mms_clean_inputs_train,
#  mms_clean_outputs_train,
#  mms_gapped_inputs_train,
#  mms_gapped_outputs_train,
#  mms_filled_inputs_0_train,
#  mms_filled_inputs_0_train_flat,
#  mms_filled_inputs_9_train,
#  mms_filled_inputs_9_train_flat,
#  mms_filled_outputs_0_train,
#  mms_lint_inputs_train,
#  mms_lint_inputs_train_flat,
#  mms_lint_outputs_train,
#  mms_gapped_inputs_train_prop_removed
#  ) = mag_interval_pipeline_gap(
#     mms_inputs_train_list,
#     "mms_train",
#     n_copies=15,
#     freq='0.008S',
#     dt=0.008,
#     min_removal_percent=0,
#     max_removal_percent=50)

# # Saving mms training outputs
# # CHANGE SO AS TO USE FEWER LINES OF CODE
# np.save(file='data_processed/mms/mms_clean_inputs_train',
#         arr=mms_clean_inputs_train)
# np.save(file='data_processed/mms/mms_clean_outputs_train',
#         arr=mms_clean_outputs_train)
# np.save(file='data_processed/mms/mms_gapped_inputs_train',
#         arr=mms_gapped_inputs_train)
# np.save(file='data_processed/mms/mms_gapped_outputs_train',
#         arr=mms_gapped_outputs_train)
# np.save(file='data_processed/mms/mms_filled_inputs_0_train',
#         arr=mms_filled_inputs_0_train)
# np.save(file='data_processed/mms/mms_filled_inputs_0_train_flat',
#         arr=mms_filled_inputs_0_train_flat)
# np.save(file='data_processed/mms/mms_filled_outputs_0_train',
#         arr=mms_filled_outputs_0_train)
# np.save(file='data_processed/mms/mms_filled_inputs_9_train',
#         arr=mms_filled_inputs_9_train)
# np.save(file='data_processed/mms/mms_filled_inputs_9_train_flat',
#         arr=mms_filled_inputs_9_train_flat)
# np.save(file='data_processed/mms/mms_lint_inputs_train',
#         arr=mms_lint_inputs_train)
# np.save(file='data_processed/mms/mms_lint_inputs_train_flat',
#         arr=mms_lint_inputs_train_flat)
# np.save(file='data_processed/mms/mms_lint_outputs_train',
#         arr=mms_lint_outputs_train)
# np.save(file='data_processed/mms/mms_gapped_inputs_train_prop_removed',
#         arr=mms_gapped_inputs_train_prop_removed)

# print("\nFINISHED PROCESSING MMS TRAINING DATA \n")

# print("\nTIME: ", datetime.datetime.now())


# print("\n\nPROCESSING mms VALIDATION DATA \n")

# (mms_clean_inputs_validate,
#  mms_clean_outputs_validate,
#  mms_gapped_inputs_validate,
#  mms_gapped_outputs_validate,
#  mms_filled_inputs_0_validate,
#  mms_filled_inputs_0_validate_flat,
#  mms_filled_inputs_9_validate,
#  mms_filled_inputs_9_validate_flat,
#  mms_filled_outputs_0_validate,
#  mms_lint_inputs_validate,
#  mms_lint_inputs_validate_flat,
#  mms_lint_outputs_validate,
#  mms_gapped_inputs_validate_prop_removed
#  ) = mag_interval_pipeline_gap(
#     mms_inputs_validate_list,
#     "mms_validate",
#     n_copies=5,
#     freq='0.008S',
#     dt=0.008,
#     min_removal_percent=0,
#     max_removal_percent=50)

# # Saving mms validation outputs
# # CHANGE SO AS TO USE FEWER LINES OF CODE
# np.save(file='data_processed/mms/mms_clean_inputs_validate',
#         arr=mms_clean_inputs_validate)
# np.save(file='data_processed/mms/mms_clean_outputs_validate',
#         arr=mms_clean_outputs_validate)
# np.save(file='data_processed/mms/mms_gapped_inputs_validate',
#         arr=mms_gapped_inputs_validate)
# np.save(file='data_processed/mms/mms_gapped_outputs_validate',
#         arr=mms_gapped_outputs_validate)
# np.save(file='data_processed/mms/mms_filled_inputs_0_validate',
#         arr=mms_filled_inputs_0_validate)
# np.save(file='data_processed/mms/mms_filled_inputs_0_validate_flat',
#         arr=mms_filled_inputs_0_validate_flat)
# np.save(file='data_processed/mms/mms_filled_outputs_0_validate',
#         arr=mms_filled_outputs_0_validate)
# np.save(file='data_processed/mms/mms_filled_inputs_9_validate',
#         arr=mms_filled_inputs_9_validate)
# np.save(file='data_processed/mms/mms_filled_inputs_9_validate_flat',
#         arr=mms_filled_inputs_9_validate_flat)
# np.save(file='data_processed/mms/mms_lint_inputs_validate',
#         arr=mms_lint_inputs_validate)
# np.save(file='data_processed/mms/mms_lint_inputs_validate_flat',
#         arr=mms_lint_inputs_validate_flat)
# np.save(file='data_processed/mms/mms_lint_outputs_validate',
#         arr=mms_lint_outputs_validate)
# np.save(file='data_processed/mms/mms_gapped_inputs_validate_prop_removed',
#         arr=mms_gapped_inputs_validate_prop_removed)

# print("\nFINISHED PROCESSING MMS VALIDATION DATA \n")

# print("\nTIME: ", datetime.datetime.now())


# Duplicating, gapping, and calculating structure functions for MMS test set

print("\n\nPROCESSING MMS TEST DATA \n")

(mms_clean_inputs_test,
 mms_clean_outputs_test,
 mms_gapped_inputs_test,
 mms_gapped_outputs_test,
 mms_filled_inputs_0_test,
 mms_filled_inputs_9_test,
 mms_filled_outputs_0_test,
 mms_lint_inputs_test,
 mms_lint_outputs_test,
 mms_gapped_inputs_test_prop_removed
 ) = mag_interval_pipeline_gap(
    mms_inputs_test_list,
    "mms_test",
    n_copies=5,
    freq='0.008S',
    dt=0.008,
    min_removal_percent=0,
    max_removal_percent=95)

# Saving mms testing outputs
# CHANGE SO AS TO USE FEWER LINES OF CODE
np.save(file='data_processed/mms/mms_clean_inputs_test',
        arr=mms_clean_inputs_test)
np.save(file='data_processed/mms/mms_clean_outputs_test',
        arr=mms_clean_outputs_test)
np.save(file='data_processed/mms/mms_gapped_inputs_test',
        arr=mms_gapped_inputs_test)
np.save(file='data_processed/mms/mms_gapped_outputs_test',
        arr=mms_gapped_outputs_test)
np.save(file='data_processed/mms/mms_filled_inputs_0_test',
        arr=mms_filled_inputs_0_test)
np.save(file='data_processed/mms/mms_filled_outputs_0_test',
        arr=mms_filled_outputs_0_test)
np.save(file='data_processed/mms/mms_filled_inputs_9_test',
        arr=mms_filled_inputs_9_test)
np.save(file='data_processed/mms/mms_lint_inputs_test',
        arr=mms_lint_inputs_test)
np.save(file='data_processed/mms/mms_lint_outputs_test',
        arr=mms_lint_outputs_test)
np.save(file='data_processed/mms/mms_gapped_inputs_test_prop_removed',
        arr=mms_gapped_inputs_test_prop_removed)

print("\nTIME: ", datetime.datetime.now())

print("\nFINISHED PROCESSING MMS TEST DATA\n")

print("\nFINISHED PROCESSING ALL DATA\n")
