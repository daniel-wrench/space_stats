
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import math
from remove_obs_funcs import remove_chunks_df, remove_obs_df

#Function for plotting observed curves against predicted for the test subsets

def plot_results(predictions_arr, observed_arr, math_arr, no_samples, plot_title = 'ANN predictions on test set'):

    predictions = np.load(predictions_arr)
    predicted = pd.DataFrame(predictions)
    predicted = predicted.transpose()
    
    observations = np.load(observed_arr)
    observed = pd.DataFrame(observations)
    observed = observed.transpose()
    

    # These only work for if all are gapped datasets    
    math_arr = np.load(math_arr)
    math_obs = pd.DataFrame(math_arr)
    math_obs = math_obs.transpose()
    
    observations_dataset = observed
    predictions_dataset = predicted
    math_obs_dataset = math_obs
    
# plots will be shown on a nplotx,nplotx grid
    if np.modf(np.sqrt(no_samples))[0] == 0:
       nplotx=int(np.modf(np.sqrt(no_samples))[1])
    else: 
       nplotx=int(np.modf(np.sqrt(no_samples))[1]+1)

    fig, axs = plt.subplots(nplotx,nplotx)
    ## axs[] NEEDS TWO DIMENSIONS

    for i in np.arange(0, no_samples):
        r, c = int(np.arange(0, no_samples)[i]/nplotx), np.mod(np.arange(0, no_samples)[i],nplotx)
        #axs[0].plot(test_inputs[i], label = "subset" + str(i+1))
        axs[r, c].plot(observations_dataset[i], label = "clean target" )
        axs[r, c].plot(predictions_dataset[i], label = "predicted")
        axs[r, c].plot(math_obs_dataset[i], label = "eqn on input")
        #axs[r, c].semilogx()
        #axs[r, c].semilogy()

    axs[0,0].legend(prop={'size': 7})
    axs[0,1].set(title = plot_title)
    plt.show()

    
# Normalize data
def normalize(data):
    # Remove any NA values to calculate mean and std, but leave them in the output set
    clean_data = data[~np.isnan(data)]
    mean = np.mean(clean_data)
    std = clean_data.std()
    result = (data - mean)/std
    return result

