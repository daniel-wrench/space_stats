
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import math

#Function for plotting observed curves against predicted for the test subsets

def plot_results(predictions_arr, observed_arr, no_samples, spacecraft, model):

    predicted = np.load(predictions_arr)
    predicted = pd.DataFrame(predicted)
    predicted = predicted.transpose()

    observed = np.load(observed_arr)
    observed = pd.DataFrame(observed)
    observed = observed.transpose()
    
# plots will be shown on a nplotx,nplotx grid
    if np.modf(np.sqrt(no_samples))[0] == 0:
       nplotx=int(np.modf(np.sqrt(no_samples))[1])
    else: 
       nplotx=int(np.modf(np.sqrt(no_samples))[1]+1)

    fig, axs = plt.subplots(nplotx,nplotx,sharey=True, figsize = (10,10))

    for i in np.arange(0, no_samples):
        r, c = int(np.arange(0, no_samples)[i]/nplotx), np.mod(np.arange(0, no_samples)[i],nplotx)
        #axs[0].plot(test_inputs[i], label = "subset" + str(i+1))
        axs[r, c].plot(observed[i], label = "observed" )
        axs[r, c].plot(predicted[i], label = "predicted")
        axs[r, c].semilogx()
        axs[r, c].semilogy()

    axs[0,0].legend()
    axs[0,0].set(title = "Model predictions on " + spacecraft + " test set")
    plt.savefig('results/' + model + spacecraft + '_predictions_plot.png')
    plt.show()


    
# Normalize data
def normalize(data):
    # Remove any NA values to calculate mean and std, but leave them in the output set
    clean_data = data[~np.isnan(data)]
    mean = np.mean(clean_data)
    std = clean_data.std()
    result = (data - mean)/std
    return result

