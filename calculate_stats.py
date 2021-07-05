
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import math
from remove_obs_funcs import remove_chunks_df, remove_obs_df

#Function for plotting observed curves against predicted for the test subsets

def plot_results(predictions_arr, observed_arr, no_samples, gapped = False):

    test_gapped_indices = np.load('test_gapped_indices.npy')
    
#    predictions = model_name.predict(test_inputs)
    predictions = np.load(predictions_arr)
    predicted = pd.DataFrame(predictions)
    predicted = predicted.transpose()
    
    test_pred_gapped = predictions[test_gapped_indices==1]
    gapped_predicted = pd.DataFrame(test_pred_gapped)
    gapped_predicted = gapped_predicted.transpose()

#    observed = pd.DataFrame(test_outputs.numpy())
    observations = np.load(observed_arr)
    observed = pd.DataFrame(observations)
    observed = observed.transpose()
    
    test_obs_gapped = observations[test_gapped_indices==1]
    gapped_observed = pd.DataFrame(test_obs_gapped)
    gapped_observed = gapped_observed.transpose()

#These only work for if all are gapped datasets    
#    math_arr = np.load(math_arr)
#    math_obs = pd.DataFrame(math_arr)
#    math_obs = math_obs.transpose()
    
    observations_dataset = observed
    predictions_dataset = predicted
#    math_obs_dataset = math_obs
    title = 'Predictions from NN of test set of complete and gapped data'
    
    if gapped == True:
      observations_dataset = gapped_observed
      predictions_dataset = gapped_predicted
#      math_obs_dataset = math_obs
      title = 'Predictions from NN of test set of only gapped data'
    
# plots will be shown on a nplotx,nplotx grid
    if np.modf(np.sqrt(no_samples))[0] == 0:
       nplotx=int(np.modf(np.sqrt(no_samples))[1])
    else: 
       nplotx=int(np.modf(np.sqrt(no_samples))[1]+1)

    fig, axs = plt.subplots(nplotx,nplotx,sharey=True)
    ## axs[] NEEDS TWO DIMENSIONS

    for i in np.arange(0, no_samples):
        r, c = int(np.arange(0, no_samples)[i]/nplotx), np.mod(np.arange(0, no_samples)[i],nplotx)
        #axs[0].plot(test_inputs[i], label = "subset" + str(i+1))
        axs[r, c].plot(observations_dataset[i], label = "observed subset" )
        axs[r, c].plot(predictions_dataset[i], label = "predicted subset")
 #       axs[r, c].plot(math_obs_dataset[i], label = "observed from gapped subset")
        axs[r, c].semilogx()
        axs[r, c].semilogy()

    
    axs[0,1].legend()

    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    axs[0,1].set(title = title)
    # axs[1].set(title = 'Next 20 time steps', xlabel = 'Time')
    # axs[2].set(title = 'PREDICTED next 20 time steps', xlabel = 'Time')
    #plt.title(title)
    plt.show()

#calc_struct_sdk(real_data, 1/48)

# Normalize data
def normalize(data):
    # Remove any NA values to calculate mean and std, but leave them in the output set
    clean_data = data[~np.isnan(data)]
    mean = np.mean(clean_data)
    std = clean_data.std()
    result = (data - mean)/std
    return result

