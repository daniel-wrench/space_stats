
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import math
from remove_obs_funcs import remove_chunks_df, remove_obs_df

# voyager_data_raw = read_cdfs(["voyager/voyager1_48s_mag-vim_20090101_v01.cdf", 
#             "voyager/voyager1_48s_mag-vim_20100101_v01.cdf", 
#             "voyager/voyager1_48s_mag-vim_20110101_v01.cdf",
#             "voyager/voyager1_48s_mag-vim_20120101_v01.cdf",
#             "voyager/voyager1_48s_mag-vim_20130101_v01.cdf"],
#             ['Epoch', 'F1', 'BT', 'BR', 'BN'])

# voyager_data = date_1d_dict(voyager_data_raw, '48S')

# real_data = voyager_data['2009-05-10 12:38':'2009-05-10 23:59']['F1'].dropna()
# len(real_data)
# real_data.plot()
# plt.show()


#Values of lag may eventually need to follow log rule, so as not too be too computationally expensive

def calc_struct_sdk(data, freq, plot):
    #Calculate lags
    lag_function = {}
    for i in np.arange(1, round(0.2*len(data))): #Limiting maximum lag to 20% of dataset length
        lag_function[i] = data.diff(i)

    #Initialise dataframe
    structure_functions = pd.DataFrame(index = np.arange(1,len(data)))

    #Calculate different orders of structure functions
    for p in np.arange(1, 5):
        lag_dataframe = pd.DataFrame(lag_function).abs()**p
        structure_functions[p] = pd.DataFrame(lag_dataframe.mean())

    #Converting lag values from points to seconds
    structure_functions.index = structure_functions.index/freq 

    #Calculate kurtosis
    sdk = structure_functions[[2, 4]]
    sdk.columns = ['2', '4']
    sdk['2^2']=sdk['2']**2
    sdk['kurtosis'] = sdk['4'].div(sdk['2^2'])

    if plot == True:

        #Plot structure functions
        fig, axs = plt.subplots(1,2)
        axs[0].plot(structure_functions)
        axs[0].semilogx()
        axs[0].semilogy()
        axs[0].set(title = 'Structure functions (orders 1-4)',
                xlabel = 'Lag (s)')
        axs[1].plot(sdk['kurtosis'])
        axs[1].semilogx()
        axs[1].set(title = 'Kurtosis', 
                xlabel = 'Lag (s)')
        plt.show()
    
    else:
        return structure_functions.dropna()



#Function for plotting observed curves against predicted for the test subsets

def plot_results(predictions_arr, observed_arr, no_samples, title):

#    predictions = model_name.predict(test_inputs)
    predicted = np.load(predictions_arr)
    predicted = pd.DataFrame(predicted)
    predicted = predicted.transpose()

#    observed = pd.DataFrame(test_outputs.numpy())
    observed = np.load(observed_arr)
    observed = pd.DataFrame(observed)
    observed = observed.transpose()

    fig, axs = plt.subplots(1, no_samples, sharey=True)

    for i in np.arange(0, no_samples):
        #axs[0].plot(test_inputs[i], label = "subset" + str(i+1))
        axs[i].plot(observed[i], label = "observed subset" )
        axs[i].plot(predicted[i], label = "predicted subset")
        axs[i].semilogx()
        axs[i].semilogy()

    
    axs[1].legend()

    # axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    axs[0].set(title =title)
    # axs[1].set(title = 'Next 20 time steps', xlabel = 'Time')
    # axs[2].set(title = 'PREDICTED next 20 time steps', xlabel = 'Time')
    #plt.title(title)
    plt.show()

#calc_struct_sdk(real_data, 1/48)

#Normalize data
def normalize(data):
    nd = data - np.mean(data)
    return nd/nd.std()

