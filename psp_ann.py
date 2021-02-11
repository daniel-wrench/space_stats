
import numpy as np
import pandas as pd
# import math as m
import tensorflow as tf
import warnings
from matplotlib import pyplot as plt
from data_import_funcs import read_cdfs, date_1d_dict, read_asc_ts, extract_components
from remove_obs_funcs import remove_chunks_df, remove_obs_array
from calculate_stats import calc_struct_sdk, plot_results, normalize


psp_data = read_cdfs(["PSP\psp_fld_l2_mag_rtn_2018110400_v01.cdf",
                        "PSP\psp_fld_l2_mag_rtn_2018110406_v01.cdf",
                        "PSP\psp_fld_l2_mag_rtn_2018110412_v01.cdf"],
                        {'epoch_mag_RTN':(0), 'psp_fld_l2_mag_RTN':(0,3), 'label_RTN':(0,3)})

psp_data_final = extract_components(psp_data, var_name='psp_fld_l2_mag_RTN', label_name='label_RTN', time_var='epoch_mag_RTN', dim=3)

df = pd.DataFrame(psp_data_final)

df['Time'] = pd.to_datetime('2000-01-01 12:00') + pd.to_timedelta(df['epoch_mag_RTN'], unit= 'ns')
df = df.drop(columns = 'epoch_mag_RTN').set_index('Time')

df = df.resample('0.007S').mean()

#Final data to use
data = df['B_R']
len(data)

######################################

#Normalizing data
normalized_arr = normalize([data])
time_index = data.index
normalized_data = pd.DataFrame(data = normalized_arr[0], index = time_index, columns=['B_R'])
normalized_data = normalized_data['B_R']

#Calculating sdk (expected outputs) for a series of subsets (inputs)
chunks = np.split(normalized_data[:2000000], 1000)
row_length = round(0.2*len(chunks[0])-1)
outputs = np.zeros(shape = [1,row_length])

warnings.filterwarnings("ignore") # specify to ignore warning messages

for index, value in enumerate(chunks):
    single_output = calc_struct_sdk(chunks[index], 1/0.007, plot = False)
    output_array = np.array(single_output[1])
    outputs = np.vstack((outputs, output_array))

inputs = np.array(chunks)
outputs = outputs[1:]

np.save(file = 'psp_inputs_norm', arr = inputs)
np.save(file = 'psp_outputs_norm', arr = outputs)

outputs = np.load('psp_outputs_norm.npy')
inputs = np.load('psp_inputs_norm.npy')


#Checking dimensions of inputs and outputs. The length of each output will be the number of nodes in the output layer.
inputs.shape
outputs.shape

#An example of an input and the expected output
pd.DataFrame(inputs[15]).plot()
plt.show()

pd.DataFrame(outputs[15]).plot()
plt.semilogx()
plt.semilogy()
plt.show()

#Define inputs (features) as tensorflow objects, split into training and test sets
inputs_train = tf.constant(inputs[:round(0.9*len(inputs))])
inputs_test = tf.constant(inputs[round(0.9*len(inputs)):])

outputs_train = tf.constant(outputs[:round(0.9*len(inputs))])
outputs_test = tf.constant(outputs[round(0.9*len(inputs)):])

#Checking for missing values. Currently if input set contains missing values, all loss function values give 0
print(np.any(np.isnan(inputs_train)))
print(np.any(np.isnan(outputs_train)))

############################################################################################

########  CONSTRUCT AND FIT A NEURAL NETWORK  ########

#Layers of network
sf_ann = tf.keras.Sequential()
sf_ann.add(tf.keras.layers.Dense(20, activation='relu')) 
sf_ann.add(tf.keras.layers.Dense(20, activation='relu'))
sf_ann.add(tf.keras.layers.Dense(20, activation='relu'))
sf_ann.add(tf.keras.layers.Dense(20, activation='relu'))
sf_ann.add(tf.keras.layers.Dropout(0.25)) #This step is designed to prevent over-fitting
sf_ann.add(tf.keras.layers.Dense(399)) #Number of nodes = number of points in each output

#Specifying optimizer, learning rate, and loss function
sf_ann.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss = tf.keras.losses.MeanSquaredError())
# A lower learning rate makes it easier to identify overfitting by comparing the validation loss with the training loss

#Training the model
sf_ann.fit(inputs_train, outputs_train, validation_split=0.2, epochs = 2000) 

#Evaluating model performance
print(sf_ann.summary())
print(sf_ann.evaluate(inputs_test, outputs_test))

plot_results(sf_ann, test_inputs= inputs_test, test_outputs = outputs_test, no_samples = 5, title='Predictions for some PSP test samples, from 6-layer ANN')