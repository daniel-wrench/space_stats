
import numpy as np
import tensorflow as tf

outputs = np.load('psp_outputs_norm.npy')
inputs = np.load('psp_inputs_norm.npy')

#Define inputs (features) as tensorflow objects, split into training and test sets
inputs_train = tf.constant(inputs[:round(0.9*len(inputs))])
inputs_test = tf.constant(inputs[round(0.9*len(inputs)):])

outputs_train = tf.constant(outputs[:round(0.9*len(inputs))])
outputs_test = tf.constant(outputs[round(0.9*len(inputs)):])

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