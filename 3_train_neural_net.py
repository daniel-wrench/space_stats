#############################################################################
# TENSORFLOW PROGRAM TO CONSTRUCT AND EVALUATE NEURAL NETWORK
#############################################################################

# NEXT STEPS
# Download the processed data from Raapoi and try run this locally.

#############################################################################

import random
import tensorflow as tf
import keras_tuner as kt
import numpy as np

# Getting issue with allow_pickle being set to False (implicitly) for some reason. Here is the stack overflow work-around

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


model_name = "may_6/mod_2/"

random.seed(5)

# Load PSP data - NOW ALL WITHOUT MASK VECTOR
inputs_train_npy = np.load('data_processed/psp/psp_filled_inputs_0_train_flat.npy')
outputs_train_npy = np.load('data_processed/psp/psp_clean_outputs_train.npy')

inputs_validate_npy = np.load('data_processed/psp/psp_filled_inputs_0_validate_flat.npy')
outputs_validate_npy = np.load('data_processed/psp/psp_clean_outputs_validate.npy')

print("\nThe dimensions of the input training data are", inputs_train_npy.shape)
print("The dimensions of the output training data are", outputs_train_npy.shape)
print("\nThe dimensions of the input training data are", inputs_validate_npy.shape)
print("The dimensions of the output training data are", outputs_validate_npy.shape)

# Define features as tensorflow objects
inputs_train = tf.constant(inputs_train_npy)
outputs_train = tf.constant(outputs_train_npy)

inputs_validate = tf.constant(inputs_validate_npy)
outputs_validate = tf.constant(outputs_validate_npy)

print("\nHere is the first training input:\n", inputs_train[0])
print("\nHere is the first training output:\n", outputs_train[0], "\n")

#### CONSTRUCTING NETWORK ####

# Layers
sf_ann = tf.keras.Sequential()
sf_ann.add(tf.keras.Input(shape=30000,))

sf_ann.add(tf.keras.layers.Dense(10, activation='relu'))
sf_ann.add(tf.keras.layers.Dense(10, activation='relu'))

# Optional dropout layer for preventing overfitting
# sf_ann.add(tf.keras.layers.Dropout(0.25))

sf_ann.add(tf.keras.layers.Dense(2000))

# Specifying an optimizer, learning rate, and loss function
sf_ann.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError())

# Specifying early stopping criteria
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when 'val_loss' is no longer improving
        monitor='val_loss',
        # "no longer improving" is defined as "no better than 0 less"
        min_delta=0.01,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1
    )
]

# Training the model (remove callbacks argument for no early stopping)
# NB: Data is automatically shuffled before each epoch
history = sf_ann.fit(inputs_train,
                     outputs_train,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_data=(inputs_validate, outputs_validate),
                     epochs=500)

np.save(file='results/' + model_name + 'loss', arr=history.history['loss'])
np.save(file='results/' + model_name + 'val_loss', arr=history.history['val_loss'])

######################################################################################

# Saving predictions on validation sets

validate_predictions = sf_ann.predict(inputs_validate)
np.save(file='results/' + model_name + 'outputs_validate_predict', arr=validate_predictions)

######################################################################################

# Evaluating final model on test sets - LEAVE TILL THE VERY END

# print(sf_ann.summary())
# print('MSE on PSP test set=', sf_ann.evaluate(inputs_test_psp, outputs_test_psp))
# print('MSE on MMS test set=', sf_ann.evaluate(inputs_test_mms, outputs_test_mms))

# test_predictions_mms = sf_ann.predict(inputs_test_psp)
# np.save(file='results/' + model_name +
#         'psp_outputs_test_predict', arr=test_predictions_mms)

# test_predictions_mms = sf_ann.predict(inputs_test_mms)
# np.save(file='results/' + model_name +
#         'mms_outputs_test_predict', arr=test_predictions_mms)

######################################################################################

#test_predictions_psp_2020 = sf_ann.predict(inputs_test_2020)
#np.save(file = 'results/' + model_name + 'psp_2020_outputs_test_predict', arr = test_predictions_psp_2020)

########################################################################
