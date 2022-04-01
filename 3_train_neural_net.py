
# TENSORFLOW PROGRAM TO CONSTRUCT AND EVALUATE NEURAL NETWORK

model_name = "april_1/mod_1/"


import numpy as np
import tensorflow as tf
import random

random.seed(5)

# Load data
input_training = np.load('data_processed/psp/psp_filled_inputs_train_flat.npy')
output_training = np.load('data_processed/psp/psp_clean_outputs_train.npy')

input_training_b = np.load('data_processed/psp/psp_filled_inputs_train_flat_no_ind.npy')
input_training_2 = np.load('data_processed/psp/psp_lint_inputs_train_flat.npy')
input_training_2b = np.load('data_processed/psp/psp_lint_inputs_train_flat_no_ind.npy')

input_testing = np.load('data_processed/psp/psp_filled_inputs_test_flat.npy')
output_testing = np.load('data_processed/psp/psp_clean_outputs_test.npy')

input_testing_b = np.load('data_processed/psp/psp_filled_inputs_test_flat_no_ind.npy')
input_testing_2 = np.load('data_processed/psp/psp_lint_inputs_test_flat.npy')
input_testing_2b = np.load('data_processed/psp/psp_lint_inputs_test_flat_no_ind.npy')

input_testing_mms = np.load('data_processed/mms/mms_filled_inputs_test_flat.npy')
output_testing_mms = np.load('data_processed/mms/mms_clean_outputs_test.npy')

input_testing_mms_b = np.load('data_processed/mms/mms_filled_inputs_test_flat_no_ind.npy')
input_testing_mms_2 = np.load('data_processed/mms/mms_lint_inputs_test_flat.npy')
input_testing_mms_2b = np.load('data_processed/mms/mms_lint_inputs_test_flat_no_ind.npy')

# PSP data from 2020

#input_testing_2020 = np.load('data_processed/psp/psp_filled_inputs_test_flat_2020.npy')
#output_testing_2020 = np.load('data_processed/psp/psp_clean_outputs_test_2020.npy')

#input_testing_b_2020 = np.load('data_processed/psp/psp_filled_inputs_test_flat_no_ind_2020.npy')
#input_testing_2_2020 = np.load('data_processed/psp/psp_lint_inputs_test_flat_2020.npy')
#input_testing_2b_2020 = np.load('data_processed/psp/psp_lint_inputs_test_flat_no_ind_2020.npy')

print("\nThe dimensions of the input training data are", input_training.shape)
print("The dimensions of the output training data are", output_training.shape)
print("The dimensions of the input testing data are", input_testing.shape)
print("The dimensions of the output testing data are", output_testing.shape)
print("The dimenstions of the MMS input test data are", input_testing_mms.shape)
print("The dimensions of the MMS output test data are", output_testing_mms.shape)
#print("The dimensions of the 2020 PSP input testing data are", input_testing_2020.shape)
#print("The dimensions of the 2020 PSP output testing data are", output_testing_2020.shape)

# Define features as tensorflow objects
inputs_train = tf.constant(input_training_2b)
outputs_train = tf.constant(output_training)

inputs_test = tf.constant(input_testing_2b)
outputs_test = tf.constant(output_testing)

#inputs_test_2020 = tf.constant(input_testing_2b_2020)
#outputs_test_2020 = tf.constant(output_testing_2020)

inputs_test_mms = tf.constant(input_testing_mms_2b)
outputs_test_mms = tf.constant(output_testing_mms)

print("\nHere is the first training input:\n", inputs_train[0])
print("\nHere is the first training output:\n", outputs_train[0], "\n")

#### CONSTRUCTING NETWORK ####

# Layers
sf_ann = tf.keras.Sequential()
sf_ann.add(tf.keras.Input(shape =30000,))

sf_ann.add(tf.keras.layers.Dense(10, activation='relu'))
sf_ann.add(tf.keras.layers.Dense(10, activation='relu'))

# Optional dropout layer for preventing overfitting
#sf_ann.add(tf.keras.layers.Dropout(0.25))

sf_ann.add(tf.keras.layers.Dense(2000))

# Specifying an optimizer, learning rate, and loss function
sf_ann.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanAbsolutePercentageError())

# Specifying early stopping criteria
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        # Stop training when 'val_loss' is no longer improving
        monitor = 'val_loss',
        # "no longer improving" is defined as "no better than 0 less"
        min_delta=0.01,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1
    )
]

# Training the model (remove callbacks argument for no early stopping)
# NB: Data is automatically shuffled before each epoch
history = sf_ann.fit(inputs_train, outputs_train, shuffle = True, callbacks = callbacks, validation_split=0.05, epochs=50)

np.save(file = 'results/' + model_name + 'loss', arr = history.history['loss'])
np.save(file = 'results/' + model_name + 'val_loss', arr = history.history['val_loss'])

# Evaluating model performance
print(sf_ann.summary())
print('MSE on PSP test set=', sf_ann.evaluate(inputs_test, outputs_test))
print('MSE on MMS test set=', sf_ann.evaluate(inputs_test_mms, outputs_test_mms))
#print('MSE on PSP 2020 test set=', sf_ann.evaluate(inputs_test_2020, outputs_test_2020))

# Saving predictions on test sets
test_predictions = sf_ann.predict(inputs_test)
np.save(file = 'results/' + model_name + 'psp_outputs_test_predict', arr = test_predictions)

test_predictions_mms = sf_ann.predict(inputs_test_mms)
np.save(file = 'results/' + model_name + 'mms_outputs_test_predict', arr = test_predictions_mms)

#test_predictions_psp_2020 = sf_ann.predict(inputs_test_2020)
#np.save(file = 'results/' + model_name + 'psp_2020_outputs_test_predict', arr = test_predictions_psp_2020)
