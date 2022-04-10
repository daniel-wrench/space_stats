
# TENSORFLOW PROGRAM TO CONSTRUCT AND EVALUATE NEURAL NETWORK

import random
import tensorflow as tf
import keras_tuner as kt
import numpy as np

# Getting issue with allow_pickle being set to False (implicitly) for some reason. Here is the stack overflow work-around

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


model_name = "april_5/mod_1/"

random.seed(5)

# Load PSP data
input_training_psp = np.load('data_processed/psp/psp_filled_inputs_train_flat.npy')
output_training_psp = np.load('data_processed/psp/psp_clean_outputs_train.npy')

input_training_b_psp = np.load('data_processed/psp/psp_filled_inputs_train_flat_no_ind.npy')
input_training_2_psp = np.load('data_processed/psp/psp_lint_inputs_train_flat.npy')
input_training_2b_psp = np.load('data_processed/psp/psp_lint_inputs_train_flat_no_ind.npy')

input_testing_psp = np.load('data_processed/psp/psp_filled_inputs_test_flat.npy')
output_testing_psp = np.load('data_processed/psp/psp_clean_outputs_test.npy')

input_testing_b_psp = np.load('data_processed/psp/psp_filled_inputs_test_flat_no_ind.npy')
input_testing_2_psp = np.load('data_processed/psp/psp_lint_inputs_test_flat.npy')
input_testing_2b_psp = np.load('data_processed/psp/psp_lint_inputs_test_flat_no_ind.npy')

input_validate_psp = np.load('data_processed/psp/psp_filled_inputs_validate_flat.npy')
output_validate_psp = np.load('data_processed/psp/psp_clean_outputs_validate.npy')

input_validate_b_psp = np.load('data_processed/psp/psp_filled_inputs_validate_flat_no_ind.npy')
input_validate_2_psp = np.load('data_processed/psp/psp_lint_inputs_validate_flat.npy')
input_validate_2b_psp = np.load('data_processed/psp/psp_lint_inputs_validate_flat_no_ind.npy')

# Load MMS data
input_training_mms = np.load('data_processed/mms/mms_filled_inputs_train_flat.npy')
output_training_mms = np.load('data_processed/mms/mms_clean_outputs_train.npy')

input_training_b_mms = np.load('data_processed/mms/mms_filled_inputs_train_flat_no_ind.npy')
input_training_2_mms = np.load('data_processed/mms/mms_lint_inputs_train_flat.npy')
input_training_2b_mms = np.load('data_processed/mms/mms_lint_inputs_train_flat_no_ind.npy')

input_testing_mms = np.load('data_processed/mms/mms_filled_inputs_test_flat.npy')
output_testing_mms = np.load('data_processed/mms/mms_clean_outputs_test.npy')

input_testing_b_mms = np.load('data_processed/mms/mms_filled_inputs_test_flat_no_ind.npy')
input_testing_2_mms = np.load('data_processed/mms/mms_lint_inputs_test_flat.npy')
input_testing_2b_mms = np.load('data_processed/mms/mms_lint_inputs_test_flat_no_ind.npy')

input_validate_mms = np.load('data_processed/mms/mms_filled_inputs_validate_flat.npy')
output_validate_mms = np.load('data_processed/mms/mms_clean_outputs_validate.npy')

input_validate_b_mms = np.load('data_processed/mms/mms_filled_inputs_validate_flat_no_ind.npy')
input_validate_2_mms = np.load('data_processed/mms/mms_lint_inputs_validate_flat.npy')
input_validate_2b_mms = np.load('data_processed/mms/mms_lint_inputs_validate_flat_no_ind.npy')

# Combine sets (but keep test sets apart)
input_training = np.concatenate((input_training_psp, input_training_mms))
output_training = np.concatenate((output_testing_psp, output_testing_mms))

input_training_b = np.concatenate((input_training_b_psp, input_training_b_mms))
input_training_2 = np.concatenate((input_training_2_psp, input_training_2_mms))
input_training_2b = np.concatenate((input_training_2b_psp, input_training_2b_mms))

input_validate = np.concatenate((input_validate_psp, input_validate_mms))
output_validate = np.concatenate((output_testing_psp, output_testing_mms))

input_validate_b = np.concatenate((input_validate_b_psp, input_validate_b_mms))
input_validate_2 = np.concatenate((input_validate_2_psp, input_validate_2_mms))
input_validate_2b = np.concatenate((input_validate_2b_psp, input_validate_2b_mms))

# PSP data from 2020

#input_testing_2020 = np.load('data_processed/psp/psp_filled_inputs_test_flat_2020.npy')
#output_testing_2020 = np.load('data_processed/psp/psp_clean_outputs_test_2020.npy')

#input_testing_b_2020 = np.load('data_processed/psp/psp_filled_inputs_test_flat_no_ind_2020.npy')
#input_testing_2_2020 = np.load('data_processed/psp/psp_lint_inputs_test_flat_2020.npy')
#input_testing_2b_2020 = np.load('data_processed/psp/psp_lint_inputs_test_flat_no_ind_2020.npy')

print("\nThe dimensions of the input training data are", input_training.shape)
print("The dimensions of the output training data are", output_training.shape)
print("\nThe dimensions of the input training data are", input_validate.shape)
print("The dimensions of the output training data are", output_validate.shape)
print("The dimensions of the PSP input testing data are", input_testing_psp.shape)
print("The dimensions of the PSP output testing data are", output_testing_psp.shape)
print("The dimenstions of the MMS input testing data are", input_testing_mms.shape)
print("The dimensions of the MMS output testing data are", output_testing_mms.shape)
#print("The dimensions of the 2020 PSP input testing data are", input_testing_2020.shape)
#print("The dimensions of the 2020 PSP output testing data are", output_testing_2020.shape)

# Define features as tensorflow objects
inputs_train = tf.constant(input_training_2b)
outputs_train = tf.constant(output_training)

inputs_validate = tf.constant(input_validate)
outputs_validate = tf.constant(output_validate)

inputs_test_psp = tf.constant(input_testing_2b_psp)
outputs_test_psp = tf.constant(output_testing_psp)

#inputs_test_2020 = tf.constant(input_testing_2b_2020)
#outputs_test_2020 = tf.constant(output_testing_2020)

inputs_test_mms = tf.constant(input_testing_2b_mms)
outputs_test_mms = tf.constant(output_testing_mms)

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
history = sf_ann.fit(inputs_train, outputs_train, shuffle=True,
                     callbacks=callbacks, validation_data=(inputs_validate, outputs_validate), epochs=50)

np.save(file='results/' + model_name + 'loss', arr=history.history['loss'])
np.save(file='results/' + model_name + 'val_loss',
        arr=history.history['val_loss'])

# Evaluating model performance
print(sf_ann.summary())
print('MSE on PSP test set=', sf_ann.evaluate(inputs_test_psp, outputs_test_psp))
print('MSE on MMS test set=', sf_ann.evaluate(inputs_test_mms, outputs_test_mms))
#print('MSE on PSP 2020 test set=', sf_ann.evaluate(inputs_test_2020, outputs_test_2020))

# Saving predictions on test sets
test_predictions = sf_ann.predict(inputs_test_psp)
np.save(file='results/' + model_name +
        'psp_outputs_test_predict', arr=test_predictions)

test_predictions_mms = sf_ann.predict(inputs_test_mms)
np.save(file='results/' + model_name +
        'mms_outputs_test_predict', arr=test_predictions_mms)

#test_predictions_psp_2020 = sf_ann.predict(inputs_test_2020)
#np.save(file = 'results/' + model_name + 'psp_2020_outputs_test_predict', arr = test_predictions_psp_2020)

########################################################################

## TRYING KERAS_TUNER HYPERPARAMETER TUNING ##

# Define the model, including the hyperparameter search space

# def model_builder(hp):
#   model = tf.keras.Sequential()
#   model.add(tf.keras.layers.Flatten(input_shape=(10000, 3)))

#   # Tune the number of units in the first Dense layer
#   # Choose an optimal value between 32-512
#   hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
#   model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))

#   # Tune whether to use dropout
#   if hp.Boolean("dropout"):
#       model.add(tf.keras.layers.Dropout(rate=0.25))

#   model.add(tf.keras.layers.Dense(10))

#   # Tune the learning rate for the optimizer
#   # Choose an optimal value from 0.01, 0.001, or 0.0001
#   hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

#   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])

#   return model

# # Check that the model builds successfully
# model_builder(kt.HyperParameters())

# # Instantiate the tuner and perform hypertuning

# # Multiple different tuners available, such as Hyperband and RandomSearch
# tuner = kt.Hyperband(model_builder,
#                      objective='val_accuracy',
#                      max_epochs=10,
#                      factor=3,
#                      directory='my_dir',
#                      project_name='intro_to_kt')

# # Create EarlyStoppingCriteria callback
# stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# # Run the search
# tuner.search(inputs_train, outputs_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
# # or, validation_data=(x_val, y_val)

# # Get the optimal hyperparameters
# best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# # (Alternatively, we can return the best model using the following line)
# # best_model = tuner.get_best_models()[0]

# print(f"""
# The hyperparameter search is complete. The optimal number of units in the first densely-connected
# layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
# is {best_hps.get('learning_rate')}.
# """)

# # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
# model = tuner.hypermodel.build(best_hps)
# history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

# val_acc_per_epoch = history.history['val_accuracy']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))

# # Re-instantiate the hypermodel and train it a final time with the optimal number of epochs from above.

# hypermodel = tuner.hypermodel.build(best_hps)

# # Retrain the model
# hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

# # Finally, evaluate the final model on the test data

# eval_result = hypermodel.evaluate(img_test, label_test)
# print("[test loss, test accuracy]:", eval_result)
