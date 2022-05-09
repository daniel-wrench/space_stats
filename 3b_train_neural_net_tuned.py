#############################################################################
# TENSORFLOW PROGRAM TO 'OPTIMALLY' CONSTRUCT AND EVALUATE NEURAL NETWORK 

model_name = "may_9/mod_4/"

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
#np_load_old = np.load

# modify the default parameters of np.load
#np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

random.seed(5)

# Load PSP data
inputs_train_npy = np.load('data_processed/psp/psp_filled_inputs_0_train.npy')
outputs_train_npy = np.load('data_processed/psp/psp_clean_outputs_train.npy')

inputs_validate_npy = np.load('data_processed/psp/psp_filled_inputs_0_validate.npy')
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

#################################################################

#### CONSTRUCT NETWORK USING KERAS_TUNER LIBRARY ####

def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(3, 10000)))

  # Tune the number of hidden layers
  for i in range(hp.Int('num_layers', 2, 20)):
    # Tune the number of units in the each layer
    model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i), 
                                                min_value=32, 
                                                max_value=512, 
                                                step=32), 
                                    activation='relu'))

  # Tune whether to use dropout
  if hp.Boolean("dropout"):
        model.add(tf.keras.layers.Dropout(rate=0.25))

  model.add(tf.keras.layers.Dense(units=2000))

  # Specify an optimizer, learning rate, and loss function
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  model.compile(
      optimizer=tf.keras.optimizers.Adam(
          learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
      loss=tf.keras.losses.MeanSquaredError())

  return model

# Check that the model builds successfully
model_builder(kt.HyperParameters())

# Instantiate the tuner and perform hypertuning

# Multiple different tuners available, such as Hyperband and RandomSearch
# WARNING - if you change the hyperparameter choices but use an old project_name, it will load the old tuner!

tuner = kt.RandomSearch(model_builder,
                     objective='val_loss',
                     max_trials=10, # No. of trials to run (different combinations of hyperparameters)
                     executions_per_trial=5, # No. of models to fit for each trial (same hyperparams for each execution within a trial),
                     # removes need to run this multiple times
                     directory='my_dir',
                     project_name=model_name) 

print("\nHere is a summary of the tuner search space:\n")
print(tuner.search_space_summary())

# Create EarlyStoppingCriteria callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Run the search
tuner.search(inputs_train, 
              outputs_train,  
              callbacks=[stop_early],
              validation_data=(inputs_validate, outputs_validate),
              epochs=10)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of hidden layers is {best_hps.get('num_layers')} 
and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

print("Here is a more detailed summary of the tuner results:\n ")
print(tuner.results_summary())
# Find detailed logs in directory/project_name

# Extract the model with the best hps
best_model = tuner.get_best_models()[0]

print("\nHere is a summary of the model with the tuned hyperparameters, which will now be trained:\n ")
print(best_model.summary())

# Build the model with the optimal hyperparameters and train it on the data for 500 epochs
history = best_model.fit(inputs_train, 
                          outputs_train, 
                          shuffle=True,
                          callbacks=stop_early,
                          validation_data=(inputs_validate, outputs_validate),
                          epochs=500)

# Alternatively, create best_model with model = tuner.hypermodel.build(best_hps)

# You can now use this as your final model, OR use the following code to only train
# until the minimum validation loss was achieved
# val_loss_per_epoch = history.history['val_loss']
# best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
# print('\nBest epoch: %d' % (best_epoch,))
# print('\mProceeding to train model for only this number of epochs.\n')

# # Re-instantiate the hypermodel and train it a final time with the optimal number of epochs from above.
# hypermodel = tuner.hypermodel.build(best_hps)

# # Retrain the model
# history = hypermodel.fit(inputs_train, 
#                           outputs_train,  
#                           shuffle=True,
#                           validation_data=(inputs_validate, outputs_validate),
#                           epochs=best_epoch)

print(best_model.summary())

# Save the model
best_model.save('results/' + model_name + 'model')

# Save loss over time
np.save(file='results/' + model_name + 'loss', arr=history.history['loss'])
np.save(file='results/' + model_name + 'val_loss', arr=history.history['val_loss'])

# Print performance on validation set
eval_result = best_model.evaluate(inputs_validate, outputs_validate)
print("\nFinal validation loss (tuned model)=", eval_result)

# Save predictions on validation set
validate_predictions = best_model.predict(inputs_validate)
np.save(file='results/' + model_name + 'outputs_validate_predict', arr=validate_predictions)

print("\nFINISHED TRAINING TUNED MODEL\n\n\n")

######################################################################################

# Evaluating final model on test sets - LEAVE TILL THE VERY END

# 
# print('MSE on PSP test set=', hypermodel.evaluate(inputs_test_psp, outputs_test_psp))
# print('MSE on MMS test set=', hypermodel.evaluate(inputs_test_mms, outputs_test_mms))

# test_predictions_mms = model.predict(inputs_test_psp)
# np.save(file='results/' + model_name +
#         'psp_outputs_test_predict', arr=test_predictions_mms)

# test_predictions_mms = model.predict(inputs_test_mms)
# np.save(file='results/' + model_name +
#         'mms_outputs_test_predict', arr=test_predictions_mms)

######################################################################################

#test_predictions_psp_2020 = model.predict(inputs_test_2020)
#np.save(file = 'results/' + model_name + 'psp_2020_outputs_test_predict', arr = test_predictions_psp_2020)

########################################################################
