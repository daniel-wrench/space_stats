
# TENSORFLOW PROGRAM TO CONSTRUCT AND EVALUATE NEURAL NETWORK

import random
import tensorflow as tf
#import keras_tuner as kt
import numpy as np

# Getting issue with allow_pickle being set to False (implicitly) for some reason. Here is the stack overflow work-around

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


model_name = "may_6/mod_1/"

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
history = sf_ann.fit(inputs_train, outputs_train, shuffle=True,
                     callbacks=callbacks, validation_data=(inputs_validate, outputs_validate), epochs=50)

np.save(file='results/' + model_name + 'loss', arr=history.history['loss'])
np.save(file='results/' + model_name + 'val_loss',
        arr=history.history['val_loss'])

######################################################################################

# Saving predictions on validation sets

validate_predictions = sf_ann.predict(inputs_validate)
np.save(file='results/' + model_name +
        'outputs_validate_predict', arr=validate_predictions)

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
