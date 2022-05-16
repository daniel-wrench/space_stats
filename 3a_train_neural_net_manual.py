#############################################################################
# TENSORFLOW PROGRAM TO 'MANUALLY' CONSTRUCT AND EVALUATE NEURAL NETWORK

model_name = "may_9/mod_14/"

#############################################################################

# NEXT STEPS
# Download the processed data from Raapoi and try run this locally.

#############################################################################

import random
import tensorflow as tf
import numpy as np

# Getting issue with allow_pickle being set to False (implicitly) for some reason. Here is the stack overflow work-around

# save np.load
#np_load_old = np.load

# modify the default parameters of np.load
#np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

random.seed(5)

# Load PSP data
inputs_train_npy = np.load('data_processed/psp/psp_lint_inputs_train.npy')
outputs_train_npy = np.load('data_processed/psp/psp_clean_outputs_train.npy')

inputs_validate_npy = np.load('data_processed/psp/psp_lint_inputs_validate.npy')
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

print("\nHere is the first validation input:\n", inputs_validate[0])
print("\nHere is the first validation output:\n", outputs_validate[0], "\n")

# Draft code for a manual grid search that plots results for each model

# layers = [2, 5, 10, 20, 50]
# nodes = [10, 20, 50, 100, 1000]
# inputs = ["lint", "filled_9", "filled_0"]

# x = 0
# for i in range(len(layers)):
#     for j in range(len(nodes)):
#         for k in range(len(inputs)):
#             x += 1
#             print("mod_" + str(x))
#             print("Layers = " + str(layers[i]))
#             print("Nodes in each layer = " + str(nodes[j]))
#             print("Input version = " + inputs[k])
            



#################################################################

#### CONSTRUCT NETWORK USING MANUAL SPECIFICATION NETWORK ####

# Layers
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(3,10000)))

model.add(tf.keras.layers.Dense(units=15000, activation='relu'))
model.add(tf.keras.layers.Dense(units=7500, activation='relu'))
model.add(tf.keras.layers.Dense(units=3750, activation='relu'))

# Optional dropout layer for preventing overfitting
# model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(units=2000))

# Specify an optimizer, learning rate, and loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=tf.keras.losses.MeanSquaredError())

# Create EarlyStoppingCriteria callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.01)

history = model.fit(inputs_train,
                     outputs_train,
                     shuffle=True,
                     callbacks=stop_early,
                     validation_data=(inputs_validate, outputs_validate),
                     epochs=500)

print(model.summary())

# Save the model
model.save('results/' + model_name + 'model')

# Save loss over time
np.save(file='results/' + model_name + 'loss', arr=history.history['loss'])
np.save(file='results/' + model_name + 'val_loss', arr=history.history['val_loss'])

# Print performance on validation set
eval_result = model.evaluate(inputs_validate, outputs_validate)
print("\nFinal validation loss (manual model)=", eval_result)

# Save predictions on validation set
validate_predictions = model.predict(inputs_validate)
np.save(file='results/' + model_name + 'outputs_validate_predict', arr=validate_predictions)

print("\nFINISHED TRAINING MANUAL MODEL\n\n\n")

######################################################################################

