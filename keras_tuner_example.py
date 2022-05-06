
# Example of using the Keras Tuner to find the optimal hyperparameters of a neural network in Tensorflow

# Taken from documentation including https://www.tensorflow.org/tutorials/keras/keras_tuner and https://keras.io/guides/keras_tuner/getting_started/

import keras
import keras_tuner as kt
import tensorflow as tf

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

# Define the model, including the hyperparameter search space

def model_builder(hp):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

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

  model.add(tf.keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  return model

# Check that the model builds successfully
model_builder(kt.HyperParameters())

# Instantiate the tuner and perform hypertuning

# Multiple different tuners available, such as Hyperband and RandomSearch
tuner = kt.RandomSearch(model_builder,
                     objective='val_accuracy',
                     max_trials=5, # No. of trials to run (different combinations of hyperparameters)
                     executions_per_trial=1, # No. of models to fit for each trial (same hyperparams for each execution within a trial)
                     directory='my_dir',
                     project_name='testing') 
                     # WARNING - if you change the hyperparameter choices but use an old project_name, it will load the old tuner!

print("\nHere is a summary of the tuner search space:\n")
print(tuner.search_space_summary())

# Create EarlyStoppingCriteria callback
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Run the search
tuner.search(img_train, label_train, epochs=2, validation_split=0.2, callbacks=[stop_early])
# or, validation_data=(x_val, y_val)

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

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
history = best_model.fit(img_train, label_train, epochs=2, validation_split=0.2)

# Alternatively, create best_model with model = tuner.hypermodel.build(best_hps)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('\nBest epoch: %d' % (best_epoch,))
print('Proceeding to train for this number of epochs.\n')

# Re-instantiate the hypermodel and train it a final time with the optimal number of epochs from above.
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

# Finally, evaluate the final model on the test data
eval_result = hypermodel.evaluate(img_test, label_test)
print("\n[test loss, test accuracy]:", eval_result)

print("\nFINISHED")