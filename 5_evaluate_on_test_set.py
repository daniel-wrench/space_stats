
# Evaluating final model on test sets - LEAVE TILL THE VERY END

model_name = "may_9/mod_13/"

########################################################################

import numpy as np
import tensorflow as tf

# Load PSP data
inputs_test_psp_npy = np.load('data_processed/psp/psp_lint_inputs_test.npy')
outputs_test_psp_npy = np.load('data_processed/psp/psp_clean_outputs_test.npy')

inputs_test_mms_npy = np.load('data_processed/mms/mms_lint_inputs_test.npy')
outputs_test_mms_npy = np.load('data_processed/mms/mms_clean_outputs_test.npy')

print("\nThe dimensions of the PSP input testing data are", inputs_test_psp_npy.shape)
print("The dimensions of the PSP output testing data are", outputs_test_psp_npy.shape)

print("\nThe dimensions of the MMS input testing data are", inputs_test_mms_npy.shape)
print("The dimensions of the MMS output testing data are", outputs_test_mms_npy.shape)

# Define features as tensorflow objects
inputs_test_psp = tf.constant(inputs_test_psp_npy)
outputs_test_psp = tf.constant(outputs_test_psp_npy)

inputs_test_mms = tf.constant(inputs_test_mms_npy)
outputs_test_mms = tf.constant(outputs_test_mms_npy)

print("\nHere is the first PSP testing input:\n", inputs_test_psp[0])
print("\nHere is the first PSP testing output:\n", outputs_test_psp[0], "\n")

print("\nHere is the first MMS testing input:\n", inputs_test_mms[0])
print("\nHere is the first MMS testing output:\n", outputs_test_mms[0], "\n")

########################################################################

# Load model and evaluate on test set
print("Loading saved model...")
model = tf.keras.models.load_model('results/' + model_name + 'model')

print(model.summary())
print('MSE on PSP test set=', model.evaluate(inputs_test_psp, outputs_test_psp))
print('MSE on MMS test set=', model.evaluate(inputs_test_mms, outputs_test_mms))

test_predictions_psp = model.predict(inputs_test_psp)
test_predictions_mms = model.predict(inputs_test_mms)

np.save(file='results/' + model_name + 'psp_outputs_test_predict', arr=test_predictions_psp)
np.save(file='results/' + model_name + 'mms_outputs_test_predict', arr=test_predictions_mms)


########################################################################