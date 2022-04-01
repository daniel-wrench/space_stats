# Plot results of neural network

import calculate_stats as calc

model_path = 'april_1/mod_1/'

calc.plot_validation_error(
path = 'results/' + model_path)
 
calc.plot_results(
predictions_arr = 'results/' + model_path + 'psp_outputs_test_predict.npy', 
observed_arr = 'data_processed/psp/psp_clean_outputs_test.npy',
no_samples = 16, 
spacecraft = 'PSP', 
model = model_path)

calc.plot_log_results(
predictions_arr = 'results/' + model_path + 'psp_outputs_test_predict.npy', 
observed_arr = 'data_processed/psp/psp_clean_outputs_test.npy',
no_samples = 16, 
spacecraft = 'PSP_log',
model = model_path)

calc.plot_results(
predictions_arr = 'results/' + model_path + 'mms_outputs_test_predict.npy', 
observed_arr = 'data_processed/mms/mms_clean_outputs_test.npy',
no_samples = 9, 
spacecraft = 'MMS', 
model = model_path)



