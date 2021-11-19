# Plot results of neural network

import space_stats.calculate_stats as calc


model_path = 'nov_19/mod_3_2b/'

calc.plot_validation_error(
path = 'results/' + model_path)
 
calc.plot_results(
predictions_arr = 'results/nov_19/mod_3_2b/psp_outputs_test_predict.npy', 
observed_arr = 'data_processed/psp/psp_clean_outputs_test.npy',
no_samples = 16, 
spacecraft = 'PSP', 
model = model_path)

calc.plot_log_results(
predictions_arr = 'results/nov_19/mod_3_2b/psp_outputs_test_predict.npy',
observed_arr = 'data_processed/psp/psp_clean_outputs_test.npy',
no_samples = 16, 
spacecraft = 'PSP_log',
model = model_path)

calc.plot_results(
predictions_arr = 'results/nov_19/mod_3_2b/mms_outputs_test_predict.npy', 
observed_arr = 'data_processed/mms/mms_clean_outputs_test.npy',
no_samples = 9, 
spacecraft = 'MMS', 
model = model_path)



