# Plot results of neural network on validation set
# Use these to evaluate to the hyperparameters(?)

import calculate_stats as calc

model_path = 'april_13/mod_1/'

calc.plot_validation_error(path = 'results/' + model_path)
 
calc.plot_results(
    predictions_arr = 'results/' + model_path + 'psp_outputs_validate_predict.npy', 
    observed_arr = 'data_processed/psp/psp_clean_outputs_validate.npy',
    no_samples = 16, 
    spacecraft = 'PSP', 
    model = model_path
)

calc.plot_log_results(
    predictions_arr = 'results/' + model_path + 'psp_outputs_validate_predict.npy', 
    observed_arr = 'data_processed/psp/psp_clean_outputs_validate.npy',
    no_samples = 16, 
    spacecraft = 'PSP_log',
    model = model_path
)

calc.plot_results(
    predictions_arr = 'results/' + model_path + 'mms_outputs_validate_predict.npy', 
    observed_arr = 'data_processed/mms/mms_clean_outputs_validate.npy',
    no_samples = 16, 
    spacecraft = 'MMS', 
    model = model_path
)

calc.plot_log_results(
    predictions_arr = 'results/' + model_path + 'mms_outputs_validate_predict.npy', 
    observed_arr = 'data_processed/mms/mms_clean_outputs_validate.npy',
    no_samples = 16, 
    spacecraft = 'MMS_log', 
    model = model_path
)


