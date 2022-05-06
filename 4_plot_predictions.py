# Plot results of neural network on validation set
# Use these to evaluate to the hyperparameters(?)

import calculate_stats as calc

model_name = 'may_6/mod_1/'

calc.plot_validation_error(path = 'results/' + model_name)

calc.plot_results(
    predictions_arr = 'results/' + model_name + 'outputs_validate_predict.npy', 
    observed_arr = 'data_processed/psp/psp_clean_outputs_validate.npy',
    no_samples = 16, 
    spacecraft = 'PSP', 
    model = model_name
)

calc.plot_log_results(
    predictions_arr = 'results/' + model_name + 'outputs_validate_predict.npy', 
    observed_arr = 'data_processed/psp/psp_clean_outputs_validate.npy',
    no_samples = 16, 
    spacecraft = 'PSP_log', 
    model = model_name
)

# PLOTTING PREDICTIONS ON TEST SET (do not using for choosing hyperparameters)

# calc.plot_results(
#     predictions_arr = 'results/' + model_name + 'psp_outputs_test_predict.npy', 
#     observed_arr = 'data_processed/psp/psp_clean_outputs_test.npy',
#     no_samples = 16, 
#     spacecraft = 'PSP', 
#     model = model_name
# )

# calc.plot_log_results(
#     predictions_arr = 'results/' + model_name + 'psp_outputs_test_predict.npy', 
#     observed_arr = 'data_processed/psp/psp_clean_outputs_test.npy',
#     no_samples = 16, 
#     spacecraft = 'PSP_log',
#     model = model_name
# )

# calc.plot_results(
#     predictions_arr = 'results/' + model_name + 'mms_outputs_test_predict.npy', 
#     observed_arr = 'data_processed/mms/mms_clean_outputs_test.npy',
#     no_samples = 16, 
#     spacecraft = 'MMS', 
#     model = model_name
# )

# calc.plot_log_results(
#     predictions_arr = 'results/' + model_name + 'mms_outputs_test_predict.npy', 
#     observed_arr = 'data_processed/mms/mms_clean_outputs_test.npy',
#     no_samples = 16, 
#     spacecraft = 'MMS_log', 
#     model = model_name
# )


