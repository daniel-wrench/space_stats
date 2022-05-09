# Plot results of neural network on validation set
# Use these to inform the choice of ANN architecture

model_name = 'may_6/mod_2/'

#############################################################################

import calculate_stats as calc

calc.plot_validation_error(path = 'results/' + model_name)

calc.plot_results(
    predictions_arr = 'results/' + model_name + 'outputs_validate_predict.npy', 
    observed_arr = 'data_processed/psp/psp_clean_outputs_validate.npy',
    no_samples = 5, 
    spacecraft = 'PSP', 
    model = model_name
)

calc.plot_results(
    predictions_arr = 'results/' + model_name + 'outputs_validate_predict.npy', 
    observed_arr = 'data_processed/psp/psp_clean_outputs_validate.npy',
    no_samples = 5, 
    spacecraft = 'PSP_log', 
    model = model_name,
    log = True
)

print("FINISHED PLOTTING")

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


