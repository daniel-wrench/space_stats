####### ANALYSING AND PLOTTING RESULTS OF NEURAL NETWORK ######

#NB  has been removed due to standardisation of input series

# As well as this script for analysing and visualisation of my final model,
# I have a short script to run in the Raapoi terminal to view a grid of 
# model predictions against expected curves to quickly evaluate each model.

import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numerictypes import sctype2char
import pandas as pd
pd.set_option('display.max_columns', 10)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.formula.api import ols
import seaborn as sns

#######################################

# Importing test data and results

## PSP: 39 outputs gapped 5 times each

psp_clean_inputs_test = np.load(file = 'oct_10_results/datasets/psp_clean_inputs_test.npy')
psp_clean_outputs_test = np.load(file = 'oct_10_results/datasets/psp_clean_outputs_test.npy')

psp_gapped_inputs_test = np.load(file = 'oct_10_results/datasets/psp_gapped_inputs_test.npy')
psp_gapped_outputs_test = np.load(file = 'oct_10_results/datasets/psp_gapped_outputs_test.npy')

psp_filled_inputs_test = np.load(file = 'oct_10_results/datasets/psp_filled_inputs_test.npy')
psp_filled_outputs_test = np.load(file = 'oct_10_results/datasets/psp_filled_outputs_test.npy')

psp_lint_inputs_test = np.load(file = 'oct_10_results/datasets/psp_lint_inputs_test.npy')
psp_lint_outputs_test = np.load(file = 'oct_10_results/datasets/psp_lint_outputs_test.npy')

psp_gapped_inputs_test_prop_removed = np.load(file = 'oct_10_results/psp_gapped_inputs_test_prop_removed.npy')

psp_outputs_predictions = np.load('oct_10_results/datasets/psp_outputs_test_predict.npy')

### FOR SAVING ARRAYS OF ONLY THE INTERVALS TO PLOT (saves space when sharing)

# Select the intervals to plot
psp_indices = [12, 90, 93, 15]

psp_clean_inputs_test_selected = psp_clean_inputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_clean_inputs_test_selected', arr = psp_clean_inputs_test_selected)

psp_clean_outputs_test_selected = psp_clean_outputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_clean_outputs_test_selected', arr = psp_clean_outputs_test_selected)

psp_gapped_inputs_test_selected = psp_gapped_inputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_gapped_inputs_test_selected', arr = psp_gapped_inputs_test_selected)

psp_gapped_outputs_test_selected = psp_gapped_outputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_gapped_outputs_test_selected', arr = psp_gapped_outputs_test_selected)

psp_filled_inputs_test_selected = psp_filled_inputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_filled_inputs_test_selected', arr = psp_filled_inputs_test_selected)

psp_filled_outputs_test_selected = psp_filled_outputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_filled_outputs_test_selected', arr = psp_filled_outputs_test_selected)

psp_lint_inputs_test_selected = psp_lint_inputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_lint_inputs_test_selected', arr = psp_lint_inputs_test_selected)

psp_lint_outputs_test_selected = psp_lint_outputs_test[psp_indices]
np.save(file = 'march_6_22_plots/psp_lint_outputs_test_selected', arr = psp_lint_outputs_test_selected)

psp_outputs_test_predict_selected = psp_outputs_predictions[psp_indices]
np.save(file = 'march_6_22_plots/psp_outputs_test_predict_selected', arr = psp_outputs_test_predict_selected)

##############################

## PSP 2020: 27 outputs gapped 5 times each

# psp_clean_inputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_clean_inputs_test_2020.npy')
# psp_clean_outputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_clean_outputs_test_2020.npy')

# psp_gapped_inputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_gapped_inputs_test_2020.npy')
# psp_gapped_outputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_gapped_outputs_test_2020.npy')

# psp_filled_inputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_filled_inputs_test_2020.npy')
# psp_filled_outputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_filled_outputs_test_2020.npy')

# psp_lint_inputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_lint_inputs_test_2020.npy')
# psp_lint_outputs_test_2020 = np.load(file = 'nov_5_results/datasets/psp_lint_outputs_test_2020.npy')

# psp_gapped_inputs_test_2020_prop_removed = np.load(file = 'nov_5_results/datasets/psp_gapped_inputs_test_prop_removed_2020.npy')

# psp_outputs_predictions_2020 = np.load('nov_5_results/datasets/psp_2020_outputs_test_predict.npy')


## MMS: 29 outputs gapped 5 times each

# mms_clean_inputs_test = np.load(file = 'march_6_22_plots/datasets/mms_clean_inputs_test.npy')
# mms_clean_outputs_test = np.load(file = 'march_6_22_plots/datasets/mms_clean_outputs_test.npy')

# mms_gapped_inputs_test = np.load(file = 'march_6_22_plots/datasets/mms_gapped_inputs_test.npy')
# mms_gapped_outputs_test = np.load(file = 'march_6_22_plots/datasets/mms_gapped_outputs_test.npy')

# mms_filled_inputs_test = np.load(file = 'march_6_22_plots/datasets/mms_filled_inputs_test.npy')
# mms_filled_outputs_test = np.load(file = 'march_6_22_plots/datasets/mms_filled_outputs_test.npy')

# mms_lint_inputs_test = np.load(file = 'march_6_22_plots/datasets/mms_lint_inputs_test.npy')
# mms_lint_outputs_test = np.load(file = 'march_6_22_plots/datasets/mms_lint_outputs_test.npy')

# mms_gapped_inputs_test_prop_removed = np.load(file = 'march_6_22_plots/mms_gapped_inputs_test_prop_removed.npy')

# mms_outputs_predictions = np.load('march_6_22_plots/datasets/mms_outputs_test_predict.npy')

#######################################

# Plotting all predicted vs. expected structure functions

# fig, ax = plt.subplots(2,2, sharey=True, figsize = [7, 7])
# [ax[0,0].plot(psp_clean_outputs_test[i]) for i in range(len(psp_clean_outputs_test))]
# [ax[0,1].plot(psp_outputs_predictions[i]) for i in range(len(psp_outputs_predictions))]
# ax[0,0].set(title = "PSP true structure functions")
# ax[0,1].set(title = "PSP predicted structure functions")

# [ax[1,0].plot(mms_clean_outputs_test[i]) for i in range(len(mms_clean_outputs_test))]
# [ax[1,1].plot(mms_outputs_predictions[i]) for i in range(len(mms_outputs_predictions))]
# ax[1,0].set(title = "MMS true structure functions")
# ax[1,1].set(title = "MMS predicted structure functions")

# ax[0,0].semilogx()
# ax[0,0].semilogy()
# ax[0,1].semilogx()
# ax[0,1].semilogy()
# ax[1,0].semilogx()
# ax[1,0].semilogy()
# ax[1,1].semilogx()
# ax[1,1].semilogy()

# plt.savefig('march_6_22_plots/plots/all_sfs.png')
# #plt.show()

# # Plotting all predicted vs. expected structure functions FOR DIFFERENT PSP YEARS

# fig, ax = plt.subplots(2,2, sharey=True, figsize = [7, 7])
# [ax[0,0].plot(psp_clean_outputs_test[i]) for i in range(130)]
# [ax[0,1].plot(psp_outputs_predictions[i]) for i in range(130)]
# ax[0,0].set(title = "PSP true structure functions")
# ax[0,1].set(title = "PSP predicted structure functions")

# [ax[1,0].plot(psp_clean_outputs_test_2020[i]) for i in range(130)]
# [ax[1,1].plot(psp_outputs_predictions_2020[i]) for i in range(130)]
# ax[1,0].set(title = "PSP 2020 true structure functions")
# ax[1,1].set(title = "PSP 2020 predicted structure functions")

# ax[0,0].semilogx()
# ax[0,0].semilogy()
# ax[0,1].semilogx()
# ax[0,1].semilogy()
# ax[1,0].semilogx()
# ax[1,0].semilogy()
# ax[1,1].semilogx()
# ax[1,1].semilogy()

# ax[0,0].semilogx()
# ax[0,0].semilogy()
# ax[0,1].semilogx()
# ax[0,1].semilogy()
# ax[1,0].semilogx()
# ax[1,0].semilogy()
# ax[1,1].semilogx()
# ax[1,1].semilogy()

# ax[0,0].set(ylim = [0.005, 15])

# plt.savefig('nov_5_results/plots/all_psp_sfs.png')
# #plt.show()


#######################################

# Get MSE between two curves
def calc_mse(curve1, curve2):
    mse = np.sum((curve1-curve2)**2)/len(curve1)
    if mse == np.inf:
      mse = np.nan
    return(mse) 

# Get MAPE between two curves
def calc_mape(actual, forecast):
    actual = actual + 0.000001 # Have to add this so there is no division by 0
    mape = np.sum(np.abs((actual-forecast)/actual))/len(actual)
    if mape == np.inf:
      mape = np.nan
    return(mape) 

# Calculate metrics for every interval
def calc_all_Methods(gapped_inputs, gapped_outputs, clean_outputs, filled_outputs, lint_outputs, predictions):

  missing = []
  gapped_mape = []
  pred_mape = []
  filled_mape = []
  lint_mape = []

  pred_mse = []
  gapped_mse = []
  filled_mse = []
  lint_mse = []

  # Calculate mape and % missingness for every interval in the test set
  for i in range(len(clean_outputs)):

    missing.append(sum(np.isnan(gapped_inputs[i][0]))/len(gapped_inputs[i][0])) 
    # We actually already have the above from the processing data stage - can compare with to check

    pred_mape.append(calc_mape(clean_outputs[i], predictions[i]))
    gapped_mape.append(calc_mape(clean_outputs[i], gapped_outputs[i]))
    filled_mape.append(calc_mape(clean_outputs[i], filled_outputs[i]))
    lint_mape.append(calc_mape(clean_outputs[i], lint_outputs[i]))

    pred_mse.append(calc_mse(clean_outputs[i], predictions[i]))
    gapped_mse.append(calc_mse(clean_outputs[i], gapped_outputs[i]))
    filled_mse.append(calc_mse(clean_outputs[i], filled_outputs[i]))
    lint_mse.append(calc_mse(clean_outputs[i], lint_outputs[i]))

  all_metrics = pd.DataFrame({
    'missingness': missing, 

    'GAPPED_MAPE': gapped_mape,
    'ANN_MAPE': pred_mape, 
    'MIMP_MAPE': filled_mape, 
    'LINT_MAPE': lint_mape,

    'GAPPED_MSE': gapped_mse,
    'ANN_MSE': pred_mse, 
    'MIMP_MSE': filled_mse, 
    'LINT_MSE': lint_mse})

  return all_metrics

psp_test_metrics = calc_all_Methods(
  psp_gapped_inputs_test, 
  psp_gapped_outputs_test,
  psp_clean_outputs_test, 
  psp_filled_outputs_test, 
  psp_lint_outputs_test, 
  psp_outputs_predictions)

psp_test_metrics['spacecraft'] = 'PSP'
#psp_test_metrics.head()

psp_test_metrics.to_csv("march_6_22_plots/psp_test_metrics.csv")


# psp_test_2020_metrics = calc_all_Methods(
#   psp_gapped_inputs_test_2020, 
#   psp_gapped_outputs_test_2020,
#   psp_clean_outputs_test_2020, 
#   psp_filled_outputs_test_2020, 
#   psp_lint_outputs_test_2020, 
#   psp_outputs_predictions_2020)

# psp_test_2020_metrics['spacecraft'] = 'PSP'
# psp_test_2020_metrics.head()

# psp_test_2020_metrics.to_csv("nov_5_results/psp_test_2020_metrics.csv")

# psp_clean_inputs_test[psp_indices[1]][0]
# psp_pregapped_in[1][0]


psp_2020_indices = [3, 30, 7, 34]

# Checking they are the same
#psp_clean_inputs_test[psp_indices]

# mms_test_metrics = calc_all_Methods(
#   mms_gapped_inputs_test, 
#   mms_gapped_outputs_test,
#   mms_clean_outputs_test, 
#   mms_filled_outputs_test, 
#   mms_lint_outputs_test, 
#   mms_outputs_predictions)

# mms_test_metrics['spacecraft'] = 'mms'
# mms_test_metrics.head()

# mms_test_metrics.to_csv("march_6_22_plots/mms_test_metrics.csv")

# # Note MMS has same removal %s but different number of intervals, so same indices as PSP will be same missingness but not duplicatse

# mms_indices = [53, 82, 63, 92]

# Checking they are the same
#mms_clean_inputs_test[mms_indices]


###########
# INSERT HERE - for spacecraft in ["PSP", "MMS"]
##########

def get_regression_output(metrics, spacecraft, indices):

  print("\nMeans over all " + spacecraft + " intervals")
  print(metrics.mean())
  print("\nMethodistics for selected " + spacecraft + " intervals")
  print(metrics.iloc[indices])

  print("\nPSP pair-wise Pearson correlations \n\n", metrics.corr())

  print("\n\n" + spacecraft + " regressions on proportion missing")
  for i in metrics.columns[1:-2]:
    print("\n\n{} ~ missingness".format(i))
    lm = ols("{} ~ missingness".format(i), data=metrics)
    lm = lm.fit()
    print("\nCoefficient estimates\n", lm.params)
    print("\nP-values\n", lm.pvalues)

  metrics_long = pd.melt(metrics, id_vars = ['spacecraft', 'missingness'], var_name = 'Method', value_name='value')
  metrics_long.head()

  # Producing scatterplots and linear regression plots

  for metric in ["MAPE", "MSE"]: #NB: For some reason the updated fonts only apply to the second metric in the list
    data_to_plot = metrics_long[metrics_long.Method.isin(['MIMP_' + metric, 'LINT_' + metric, 'ANN_' + metric, 'gapped_' + metric])]
    data_to_plot.missingness = data_to_plot.missingness * 100

    fig, ax = plt.subplots(1,4, sharey=True, figsize = (12, 3))
    # plt.style.use('fivethirtyeight')
    plt.rcParams['font.size']=14
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['axes.titlepad'] = -15  # pad is in points...
    # plt.rcParams['font.sans-serif'] = ['Tahoma']
    # scatter_plot = sns.lmplot(x = 'missingness', 
    #             y = 'value', 
    #             data = data_to_plot, 
    #             hue = 'Method', 
    #             col = 'Method',
    #             height=3,
    #             fit_reg=True, ax = [ax1, ax2, ax3])
    sns.regplot(x = 'missingness', 
                y = 'value', 
                color="#e78ac3",
                scatter_kws={'alpha':0.2},
                data = data_to_plot[data_to_plot["Method"]=="ANN_"+metric], ax = ax[0])
    sns.regplot(x = 'missingness', 
                y = 'value', 
                color = "#66c2a5",
                scatter_kws={'alpha':0.2},
                line_kws={'linestyle':'dotted'},
                data = data_to_plot[data_to_plot["Method"]=="MIMP_"+metric], ax = ax[1])
    sns.regplot(x = 'missingness', 
                y = 'value', 
                color = "#377eb8",
                scatter_kws={'alpha':0.2},
                line_kws={'linestyle':'dashed'},
                data = data_to_plot[data_to_plot["Method"]=="LINT_"+metric], ax = ax[2])
    sns.regplot(x = 'missingness', 
                y = 'value', 
                color="#e78ac3",
                marker='',
                data = data_to_plot[data_to_plot["Method"]=="ANN_"+metric], ax = ax[3]) 
    sns.regplot(x = 'missingness', 
                y = 'value', 
                color = "#66c2a5",
                marker='',
                line_kws={'linestyle':'dotted'},
                data = data_to_plot[data_to_plot["Method"]=="MIMP_"+metric], ax = ax[3])
    sns.regplot(x = 'missingness', 
                y = 'value', 
                color = "#377eb8",
                marker='',
                line_kws={'linestyle':'dashed'},
                data = data_to_plot[data_to_plot["Method"]=="LINT_"+metric], ax = ax[3])
    
    if metric == "MAPE":
      plt.ylim([-0.2,3])
    if metric == "MSE":
      plt.ylim([-2,30])
                
    # ax1.fig.subplots_adjust(top=0.8, bottom = 0.15, left = 0.08)
    #scatter_plot.fig.suptitle('Scatter plots of ' + spacecraft + ' test intervals: ' +  metric +  ' vs. proportion missing')
    ax[0].set(xlabel = "% missing", ylabel = metric, title = "ANN")
    ax[1].set(xlabel = "% missing", ylabel = "", title = "M-IMP")
    ax[2].set(xlabel = "% missing", ylabel = "", title = "L-INT")
    ax[3].set(xlabel = "% missing", ylabel = "", title = "All")

    for x in ax.flatten(): 
        x.set_yticklabels(x.get_yticks(), rotation = 0)
        x.tick_params(which='both',direction='in')
        x.spines['right'].set_visible(False)
        x.spines['top'].set_visible(False)

    fig.subplots_adjust(hspace=0.11)

    # ax1.set_ylabels(metric)

    #plt.xlim([0.05,0.55])
    #plt.savefig('march_6_22_plots/plots/' + spacecraft + '_scatter_plot' + metric + '.pdf')
    plt.savefig('march_6_22_plots/plots/' + spacecraft + '_scatter_plot' + metric + '.pdf',bbox_inches='tight')
    #plt.show()


    # line_plot = sns.lmplot(x = 'missingness', 
    #             y = 'value', 
    #             markers = '', 
    #             data = data_to_plot, 
    #             hue = 'Method', 
    #             height=3.5,
    #             legend=True)
    # line_plot.fig.subplots_adjust(top=0.8, bottom = 0.15, left = 0.15)
    # #line_plot.fig.suptitle('Regression lines for ' +  spacecraft + ' test intervals \n' + metric + ' vs. proportion missing')
    # line_plot.set_xlabels("Proportion missing")
    # line_plot.set_ylabels(metric)

    # plt.ylim([-0.1,20]) # Previously upper limit 1.5
    # #plt.xlim([0,0.6])
    # plt.savefig('march_6_22_plots/plots/' + spacecraft + '_regression_lines' + metric + '.pdf')
    #plt.show()

# get_regression_output(psp_test_2020_metrics, 'PSP_2020', psp_2020_indices)
get_regression_output(psp_test_metrics, 'PSP', psp_indices)
# get_regression_output(mms_test_metrics, 'MMS', mms_indices)

#### LARGE PLOT COMPARING DIFFERENT INPUTS AND OUTPUTS FOR TWO INTERVALS, GAPPED IN TWO DIFFERENT WAYS ####

def plot_final_results(pregapped_in, out, filled_in, filled_out, lint_in, lint_out, gapped_in, gapped_out, predict, title = 'plot.png', mms = 0, psp_2020 = 0):
# plt.style.use('fivethirtyeight')
  plt.rcParams['font.size']=14
  plt.rcParams['font.family'] = 'monospace'
  plt.rcParams['axes.titlepad'] = -15  # pad is in points...
# plt.rcParams['font.sans-serif'] = ['Tahoma']
  fig, ax = plt.subplots(4,3, sharex='col',figsize = (12,14))

  ax[0,0].plot(pregapped_in[0][0], alpha = 0.8, lw = 0.01, color = '#4daf4a', label = 'Original complete')
  ax[0,0].plot(filled_in[0][0], linestyle = 'dotted', lw = 0.8, color = '#66c2a5', label = 'M-IMP')
  ax[0,0].plot(lint_in[0][0], linestyle = 'dashed', lw = 0.8, color = '#377eb8', label = 'L-INT')
  ax[0,0].plot(gapped_in[0][0], lw = 0.01, color = '#fc8d62', label = 'Gapped')
  ax[0,0].text(0.1, 0.85, 'Interval 1a:\n10% removed', size = 12, bbox = {'facecolor': 'white', 'alpha': 0.7, 'pad': 2, 'edgecolor':'white'}, verticalalignment='top', horizontalalignment='left', transform=ax[0,0].transAxes)

  ax[0,1].plot(out[0], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[0,1].plot(predict[0], lw = 1, color = '#e78ac3', label = 'ANN')
  ax[0,1].plot(filled_out[0], linestyle = 'dotted',  lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[0,1].plot(lint_out[0], linestyle = 'dashed', color = '#377eb8', label = 'L-INT')
  ax[0,1].plot(gapped_out[0], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[0,2].plot(out[0], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[0,2].plot(predict[0], lw = 1, color = '#e78ac3', label = 'ANN output')
  ax[0,2].plot(filled_out[0], linestyle = 'dotted', lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[0,2].plot(lint_out[0], linestyle = 'dashed', lw = 1, color = '#377eb8', label = 'L-INT')
  ax[0,2].plot(gapped_out[0], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[0,2].semilogx()
  ax[0,2].semilogy()

  ax[1,0].plot(pregapped_in[1][0], lw = 0.01, alpha = 0.6, color = '#4daf4a', label = 'Original complete')
  ax[1,0].plot(filled_in[1][0], linestyle = 'dotted', lw = 0.8, color = '#66c2a5', label = 'M-IMP')
  ax[1,0].plot(lint_in[1][0], linestyle = 'dashed', lw = 0.8, color = '#377eb8', label = 'L-INT')
  ax[1,0].plot(gapped_in[1][0], lw = 0.01, color = '#fc8d62', label = 'Gapped')
  ax[1,0].text(0.1, 0.85, 'Interval 1b:\n43% removed', size = 12, bbox = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2}, verticalalignment='top', horizontalalignment='left', transform=ax[1,0].transAxes)

  ax[1,1].plot(out[1], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[1,1].plot(predict[1], lw = 1, color = '#e78ac3', label = 'ANN')
  ax[1,1].plot(filled_out[1], linestyle = 'dotted',  lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[1,1].plot(lint_out[1], linestyle = 'dashed', color = '#377eb8', label = 'L-INT')
  ax[1,1].plot(gapped_out[1], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[1,2].plot(out[1], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[1,2].plot(predict[1], lw = 1, color = '#e78ac3', label = 'ANN output')
  ax[1,2].plot(filled_out[1], linestyle = 'dotted',  lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[1,2].plot(lint_out[1], linestyle = 'dashed',  lw = 1, color = '#377eb8', label = 'L-INT')
  ax[1,2].plot(gapped_out[1], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[1,2].semilogx()
  ax[1,2].semilogy()

  ax[2,0].plot(pregapped_in[2][0], lw = 0.01, alpha = 0.6, color = '#4daf4a', label = 'Original complete')
  ax[2,0].plot(filled_in[2][0], linestyle = 'dotted', lw = 0.8, color = '#66c2a5', label = 'M-IMP')
  ax[2,0].plot(lint_in[2][0], linestyle = 'dashed', lw = 0.8, color = '#377eb8', label = 'L-INT')
  ax[2,0].plot(gapped_in[2][0], lw = 0.01, color = '#fc8d62', label = 'Gapped')
  ax[2,0].text(0.35, 0.85, 'Interval 2a:\n37% removed', size = 12, bbox = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2}, verticalalignment='top', horizontalalignment='left', transform=ax[2,0].transAxes)

  ax[2,1].plot(out[2], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[2,1].plot(predict[2], lw = 0.5, color = '#e78ac3', label = 'ANN')
  ax[2,1].plot(filled_out[2], linestyle = 'dotted',  lw = 0.5, color = '#66c2a5', label = 'M-IMP')
  ax[2,1].plot(lint_out[2], linestyle = 'dashed', lw = 0.5, color = '#377eb8', label = 'L-INT')
  ax[2,1].plot(gapped_out[2], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[2,2].plot(out[2], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[2,2].plot(predict[2], lw = 1, color = '#e78ac3', label = 'ANN output')
  ax[2,2].plot(filled_out[2], linestyle = 'dotted',  lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[2,2].plot(lint_out[2], linestyle = 'dashed',  lw = 1, color = '#377eb8', label = 'L-INT')
  ax[2,2].plot(gapped_out[2], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[2,2].semilogx()
  ax[2,2].semilogy()

  # plt.show()

  ####

  ax[3,0].plot(pregapped_in[3][0], lw = 0.01, alpha = 0.6, color = '#4daf4a', label = 'Original complete')
  ax[3,0].plot(filled_in[3][0], linestyle = 'dotted', lw = 0.8, color = '#66c2a5', label = 'M-IMP')
  ax[3,0].plot(lint_in[3][0], linestyle = 'dashed', lw = 0.8, color = '#377eb8', label = 'L-INT')
  ax[3,0].plot(gapped_in[3][0], lw = 0.01, color = '#fc8d62', label = 'Gapped')
  ax[3,0].text(0.35, 0.85, 'Interval 2b\n87% removed', size = 12, bbox = {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.7, 'pad': 2}, verticalalignment='top', horizontalalignment='left', transform=ax[3,0].transAxes)

  ax[3,1].plot(out[3], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[3,1].plot(predict[3], lw = 1, color = '#e78ac3', label = 'ANN')
  ax[3,1].plot(filled_out[3], linestyle = 'dotted',  lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[3,1].plot(lint_out[3], linestyle = 'dashed', color = '#377eb8', label = 'L-INT')
  ax[3,1].plot(gapped_out[3], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[3,2].plot(out[3], lw = 2.5, color = '#4daf4a', label = 'Original complete')
  ax[3,2].plot(predict[3], lw = 1, color = '#e78ac3', label = 'ANN output')
  ax[3,2].plot(filled_out[3], linestyle = 'dotted', lw = 2, color = '#66c2a5', label = 'M-IMP')
  ax[3,2].plot(lint_out[3], linestyle = 'dashed',  lw = 1, color = '#377eb8', label = 'L-INT')
  ax[3,2].plot(gapped_out[3], lw = 0.1, color = '#fc8d62', label = 'Gapped')

  ax[3,2].semilogx()
  ax[3,2].semilogy()

  if mms == 0: 
    ylab = '$B_R$'
    plot_title = '$B_R$ PSP'
    log_lower_lim = 1e-2
    upper_lim = 7
#   ax[0,1].legend(loc='upper left', prop={'size': 7}) #frameon=True, ncol = 3
 
  if psp_2020 == 1:
    plot_title = '7500 second (15 tNL)\nPSP 2020 intervals'
    upper_lim = 10
#   ax[0,1].legend(loc='lower right', prop={'size': 7}) #frameon=True, ncol = 3

  if mms == 1:
    ylab = '$B_x$'
    plot_title = '80 second (13 tNL)\nMMS intervals'
    upper_lim = 8
    log_lower_lim = 1e-4
#   ax[0,1].legend(loc='lower right', prop={'size': 7}) #frameon=True, ncol = 3

  ax[0,1].set(ylim = [0, upper_lim])
  ax[1,1].set(ylim = [0, upper_lim])
  ax[2,1].set(ylim = [0, upper_lim])
  ax[3,1].set(ylim = [0, upper_lim])

  ax[0,2].set(ylim = [log_lower_lim, 1e1])
  ax[1,2].set(ylim = [log_lower_lim, 1e1])
  ax[2,2].set(ylim = [log_lower_lim, 1e1])
  ax[3,2].set(ylim = [log_lower_lim, 1e1])

  ax[0,0].set( ylabel = '', title = plot_title)
  ax[1,0].set( ylabel = '', title = '')
  ax[2,0].set( ylabel = '', title = '')
  ax[3,0].set( ylabel = '', title = '')

  ax[0,1].set(ylabel = '', title = '$S_2$ linear')
  ax[1,1].set(ylabel = '', title = '')
  ax[2,1].set(ylabel = '', title = '')
  ax[3,1].set(ylabel = '', title = '')
  ax[0,2].set(ylabel = '', title = '$S_2$ log-log')
  ax[1,2].set(ylabel = '', title = '')
  ax[2,2].set(ylabel = '', title = '')
  ax[3,2].set(ylabel = '', title = '')

  ax[3,0].set(xlabel = 'index')
  ax[3,1].set(xlabel = '# lags')
  ax[3,2].set(xlabel = '# lags ')

  ax[0,1].legend(loc='upper right', bbox_to_anchor=(2.0,1.25), ncol=5, prop={'size': 14} ,frameon=False)#, ncol = 3

# ax[0,2].legend(loc='lower right', prop={'size': 7}) #frameon=True, ncol = 3
  
  for x in ax.flatten(): 
    x.set_yticklabels(x.get_yticks(), rotation = 0)
    x.tick_params(which='both',direction='in')
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)

  fig.subplots_adjust(hspace=0.11)

  plt.savefig(title,bbox_inches='tight')
  #plt.show()

# PSP results
plot_final_results(
  psp_clean_inputs_test_selected, 
  psp_clean_outputs_test_selected, 
  psp_filled_inputs_test_selected,
  psp_filled_outputs_test_selected, 
  psp_lint_inputs_test_selected, 
  psp_lint_outputs_test_selected, 
  psp_gapped_inputs_test_selected, 
  psp_gapped_outputs_test_selected, 
  psp_outputs_test_predict_selected, 
  title = 'march_6_22_plots/plots/PSP_case_studies_plot.pdf')

# # PSP 2020 results
# plot_final_results(
#   psp_clean_inputs_test_2020, 
#   psp_clean_outputs_test_2020, 
#   psp_filled_inputs_test_2020,
#   psp_filled_outputs_test_2020, 
#   psp_lint_inputs_test_2020, 
#   psp_lint_outputs_test_2020, 
#   psp_gapped_inputs_test_2020, 
#   psp_gapped_outputs_test_2020, 
#   psp_outputs_predictions_2020, 
#   psp_2020_indices, psp_2020=1,
#   title = 'nov_5_results/plots/psp_2020_case_studies.png')


# # MMS results
# plot_final_results(
#   mms_clean_inputs_test, 
#   mms_clean_outputs_test, 
#   mms_filled_inputs_test,
#   mms_filled_outputs_test, 
#   mms_lint_inputs_test, 
#   mms_lint_outputs_test, 
#   mms_gapped_inputs_test, 
#   mms_gapped_outputs_test, 
#   mms_outputs_predictions, 
#   mms_indices, 
#   title = 'march_6_22_plots/plots/mms_case_studies.png', mms = 1)