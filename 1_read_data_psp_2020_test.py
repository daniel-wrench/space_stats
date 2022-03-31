
##############################################################################

# DATA PROCESSING PART 1: READING MAGNETIC FIELD DATASETS

###################### Daniel Wrench, September 2021 #########################

# This is being run in the Raapoi terminal
# The outputs are fed into Part 2: 2_process_data.py

##############################################################################

# Loading packages, including those on Git in ~bin/python_env

import warnings
import space_stats.data_import_funcs as data_import
import space_stats.calculate_stats as calcs
import space_stats.remove_obs_funcs as removal
import TurbAn.Analysis.TimeSeries.OLLibs.F90.ftsa as ftsa
import random
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##############################################################################

# Reading in data from CDF files, resampling to correct frequency and checking for missing data

# PSP DATA TAKES ~ 7 MINUTES TO READ

print(datetime.datetime.now())

random.seed(5)

# PSP

print("\n\nREADING PSP DATA \n")

psp_data = data_import.read_cdfs([
    "data_raw/psp/psp_fld_l2_mag_rtn_2020011700_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011706_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011712_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011718_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011800_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011806_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011812_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011818_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011900_v02.cdf",
                    "data_raw/psp/psp_fld_l2_mag_rtn_2020011906_v02.cdf"
    ],
    {'epoch_mag_RTN':(0), 'psp_fld_l2_mag_RTN':(0,3), 'label_RTN':(0,3)})

psp_data_ready = data_import.extract_components(psp_data, var_name='psp_fld_l2_mag_RTN', label_name='label_RTN', time_var='epoch_mag_RTN', dim=3)

psp_df = pd.DataFrame(psp_data_ready)

psp_df['Time'] = pd.to_datetime('2000-01-01 12:00') + pd.to_timedelta(psp_df['epoch_mag_RTN'], unit= 'ns')
psp_df = psp_df.drop(columns = 'epoch_mag_RTN').set_index('Time')

print("\nRaw data (before re-sampling):")
print(psp_df.head())
print("\nLength of raw data:")
print(psp_df.notnull().sum())
print("\nNumber of missing values:")
print(psp_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
psp_df = psp_df.resample('0.75S').mean() 

psp_df = psp_df["2020-01-17 03:40":"2020-01-19 12:00"]

print("\nRe-sampled data:")
print(psp_df.head())
print("\nLength of re-sampled data")
print(psp_df.notnull().sum())
print("\nNumber of missing values:")
print(psp_df.isnull().sum())

# Saving final dataframe to directory
psp_df.to_pickle("data_processed/psp/psp_df_2020_test.pkl")

#######################################

# MMS

print(datetime.datetime.now())

print("\n\nREADING MMS DATA \n")

mms_data_raw = data_import.read_cdfs([
                      "data_raw/mms/mms1_fgm_brst_l2_20171226061243_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226061513_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226061743_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226062013_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226062233_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226062503_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226062733_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226063003_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226063233_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226063503_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226063733_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226064003_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226064223_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226064453_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226064723_v5.117.3.cdf",
                      "data_raw/mms/mms1_fgm_brst_l2_20171226064953_v5.117.3.cdf"], 
                      {'Epoch':(0), 'mms1_fgm_b_dmpa_brst_l2':(0,4), 'label_b_gse':(0,4)})

mms_data = data_import.extract_components(mms_data_raw, var_name='mms1_fgm_b_dmpa_brst_l2', label_name='label_b_gse', time_var='Epoch', dim=4)
mms_df = pd.DataFrame(mms_data)
mms_df['Time'] = pd.to_datetime('2000-01-01 12:00') + pd.to_timedelta(mms_df['Epoch'], unit= 'ns')
mms_df = mms_df.drop(columns = 'Epoch').set_index('Time')
mms_df = mms_df.drop('Bt', axis = 1)

print("\nRaw data (before re-sampling):")
print(mms_df.head())
print("\nLength of raw data:")
print(mms_df.notnull().sum())
print("\nNumber of missing values:")
print(mms_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
mms_df = mms_df.resample('0.008S').mean() 

print("\nRe-sampled data:")
print(mms_df.head())
print("\nLength of re-sampled data")
print(mms_df.notnull().sum())
print("\nNumber of missing values:")
print(mms_df.isnull().sum())

# Saving final dataframe to directory
mms_df.to_pickle("data_processed/mms/mms_df.pkl")

#######################################

# WIND

# wind_data_raw = data_import.read_cdfs([
# #    "data_raw/wind/wi_h0_mfi_20151104_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151105_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151106_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151107_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151108_v05.cdf",  
#     "data_raw/wind/wi_h0_mfi_20151109_v05.cdf",  
#     "data_raw/wind/wi_h0_mfi_20151110_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151111_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151112_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151113_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151114_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151115_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151116_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151117_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151118_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151119_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151120_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151121_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151122_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151123_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151124_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151125_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151126_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151127_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151128_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151129_v05.cdf",
#     "data_raw/wind/wi_h0_mfi_20151130_v05.cdf",
#     ], {'Epoch3':(0), 'B3F1':(0)})

# #H2 data (with 0.092 frequency) had these specs: 'Epoch':(0), 'BF1':(0). Can also remove dictionary adjustment below for H2 data

# wind_data_raw['Epoch'] = wind_data_raw['Epoch3']
# del wind_data_raw['Epoch3']
# wind_data_raw['BF1'] = wind_data_raw['B3F1']
# del wind_data_raw['B3F1']
# wind_data = data_import.date_1d_dict(wind_data_raw, '5S') #Original freq is 3S. 
# print(len(wind_data))
# print(wind_data['BF1'].isna().sum()/len(wind_data['BF1'])) #0% missing data (3.4% for 3s data)

# wind_data = wind_data['BF1']

# #Clean_subset = wind_data['BF1']['2015-11-04 14:11:32':'2015-11-04 15:58:36']
# #len(clean_subset) #70,000 points long
# #wind = np.load('data_raw/wind/wind_data_clean.npy')



#################################################################################
