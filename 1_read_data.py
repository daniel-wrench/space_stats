
##############################################################################

# DATA PROCESSING PART 1: READING MAGNETIC FIELD DATASETS

###################### Daniel Wrench, April 2022 #########################

# This is submitted as a job to the Raapoi cluster via a .sh script
# The outputs are fed into Part 2: 2_process_data.py

##############################################################################

# Loading packages, including those on Git in ~bin/python_env

import data_import_funcs as data_import
import random
import datetime
import pandas as pd

##############################################################################

# Reading in data from CDF files, resampling to correct frequency and checking for missing data

print(datetime.datetime.now())

random.seed(5)

# PSP DATA: TAKES ~ 7 MINUTES TO READ

print("\n\nREADING PSP DATA \n")

psp_data = data_import.read_cdfs([
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110100_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110106_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110112_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110118_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110200_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110206_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110212_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110218_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110300_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110306_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110312_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110318_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110400_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110406_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110412_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110418_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110500_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110506_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110512_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110518_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110600_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110606_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110612_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110618_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110700_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110706_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110712_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110718_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110800_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110806_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110812_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110818_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110900_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110906_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110912_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018110918_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111000_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111006_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111012_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111018_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111100_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111106_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111112_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111118_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111200_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111206_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111212_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111218_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111300_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111306_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111312_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111318_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111400_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111406_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111412_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111418_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111500_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111506_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111512_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111518_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111600_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111606_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111612_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111618_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111700_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111706_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111712_v01.cdf",
    "data_raw/psp/psp_fld_l2_mag_rtn_2018111718_v01.cdf"
],
    {'epoch_mag_RTN': (0), 'psp_fld_l2_mag_RTN': (0, 3), 'label_RTN': (0, 3)})

psp_data_ready = data_import.extract_components(
    psp_data, var_name='psp_fld_l2_mag_RTN', label_name='label_RTN', time_var='epoch_mag_RTN', dim=3)

psp_df = pd.DataFrame(psp_data_ready)

psp_df['Time'] = pd.to_datetime(
    '2000-01-01 12:00') + pd.to_timedelta(psp_df['epoch_mag_RTN'], unit='ns')
psp_df = psp_df.drop(columns='epoch_mag_RTN').set_index('Time')

print("\nRaw data (before re-sampling):")
print(psp_df.head())
print("\nLength of raw data:")
print(psp_df.notnull().sum())
print("\nNumber of missing values:")
print(psp_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
psp_df_resampled = psp_df.resample('0.75S').mean()

print("\nRe-sampled data:")
print(psp_df_resampled.head())
print("\nLength of re-sampled data")
print(psp_df_resampled.notnull().sum())
print("\nNumber of missing values:")
print(psp_df_resampled.isnull().sum())

# Saving final dataframe to directory
psp_df_resampled.to_pickle("data_processed/psp/psp_df_1.pkl")


# 2ND PSP INTERVAL

psp_data = data_import.read_cdfs([
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112100_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112106_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112112_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112118_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112200_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112206_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112212_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112218_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112300_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112306_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112312_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112318_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112400_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112406_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112412_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112418_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112500_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112506_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112512_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112518_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112600_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112606_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112612_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112618_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112700_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112706_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112712_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112718_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112800_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112806_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112812_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112818_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112900_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112906_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112912_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018112918_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018113000_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018113006_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018113012_v01.cdf",
    "data_raw\psp\psp_fld_l2_mag_rtn_2018113018_v01.cdf"
],
    {'epoch_mag_RTN': (0), 'psp_fld_l2_mag_RTN': (0, 3), 'label_RTN': (0, 3)})

psp_data_ready = data_import.extract_components(
    psp_data, var_name='psp_fld_l2_mag_RTN', label_name='label_RTN', time_var='epoch_mag_RTN', dim=3)

psp_df = pd.DataFrame(psp_data_ready)

psp_df['Time'] = pd.to_datetime(
    '2000-01-01 12:00') + pd.to_timedelta(psp_df['epoch_mag_RTN'], unit='ns')
psp_df = psp_df.drop(columns='epoch_mag_RTN').set_index('Time')

print("\nRaw data (before re-sampling):")
print(psp_df.head())
print("\nLength of raw data:")
print(psp_df.notnull().sum())
print("\nNumber of missing values:")
print(psp_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
psp_df_resampled = psp_df.resample('0.75S').mean()

print("\nRe-sampled data:")
print(psp_df_resampled.head())
print("\nLength of re-sampled data")
print(psp_df_resampled.notnull().sum())
print("\nNumber of missing values:")
print(psp_df_resampled.isnull().sum())

# Saving final dataframe to directory
psp_df_resampled.to_pickle("data_processed/psp/psp_df_2.pkl")

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
    "data_raw/mms/mms1_fgm_brst_l2_20171226064953_v5.117.3.cdf"
],
    {'Epoch': (0), 'mms1_fgm_b_dmpa_brst_l2': (0, 4), 'label_b_gse': (0, 4)})

mms_data = data_import.extract_components(
    mms_data_raw, var_name='mms1_fgm_b_dmpa_brst_l2', label_name='label_b_gse', time_var='Epoch', dim=4)
mms_df = pd.DataFrame(mms_data)
mms_df['Time'] = pd.to_datetime(
    '2000-01-01 12:00') + pd.to_timedelta(mms_df['Epoch'], unit='ns')
mms_df = mms_df.drop(columns='Epoch').set_index('Time')
mms_df = mms_df.drop('Bt', axis=1)

print("\nRaw data (before re-sampling):")
print(mms_df.head())
print("\nLength of raw data:")
print(mms_df.notnull().sum())
print("\nNumber of missing values:")
print(mms_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
mms_df_resampled = mms_df.resample('0.008S').mean()

print("\nRe-sampled data:")
print(mms_df_resampled.head())
print("\nLength of re-sampled data")
print(mms_df_resampled.notnull().sum())
print("\nNumber of missing values:")
print(mms_df_resampled.isnull().sum())

# Saving final dataframe to directory
mms_df_resampled.to_pickle("data_processed/mms/mms_df_1.pkl")


# 2ND MMS INTERVAL

mms_data_raw = data_import.read_cdfs([
    "data_raw/mms/mms1_fgm_brst_l2_20171226194913_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226195133_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226195353_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226195613_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226195833_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226200053_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226200313_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226200543_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226200803_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226201023_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226201243_v5.118.0.cdf",
    "data_raw/mms/mms1_fgm_brst_l2_20171226201503_v5.118.0.cdf"
],
    {'Epoch': (0), 'mms1_fgm_b_dmpa_brst_l2': (0, 4), 'label_b_gse': (0, 4)})

mms_data = data_import.extract_components(
    mms_data_raw, var_name='mms1_fgm_b_dmpa_brst_l2', label_name='label_b_gse', time_var='Epoch', dim=4)
mms_df = pd.DataFrame(mms_data)
mms_df['Time'] = pd.to_datetime(
    '2000-01-01 12:00') + pd.to_timedelta(mms_df['Epoch'], unit='ns')
mms_df = mms_df.drop(columns='Epoch').set_index('Time')
mms_df = mms_df.drop('Bt', axis=1)

print("\nRaw data (before re-sampling):")
print(mms_df.head())
print("\nLength of raw data:")
print(mms_df.notnull().sum())
print("\nNumber of missing values:")
print(mms_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
mms_df_resampled = mms_df.resample('0.008S').mean()

print("\nRe-sampled data:")
print(mms_df_resampled.head())
print("\nLength of re-sampled data")
print(mms_df_resampled.notnull().sum())
print("\nNumber of missing values:")
print(mms_df_resampled.isnull().sum())

# Saving final dataframe to directory
mms_df_resampled.to_pickle("data_processed/mms/mms_df_2.pkl")


# 3rd MMS INTERVAL

mms_data_raw = data_import.read_cdfs([
    "data_raw\mms\mms1_fgm_brst_l2_20180108040003_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108040233_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108040503_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108040733_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108041003_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108041233_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108041503_v5.138.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108041733_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108042003_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108042233_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108042503_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108042733_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108043003_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108043233_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108043503_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108043733_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108044003_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108044233_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108044503_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108044733_v5.138.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108045003_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108045233_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108045503_v5.119.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20180108045733_v5.119.0.cdf"
],
    {'Epoch': (0), 'mms1_fgm_b_dmpa_brst_l2': (0, 4), 'label_b_gse': (0, 4)})

mms_data = data_import.extract_components(
    mms_data_raw, var_name='mms1_fgm_b_dmpa_brst_l2', label_name='label_b_gse', time_var='Epoch', dim=4)
mms_df = pd.DataFrame(mms_data)
mms_df['Time'] = pd.to_datetime(
    '2000-01-01 12:00') + pd.to_timedelta(mms_df['Epoch'], unit='ns')
mms_df = mms_df.drop(columns='Epoch').set_index('Time')
mms_df = mms_df.drop('Bt', axis=1)

print("\nRaw data (before re-sampling):")
print(mms_df.head())
print("\nLength of raw data:")
print(mms_df.notnull().sum())
print("\nNumber of missing values:")
print(mms_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
mms_df_resampled = mms_df.resample('0.008S').mean()

print("\nRe-sampled data:")
print(mms_df_resampled.head())
print("\nLength of re-sampled data")
print(mms_df_resampled.notnull().sum())
print("\nNumber of missing values:")
print(mms_df_resampled.isnull().sum())

# Saving final dataframe to directory
mms_df_resampled.to_pickle("data_processed/mms/mms_df_3.pkl")


# 4th MMS INTERVAL

mms_data_raw = data_import.read_cdfs([
    "data_raw\mms\mms1_fgm_brst_l2_20171224012233_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224012453_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224012723_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224012753_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224013003_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224013213_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224013413_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224013623_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224013953_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224014143_v5.117.3.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224014323_v5.117.0.cdf",
    "data_raw\mms\mms1_fgm_brst_l2_20171224014713_v5.117.3.cdf"
],
    {'Epoch': (0), 'mms1_fgm_b_dmpa_brst_l2': (0, 4), 'label_b_gse': (0, 4)})

mms_data = data_import.extract_components(
    mms_data_raw, var_name='mms1_fgm_b_dmpa_brst_l2', label_name='label_b_gse', time_var='Epoch', dim=4)
mms_df = pd.DataFrame(mms_data)
mms_df['Time'] = pd.to_datetime(
    '2000-01-01 12:00') + pd.to_timedelta(mms_df['Epoch'], unit='ns')
mms_df = mms_df.drop(columns='Epoch').set_index('Time')
mms_df = mms_df.drop('Bt', axis=1)

print("\nRaw data (before re-sampling):")
print(mms_df.head())
print("\nLength of raw data:")
print(mms_df.notnull().sum())
print("\nNumber of missing values:")
print(mms_df.isnull().sum())

# Original freq is 0.007s. Resampling to get appropriate number of correlation times in 10,000 points
mms_df_resampled = mms_df.resample('0.008S').mean()

print("\nRe-sampled data:")
print(mms_df_resampled.head())
print("\nLength of re-sampled data")
print(mms_df_resampled.notnull().sum())
print("\nNumber of missing values:")
print(mms_df_resampled.isnull().sum())

# Saving final dataframe to directory
mms_df_resampled.to_pickle("data_processed/mms/mms_df_4.pkl")


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
