import pandas as pd
import numpy as np
from spacepy import pycdf
import matplotlib.pyplot as plt
import datetime

def read_cdfs(filelist, variables, mask=None):
   """
      Read a list of cdf files, and create a dictionary with their data
      combined in the same order that the files are listed.

      filelist: List of files to read 

      variables: Dictionary with the syntax: {'variable_name':tuple}
                 where tuple is (0) for a one dimensional time series, 
                 and (0,n) where n is the number of columns in the variable
                 
                 If all variables are simple one dimensional time series,
                 they can be passed simply as a list or tuple of strings:
                 e.g. ['Epoch','br','bt','bn'] etc.

      mask: variable name and column to use for mask as a list or tuple
            e.g. for sweap on psp: mask=['DQF',0]

      returns a ditionary with variable names with corresponding data read
      from cdf files

      example:
         d=read_cdfs(['spp_swp_spc_l3i_20181031_v05.cdf',\
                      'spp_swp_spc_l3i_20181101_v07.cdf'],\
            {'vp_moment_RTN':(0,3),'np_moment':(0),'Epoch':(0)}
   """
   from spacepy import pycdf

   if type(variables) in (list,tuple):
      vardict={}
      for vrbl in variables: vardict[vrbl]=(0)
   else:
      vardict=variables.copy()

#create empty numpy arrays
   dictionary={} 
   for j in vardict.keys():
      dictionary[j]=np.empty(vardict[j])

# read data from cdf files and append the arrays.
   for i in filelist:
      print('reading file ',i)
      # Opne CDF file
      d = pycdf.CDF(i)
      # Read in the mask
      if mask is not None:
         if (type(mask) is str):
            mask=pycdf.VarCopy(d[mask])
         elif (type(mask) in (list,tuple)):
            if len(mask) == 1:
               mask=pycdf.VarCopy(d[mask[0]])
            else:
               mask=pycdf.VarCopy(d[mask[0]])[:,mask[1]]
         mask=np.array(mask)
         dictionary['mask']=mask

      # Loop over all variables
      for j in vardict.keys():
         # Read j into a temporary buffer
         tmp=pycdf.VarCopy(d[j])
         # Set bad data to np.NaN
         if 'FILLVAL' in tmp.attrs:
            tmp[np.where(tmp==tmp.attrs['FILLVAL'])] = np.NaN
         # create mask appropriate fot the variable shape
         if mask is not None: 
            if len(tmp.shape) > 1:
               mmask=np.empty(tmp.shape)
               for i in range(tmp.shape[1]): mmask[:,i]=mask
            else:
               mmask=mask
            tmp=np.ma.masked_array(data=tmp,mask=mmask,fill_value=np.NaN)
         # append the dictionary
         dictionary[j]=np.append(dictionary[j], tmp,axis=0)
   print('Done reading data')
   return dictionary

