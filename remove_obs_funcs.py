
#####These functions remove observations at random (individually or in chunks) from a dated DataFrame (also available for arrays)

# https://docs.python.org/3/library/random.html#module-random 

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

#DataFrame versions (WORKING)

def remove_obs_df(df, percent):
    num_obs = percent * len(df)
    remove = random.sample(range(len(df)), int(num_obs))
    final_dataset = df.drop(df.index[remove])

    print("The indexes of the removed observations are:")
    print(remove)
    print("The proportion of data removed is:", format(1 - len(final_dataset)/len(df)))
    return final_dataset


#FOR EACH CHUNK
#initialise y = 0
#while y = 0:
#   intialise starting point, calculate range of indices
#   check if any of these values overlap with what is already there. If yes, y still = 0. If not, y = 1
#append results to all_removes

def remove_chunks_df(df, proportion, chunks, sigma):
    '''Remove randomly-sized chunks from random points in a dataframe


     Parameters:
     
        df = DataFrame

        proportion = total proportion (decimal) of observations to remove

        chunks = number of chunks of observations to remove

        sigma = variance of sizes of chunks. Chunk sizes are drawn from a normal distribution with mean = len(df)*proportion/chunks and standard deviation = sigma*0.341*2*mean.
        If this is too large the function will return an error due to negative chunk sizes being selected
    '''
    num_obs = proportion * len(df)
    mean_obs = num_obs/chunks
    std = sigma*0.341*2*mean_obs
    all_removes = []
    for i in range(chunks) :
        num_obs = round(random.gauss(mu = mean_obs, sigma = std))
        #Comment out the line above and replace num_obs below with mean_obs to revert to equal sized chunks
        if num_obs < 0:
            raise Exception('sigma too high, got negative obs')
        y = 0
        while y == 0:
            start = random.randrange(start = num_obs, stop = len(df) - num_obs) #Starting point for each removal should be far enough from the start and end of the series
            remove = np.arange(start, start + num_obs)
            # Check if any removes values are already in all_removes array
            x = 0
            for i in range(len(remove)):
                if remove[i] in all_removes :
                    x = 1
            if x > 0:
#                print("This chunk overlaps with another chunk")
                y = 0 #In this case, random starting point will run again and chunk will be re-selected
            else:
#                print("This chunk does not overlap with another chunk")
                y = 1 #In this case, will proceed to creating next chunk
        all_removes.extend(remove)
        
    all_removes = [int(x) for x in all_removes] #Converting decimals to integers
    final_dataset = df.drop(df.index[all_removes])

#    print("The indexes of the removed observations are:")
#    print(all_removes)
    print("The proportion of data removed is:", format(1 - len(final_dataset)/len(df)))
    return final_dataset

###TESTING FUNCTION
# returns = np.random.normal(loc=0.02, scale=0.05, size=1000)
# df = pd.DataFrame(returns)
# dates = pd.date_range('2011', periods = len(returns))
# df = df.set_index(dates)

# df.plot()
# plt.show()

# new_data_2 = remove_chunks_df(df, 0.2, 5, 0.5) 
# new_data_2 = new_data_2.resample('D').mean()
# new_data_2.plot()
# plt.show()

#Array versions

def remove_obs_array(array, percent):
    num_obs = percent * len(array)
    remove = random.sample(range(len(array)), int(num_obs))
    return np.delete(array, remove)

def remove_chunks_array(array, percent, chunks):
    num_obs = percent * len(array)
    obs_per_chunk = num_obs/chunks
    all_removes = []
    for i in range(chunks) :
        
        start = random.randrange(len(array))
        remove = np.arange(start, start + obs_per_chunk)
        all_removes.extend(remove)
    all_removes = [int(x) for x in all_removes]
    print(all_removes)
    return np.delete(array, all_removes)


