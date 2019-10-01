# Example code for partial analysis of simulation box of ammonia borane (a.k.a. AB, NH3BH3) dissolved in tetrahydrofuran (a.k.a. THF, C4H8O)
# See https://en.wikipedia.org/wiki/Ammonia_borane and https://en.wikipedia.org/wiki/Tetrahydrofuran
# Intermolecular bonding between AB and THF predominantly takes the form of hydrogen bonding between the nitrogen bound hydrogen atoms (H1) on AB and the oxygen (O1) on THF
# This particular code shows the methodology used to determine the probability distribution of the number of a certain type of atom surrounding another (in this code, the number of O1s surrounding each H1)

##########################################

# import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# initialise Series for storing and updating probability distribution over each iteration
o1_h1_series = pd.Series()
it_count = 0 # counts the number of simulation frames (or iterations) as the code progresses through the entire simulation file, for tracking purposes

# read in the simulation file chunk-by-chunk, as typical files are far too big to read into memory all at once
chunksize = 29014 # represents the 29012 atoms in each frame of the simulation, as well as the two extra rows to skip in each for formatting reasons
url = 'https://raw.githubusercontent.com/Dingram23/PhD-Codes/master/example_abthf_1_frame.xyz'

for chunk in pd.read_csv(url, skiprows = 2, names = ['first', 'x', 'y', 'z', 'atom type'], delim_whitespace = True, chunksize = chunksize):

    data = chunk.drop('first', axis = 1) # drops unnecessary first column
    data = data.dropna(axis = 0) # drops extra empty rows after each chunk of atom coordinates

    # convert coordinate columns to dtype float
    data.x = data.x.astype(float)
    data.y = data.y.astype(float)
    data.z = data.z.astype(float)

    # create dataframes with only x,y,z coordinates of each atom type of interest
    h1s_only = (data[data['atom type'] == 'H1']).drop(['atom type'], axis = 1)
    o1s_only = (data[data['atom type'] == 'O1']).drop(['atom type'], axis = 1)

    # note that in the code that follows, the H1-O1 atom pairs that hydrogen bond were determined by looping through all the unique pairs of atoms, and calculating their interatom distances
    # this is very inefficient compared to the parallelised approach used in later codes (example in code_3_abnh3.py)
    # the improved approach used the distance_matrix function from scipy to create a dataframe of all the interatom distances, which was then filtered to the bonding pairs using pandas itself

    # initialise list that stores the number of hydrogen bonds per O1
    o1_h1_inter = []

    # loop over every O1
    for i in zip(o1s_only['x'], o1s_only['y'], o1s_only['z']):
        # store xyz coordinates of an O1
        pos_o1 = np.asarray(i)

        # initialise list for THIS O1, that will contain the number of H1s that it hydrogen bonds to
        o1_h1_list = []

        # for every O1, loop over every H1
        for j in zip(h1s_only['x'], h1s_only['y'], h1s_only['z']):
            # store xyz coordinates of an H1
            pos_h1 = np.asarray(j)

            # calculate O1-H1 interatom distance
            o1_h1_dist = np.linalg.norm(pos_h1 - pos_o1)

            if o1_h1_dist <= 2.7: # if in hydrogen bonding range
                o1_h1_list.append('H-bonds') # record a hydrogen bond

        # store the number of H1s that hydrogen bond to this O1
        o1_h1_inter.append(o1_h1_list.count('H-bonds'))

    # list o1_h1_inter at this point contains all O1s and how many times they H-bond to an H1
    # store this as a pandas series
    o1_h1_inter_series = pd.Series(data = o1_h1_inter)

    # add it to the series storing the probability distribution over each frame/iteration of the simulation
    o1_h1_series = o1_h1_series.add(o1_h1_inter_series.value_counts(), fill_value = 0)

    # progress tracker
    it_count += 1
    if it_count % 200 == 0:
        print(it_count)

# convert absolute numbers to a probability
o1_h1_series = o1_h1_series / o1_h1_series.sum()

# print results as a raw decimal, and as a bar graph
print('Probability distribution of the average number of H1s in hydrogen bonding distance of each O1 is:')
print(o1_h1_series, 'H1 around O1 distribution')
print(o1_h1_series.plot.bar(title = 'H1 around O1 distribution'))
