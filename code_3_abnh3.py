# Example code for partial analysis of simulation box of ammonia borane (a.k.a. AB, NH3BH3) dissolved in liquid ammonia (NH3)
# See https://en.wikipedia.org/wiki/Ammonia_borane and https://en.wikipedia.org/wiki/Ammonia
# Intermolecular bonding between AB and ammonia often takes the form of hydrogen bonding between the nitrogen bound hydrogen atoms (H1) on AB and the nitrogen (N2) on ammonia
# In this solution, dihydrogen bonding between the boron bound hydrogen atoms (H2) on AB and the hydrogens on ammonia (H3) is also very common
# Due to the small size of the ammonia molecules, a possible bonding pattern is for a single ammonia molecule to (di)hydrogen bond to both H1 and H2s on the same AB molecule
# This particular code shows the methodology used to determine the probability of this bonding pattern

######################################

# import relevant libraries
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# initialise list for storing probability over each iteration
both_prob = []

# read in the simulation file chunk-by-chunk, as typical files are far too big to read into memory all at once
chunksize = 3834 # represents the 3832 atoms in each frame of the simulation, as well as the two extra rows to skip in each for formatting reasons
iteration_count = 0 # counts the number of simulation frames (or iterations) as the code progresses through the entire simulation file, for tracking purposes
url = 'https://raw.githubusercontent.com/Dingram23/PhD-Codes/master/example_abnh3_3_frames.xyz'

for chunk in pd.read_csv(url, skiprows = 2, names = ['first', 'x', 'y', 'z', 'atom type'], delim_whitespace = True, chunksize = chunksize):

    data = chunk.drop('first', axis = 1) # drops unnecessary first column
    data = data.dropna(axis = 0) # drops extra empty rows after each chunk of atom coordinates

    # convert coordinate columns to dtype float
    data.x = data.x.astype(float)
    data.y = data.y.astype(float)
    data.z = data.z.astype(float)

    # create dataframes with only x,y,z coordinates of each atom type of interest
    h1s_only = (data[data['atom type'] == 'H1']).drop(['atom type'], axis = 1)
    n2s_only = (data[data['atom type'] == 'N2']).drop(['atom type'], axis = 1)
    h3s_only = (data[data['atom type'] == 'H3']).drop(['atom type'], axis = 1)
    h2s_only = (data[data['atom type'] == 'H2']).drop(['atom type'], axis = 1)

    # improved method over code_1 to extract hydrogen bonding atom pairs, by using scipy's distance_matrix function to calculate all the unique interatom pair distances in parallel
    h1_n2_dist_df = pd.DataFrame(data = distance_matrix(h1s_only, n2s_only))
    does_h1_n2_bond = h1_n2_dist_df[h1_n2_dist_df <= 2.7] # use pandas to filter pairs by those that hydrogen bond
    h1_n2_bond_list = list(does_h1_n2_bond[does_h1_n2_bond.notnull()].stack().index) # extract bonding pairs as list of two-element tuples - [(H1a,N2a), (H1a,N2b)] etc.

    # repeat for H2-H3 bonds
    h2_h3_dist_df = pd.DataFrame(data = distance_matrix(h2s_only, h3s_only))
    does_h2_h3_bond = h2_h3_dist_df[h2_h3_dist_df <= 2.5]
    h2_h3_bond_list = list(does_h2_h3_bond[does_h2_h3_bond.notnull()].stack().index)

    # convert H1s to which AB they are on
    ab_for_h1_list = [list(x) for x in h1_n2_bond_list]

    for i in range(0, len(ab_for_h1_list)):
        ab_for_h1_list[i][0] = ab_for_h1_list[i][0] // 3
        # no need to convert N2 to which ammonia, as there is only one N2 per ammonia (so N2 index IS ammonia index)

    # likewise convert H2s to which AB, and H3s to which ammonia
    ab_for_h2_list = [list(x) for x in h2_h3_bond_list]

    for i in range(0, len(ab_for_h2_list)):
        ab_for_h2_list[i][0] = ab_for_h2_list[i][0] // 3
        ab_for_h2_list[i][1] = ab_for_h2_list[i][1] // 3

    # reconvert to list of tuples to prevent accidental changes, and enable set() for removal of duplicates
    ab_for_h1_list = [tuple(x) for x in ab_for_h1_list]
    ab_for_h2_list = [tuple(x) for x in ab_for_h2_list]

    # concatenate the two lists, and count number of unique ABs - the number of ABs that do (di)hydrogen bond in at least one way to an ammonia molecule
    combined_ab_nh3_list = ab_for_h1_list + ab_for_h2_list
    comb_abs, comb_nh3s = zip(*combined_ab_nh3_list)
    no_abs = len(set(comb_abs))

    # then find number of unique ABs that appear in BOTH h1 and h2 lists, so which ABs are bonded to by a single ammonia via both H1-N2 and H2-H3 bonds
    both_h1_h2_same_nh3_list = [x for x in ab_for_h1_list if x in ab_for_h2_list]
    both_h1_h2_same_nh3_count = len(both_h1_h2_same_nh3_list)

    # add result for this frame/iteration to the overall list
    both_prob.append(both_h1_h2_same_nh3_count/no_abs)

    # progress tracker
    iteration_count += 1
    if iteration_count % 3000 == 0:
        print(iteration_count)

# convert final result to an average probability, and print
print("Chance of a single NH3 molecule bonding to both ends of a single AB, out of the number of ABs that bond to at least one NH3 is", round(100 * np.mean(both_prob), 1), "%")
