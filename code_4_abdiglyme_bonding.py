# Example code for analysing simulation box of ammonia borane (a.k.a. AB, NH3BH3) dissolved in liquid diglyme (C6H14O3)
# See https://en.wikipedia.org/wiki/Ammonia_borane and https://en.wikipedia.org/wiki/Diglyme
# Intermolecular bonding between the two types of molecules happens predominantly via hydrogen bonds between the nitrogen bound hydrogen atoms (H1) on AB, and the oxygen atoms on each diglyme (two O1s, and one O2)
# This particular code shows the methodology used to investigate the relative probabilities of the various possible ways in which two AB molecules can bond to one diglyme molecule

# In short, the code extracts every diglyme molecule in each frame of the simulation involved in hydrogen bonding with two different AB molecules
# It then determines how the bonding interactions occur (eg. one AB via a H1-O1 bond, and one via a H1-O2 bond), calculating the relative probabilities of each of them at the end

################################

# import relevant libraries
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from collections import Counter
from operator import itemgetter

# initialise counters for the many possible bonding patterns:
ABs_do_dih_bond = 0 # if two ABs bond to the same diglyme, do the ABs also dihydrogen bond (that is, bond to each other via a H1-H2 bond)?
ABs_dont_dih_bond = 0 # as above, but if they don't dih. bond

both_abs_both_o1s = 0 # test for if both ABs somehow bond via both O1s on the same diglyme
not_both_abs_both_o1s = 0

o2twoo1_o2oneo1 = 0 # if one AB bonds via O2 and two O1s, and the other via O2 and one O1
twoo1_o2oneo1 = 0 # if one AB bonds via two O1s, and the other via O2 and one O1
o2twoo1_oneo1 = 0 # if one AB bonds via O2 and two O1s, and the other only one O1
twoo1_oneo1 = 0 # if one AB bonds via two O1s, and the other via only one O1
o2twoo1_o2 = 0 # if one AB bonds via O2 and two O1s, and the other via only O2
twoo1_o2 = 0 # if one AB bonds via two O1s, and the other via only O2
other_strange = 0 # if one ab bonds via both O1s, and the other does something weird (used as an error check)

o2oneo1_o2oneo1 = 0 # if both abs bond via O2 and one O1
o2oneo1_oneo1 = 0 # if one ab bonds via O2 and one O1, and the other by one O1
oneo1_oneo1 = 0 # if both abs only bond via one O1
o2oneo1_o2 = 0 # if one AB bonds via O2 and one O1, and the other by O2
oneo1_o2 = 0 # if one AB bonds via one O1, and the other by O2
o2_o2 = 0  # if both ABs bond via O2

########################

# read in the simulation file chunk-by-chunk, as typical files are far too big to read into memory all at once
chunksize = 37018 # represents the 37016 atoms in each frame of the simulation, as well as the two extra rows to skip in each for formatting reasons
iteration_count = 0 # counts the number of simulation frames (or iterations) as the code progresses through the entire simulation file, for tracking purposes
url = 'https://raw.githubusercontent.com/Dingram23/PhD-Codes/master/example_abdiglyme_4_frames.xyz'

for chunk in pd.read_csv(url, skiprows = 2, delim_whitespace = True, names = ['first', 'x', 'y', 'z', 'atom type'], chunksize = chunksize): # test simulation file with only four iterations

    data = chunk.drop('first', axis = 1) # drops unnecessary first column
    data = data.dropna(axis = 0) # drops extra empty rows after each chunk of atom coordinates

    # convert coordinates to float
    data.x = data.x.astype(float)
    data.y = data.y.astype(float)
    data.z = data.z.astype(float)

    # create dataframes with only x,y,z coordinates of each atom type of interest
    h1s_only = (data[data['atom type'] == 'H1']).drop(['atom type'], axis = 1)
    o1s_only = (data[data['atom type'] == 'O1']).drop(['atom type'], axis = 1)
    o2s_only = (data[data['atom type'] == 'O2']).drop(['atom type'], axis = 1)
    h2s_only = (data[data['atom type'] == 'H2']).drop(['atom type'], axis = 1)

    ##########################

    # next step is to record which diglyme molecules bond to AB molecules, storing the indexes of each
    # start by finding all O2s that hydrogen bond to H1
    h1_o2_dist_df = pd.DataFrame(data = distance_matrix(h1s_only, o2s_only)) # extract distances between all H1s and all O2s
    does_h1_o2_bond = h1_o2_dist_df[h1_o2_dist_df <= 2.8] # find which ones are in H-bonding range (specific range known from before coding)
    h1_o2_bond_list = list(does_h1_o2_bond[does_h1_o2_bond.notnull()].stack().index) # extract the atom pairs that do bond as a list of tuples, ordered (h1, o2)

    # repeat for all O1s that bond to H1
    h1_o1_dist_df = pd.DataFrame(data = distance_matrix(h1s_only, o1s_only))
    h1_o1_bonds = h1_o1_dist_df[h1_o1_dist_df <= 3] # H1-O1 hydrogen bond is slightly longer than H1-O2
    h1_o1_bond_list = list(h1_o1_bonds[h1_o1_bonds.notnull()].stack().index)

    # extract list of which ABs bond to which diglymes via H1-O2 bonds
    ab_o2_bond_list = [list(x) for x in h1_o2_bond_list] # convert from list of immutable tuples to list of mutable lists

    for i in range(0, len(ab_o2_bond_list)):
        ab_o2_bond_list[i][0] = ab_o2_bond_list[i][0] // 3 # converts from which H1 to which AB (as each AB has three H1s)
        # no need to convert which O2 to which diglyme, as each diglyme only has one O2

    # extract list of which ABs bond to which diglymes via H1-O1 bonds
    ab_dig_bond_list = [list(x) for x in h1_o1_bond_list] # do same for H1-O1

    for i in range(0, len(ab_dig_bond_list)):
        ab_dig_bond_list[i][0] = ab_dig_bond_list[i][0] // 3 # converts from which H1 to which AB
        ab_dig_bond_list[i][1] = ab_dig_bond_list[i][1] // 2 # converts from which O1 to which diglyme (as each diglyme has two O1s)

    # extract list of which ABs bond to which O1 (used later)
    ab_o1_bond_list = [list(x) for x in h1_o1_bond_list]

    for i in range(0, len(ab_o1_bond_list)):
        ab_o1_bond_list[i][0] = ab_o1_bond_list[i][0] // 3

    # also test for which AB molecules dihydrogen bond together (H1-H2 bond)
    h1_h2_dist_df = pd.DataFrame(data = distance_matrix(h1s_only, h2s_only))
    does_h1_h2_bond = h1_h2_dist_df[h1_h2_dist_df <= 3.1]
    h1_h2_bond_list = list(does_h1_h2_bond[does_h1_h2_bond.notnull()].stack().index)

    # extract list of which ABs bond to which (other) ABs via H1-H2 bonds
    ab_ab_bond_list = [list(x) for x in h1_h2_bond_list]

    for i in range(0, len(ab_ab_bond_list)):
        ab_ab_bond_list[i][0] = ab_ab_bond_list[i][0] // 3
        ab_ab_bond_list[i][1] = ab_ab_bond_list[i][1] // 3

    # convert back to lists of tuples to prevent accidental changes
    ab_o1_bond_list = [tuple(x) for x in ab_o1_bond_list] # which O1 bonds to which AB
    ab_dig_bond_list = [tuple(x) for x in ab_dig_bond_list] # which diglyme bonds via O1 to which AB
    ab_o2_bond_list = [tuple(x) for x in ab_o2_bond_list] # which diglyme bonds via O2 to which AB
    ab_ab_bond_list_tuples = [tuple(x) for x in ab_ab_bond_list] # which AB bonds to which (other) AB via H1-H2 bonds

    ######################################
    # next step is to filter the lists to only contain the diglyme molecules that bond to TWO different AB molecules (each AB itself bonding to only one diglyme)
    # remove duplicates from each list, caused by (eg.) O1/O2 bonding to more than one H1 on each AB ('chelation' or 'bifurcated' bonding - considered unimportant to this test)
    unique_ab_o1s = list(set(ab_o1_bond_list))
    unique_ab_o2s = list(set(ab_o2_bond_list))
    unique_ab_digs = list(set(ab_dig_bond_list))
    unique_ab_abs = list(set(ab_ab_bond_list_tuples))

    # filter out instances where ABs bond to more than one diglyme
    ab_dig_comb = unique_ab_digs + unique_ab_o2s # make list of ALL AB-diglyme pairs, regardless of how they bond
    unique_ab_dig_comb = list(set(ab_dig_comb)) # remove duplicates caused by bonding to same diglyme via O1 AND O2
    comb_abs, comb_digs = zip(*unique_ab_dig_comb) # unzip to list of ABs, and list of diglymes - still in order
    count_comb_abs = Counter(comb_abs) # count number of times each AB is listed
    only_one_ab_dig = [x for x in unique_ab_dig_comb if count_comb_abs[x[0]] == 1] # get pairs where the AB only bonds to one diglyme

    # next, find the pairs where two ABs bond to the same diglyme
    abs_a, digs_a = zip(*only_one_ab_dig) # unzip to list of ABs, and list of diglymes
    dig_count = Counter(digs_a) # count number of times diglymes are bound to, separated by unique diglymes
    two_ab_one_dig = [x for x in only_one_ab_dig if dig_count[x[1]] == 2] # get pairs where two ABs bond to same diglyme
    two_ab_one_dig = sorted(two_ab_one_dig, key = itemgetter(1)) # sort by diglyme

    ########################################
    # next step is to analyse and quantify the bonding patterns between the diglyme molecules and both ABs in question

    # check if the ABs that bond to same diglyme, themselves dihydrogen bond (H1-H2)
    for i in range(0, len(two_ab_one_dig), 2): # iterate over all pairs, by diglyme
        ab_a = two_ab_one_dig[i][0] # record both ABs that bond to the diglyme
        ab_b = two_ab_one_dig[i+1][0]

        if ((ab_a, ab_b) in unique_ab_abs) or ((ab_b, ab_a) in unique_ab_abs): # if the two ABs are in the dihydrogen bonding list
            ABs_do_dih_bond += 1
        else:
            ABs_dont_dih_bond += 1

    # next, extract pairs where O1s on same diglyme bond to same AB

    # find diglymes where both O1s are involved in bonding - that is, overlap between two_ab_one_dig and ab_o1_bond_list
    both_o1s_same_ab = []

    for i in range(0, len(two_ab_one_dig), 2): # iterate over diglymes
        ab_one = two_ab_one_dig[i][0]
        ab_two = two_ab_one_dig[i+1][0]
        diglyme = two_ab_one_dig[i][1]
        o1_a = diglyme * 2 # convert diglyme into the potential O1s
        o1_b = diglyme * 2 + 1

        # now check for presence of both of these O1s in ab_o1_bond_list
        # could unzip ab_o1_bond_list and check that, but diglyme could bond to third AB etc., best to include AB here

        if ((ab_one, o1_a) in ab_o1_bond_list) and ((ab_one, o1_b) in ab_o1_bond_list): # if ab_one bonds to both O1s
            both_o1s_same_ab.append((ab_one, diglyme))

        elif ((ab_two, o1_a) in ab_o1_bond_list) and ((ab_two, o1_b) in ab_o1_bond_list): # if ab_two bonds to both O1s
            both_o1s_same_ab.append((ab_two, diglyme))

    # so both_o1s_same_ab is a list of all AB_dig pairs, in two_ab_one_dig, where both diglyme O1s bond to the AB

    both_abs, both_digs = zip(*both_o1s_same_ab)
    both_o1s_same_ab_and_other = [x for x in two_ab_one_dig if x[1] in both_digs] # combine that pair with the other AB that bonds to that diglyme

    # brief check that we don't get situations where both ABs somehow bond to the same diglyme via both O1s
    # can't think of a spatial orientation that would allow that, but just in case!

    both_digs_count = Counter(both_digs)
    both_digs_test = [x for x in both_digs if both_digs_count[x] > 1]

    if len(both_digs_test) != 0:
        both_abs_both_o1s += 1

    else:
        not_both_abs_both_o1s += 1

    #########

    # now, want to test both_o1s_same_ab_and_other for how the other AB bonds to diglyme, via O1 or O2?

    other_ab_from_both_o1 = [x for x in both_o1s_same_ab_and_other if x not in both_o1s_same_ab] # extract AB-diglyme pair where the AB doesn't bond to both O1s

    for i in range(0, len(other_ab_from_both_o1)):
        ab = other_ab_from_both_o1[i][0]
        diglyme = other_ab_from_both_o1[i][1]

        if ((ab, diglyme) in ab_dig_bond_list) and ((ab, diglyme) in ab_o2_bond_list): # via O1 and O2

            if both_o1s_same_ab[i] in ab_o2_bond_list: # if the AB bonding to two O1s also bonds to O2
                o2twoo1_o2oneo1 += 1

            else: # if the AB bonding to two O1s does not bond to O2 as well
                twoo1_o2oneo1 += 1

        elif ((ab, diglyme) in ab_dig_bond_list) and ((ab, diglyme) not in ab_o2_bond_list): # via just O1

            if both_o1s_same_ab[i] in ab_o2_bond_list: # if the AB bonding to two O1s also bonds to O2
                o2twoo1_oneo1 += 1

            else: # if the AB bonding to two O1s does not bond to O2 as well
                twoo1_oneo1 += 1

        elif ((ab, diglyme) not in ab_dig_bond_list) and ((ab, diglyme) in ab_o2_bond_list): # via just O2

            if both_o1s_same_ab[i] in ab_o2_bond_list: # if the AB bonding to two O1s also bonds to O2
                o2twoo1_o2 += 1

            else: # if the AB bonding to two O1s does not bond to O2 as well
                twoo1_o2 += 1

        else:
            other_strange += 1 # error checking, code should not run this

    #########

    # next, need to analyse bonding where an AB does not bond to diglyme via both O1s - so only one O1 bond is possible per AB

    not_both_o1s = [x for x in two_ab_one_dig if x not in both_o1s_same_ab_and_other] # extract pairs with no double O1 bonding to same AB

    for i in range(0, len(not_both_o1s), 2): # loop over all pairs, by diglyme
        diglyme = not_both_o1s[i][1]
        ab_one = not_both_o1s[i][0]
        ab_two = not_both_o1s[i+1][0]

        # first check for bonding to O1, then specifics on which AB does so, and whether it also bonds to O2

        if ((ab_one, diglyme) in ab_dig_bond_list) and ((ab_two, diglyme) in ab_dig_bond_list): # if both ABs bond via O1

            if ((ab_one, diglyme) in ab_o2_bond_list) and ((ab_two, diglyme) in ab_o2_bond_list): # and if both bond via O2
                o2oneo1_o2oneo1 += 1

            elif ((ab_one, diglyme) in ab_o2_bond_list) or ((ab_two, diglyme) in ab_o2_bond_list): #  and if one via O2
                o2oneo1_oneo1 += 1

            else: # and if neither bonds via O2
                oneo1_oneo1 += 1

        elif ((ab_one, diglyme) in ab_dig_bond_list) or ((ab_two, diglyme) in ab_dig_bond_list): # if only ONE AB bonds via O1

            if ((ab_one, diglyme) in ab_dig_bond_list) and ((ab_one, diglyme) in ab_o2_bond_list):
                # if it was ab_one via O1, and it also bonds via O2
                # ab_two must be via O2 as it must bond somehow!
                o2oneo1_o2 += 1

            elif ((ab_one, diglyme) in ab_dig_bond_list) and ((ab_one, diglyme) not in ab_o2_bond_list):
                # likewise, if ab_one doesn't bond via O2
                oneo1_o2 += 1

            elif ((ab_two, diglyme) in ab_dig_bond_list) and ((ab_two, diglyme) in ab_o2_bond_list):
                # same but with ab_two
                # would be o2_o2oneo1 but it's the equivalent of o2oneo1_o2
                o2oneo1_o2 += 1

            else:
                # only remaining combination is ab_two via O1, ab_one via O2
                oneo1_o2 += 1

        else: # if neither bond via O1, they must both bond via O2 (however spatially unlikely it may seem)
            o2_o2 += 1

    #########

    # progress check

    iteration_count += 1
    if iteration_count % 1000 == 0:
        print(iteration_count)

    #########

# calculate overall totals for relative probability calculations

both_o1_total = o2twoo1_o2oneo1 + twoo1_o2oneo1 + o2twoo1_oneo1 + twoo1_oneo1 + o2twoo1_o2 + twoo1_o2 + other_strange

not_both_o1_totals = o2oneo1_o2oneo1 + o2oneo1_oneo1 + oneo1_oneo1 + o2oneo1_o2 + oneo1_o2 + o2_o2

overall_total = both_o1_total + not_both_o1_totals

# print results

print("If two ABs bond only to the same diglyme:")
print("")

# both ABs should (spatially) not be able to bond to both O1s on the same diglyme, but if they do, print the probability
if both_abs_both_o1s != 0:
    print("Both ABs somehow bonded to both O1s on the same diglyme in this many iterations:", both_abs_both_o1s)
    print("This is compared to", not_both_abs_both_o1s, "iterations where it did not")
    print(round(100 * both_abs_both_o1s/not_both_abs_both_o1s, 1), "%")

else:
    print("No situations were observed where both ABs somehow bonded via both O1s to the same diglyme")

print("")
print("Chance of ABs that bond to the same diglyme themselves dih. bonding is", round(100 * ABs_do_dih_bond/ABs_dont_dih_bond, 1), "%")
print("")
print("Chance of one AB bonding via both O1s and O2, and the other via O2 and one O1 is", round(100 * o2twoo1_o2oneo1/overall_total, 1), "%")
print("Chance of one AB bonding via both O1s, and the other via O2 and one O1 is", round(100 * twoo1_o2oneo1/overall_total, 1), "%")
print("Chance of one AB bonding via both O1s and O2, and the other via one O1 is", round(100 * o2twoo1_oneo1/overall_total, 1), "%")
print("Chance of one AB bonding via both O1s, and the other via one O1 is", round(100 * twoo1_oneo1/overall_total, 1), "%")
print("Chance of one AB bonding via both O1s and O2, and the other via O2 is", round(100 * o2twoo1_o2/overall_total, 1), "%")
print("Chance of one AB bonding via both O1s, and the other via O2 is", round(100 * twoo1_o2/overall_total, 1), "%")

# if an unforseen error occured in the bonding pattern logic, report it
if other_strange != 0:
    print("If there was a strange error, here is the chance it happened:", round(100 * other_strange / overall_total, 1), "%")

print("")
print("Chance of both ABs bonding via O2 and one O1 is", round(100 * o2oneo1_o2oneo1/overall_total, 1), "%")
print("Chance of one AB bonding via O2 and one O1, and the other via one O1 is", round(100 * o2oneo1_oneo1/overall_total, 1), "%")
print("Chance of both ABs bonding via one O1 is", round(100 * oneo1_oneo1/overall_total, 1), "%")
print("Chance of one AB bonding via O2 and one O1, and the other via O2 is", round(100 * o2oneo1_o2/overall_total, 1), "%")
print("Chance of one AB bonding via O2, and the other via one O1 is", round(100 * oneo1_o2/overall_total, 1), "%")
print("Chance of both ABs bonding via O2 is", round(100 * o2_o2/overall_total, 1), "%")
