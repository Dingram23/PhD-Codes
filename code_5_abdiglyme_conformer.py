# Example code for determining the relative probabilities of each conformer of diglyme (C6H14O3) present in a simulated amomnia borane (a.k.a. AB, NH3BH3) solution
# See https://en.wikipedia.org/wiki/Ammonia_borane and https://en.wikipedia.org/wiki/Diglyme
# Conformers are determined by combining the six dihedral angles present in each diglyme
# See https://en.wikipedia.org/wiki/Conformational_isomerism
# Using theory for calculating dihedral angles from https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates

# In short, this code extracts the specific diglyme molecules in each frame/iteration of the simulation that hydrogen bond to AB molecules via the Oc (central oxygen atom) and at least one Oe (terminal oxygen atom)
# It then calculates the six dihedral angles present in each of these diglyme molecules
# The angles then get binned according to their conformation
# Finally, the relative probabilities of each overall diglyme conformer are determined and printed

###############################################

# import relevant libraries
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

# read in the simulation file chunk-by-chunk, as typical files are far too big to read into memory all at once
it_count = 0 # counts the number of simulation frames (or iterations) as the code progresses through the entire simulation file, for tracking purposes
chunksize = 37018 # includes the extra 2 rows to skip, there are only 37016 atoms per iteration
url = 'https://raw.githubusercontent.com/Dingram23/PhD-Codes/master/example_abdiglyme_4_frames.xyz'

for chunk in pd.read_csv(url, skiprows = 2, delim_whitespace = True, names = ['first', 'x', 'y', 'z', 'atom type'], chunksize = chunksize):

    data = chunk.drop('first', axis = 1) # drops unnecessary first column
    data = data.dropna(axis = 0) # drops extra empty rows after each chunk of atom coordinates

    # convert coordinate columns to dtype float
    data.x = data.x.astype(float)
    data.y = data.y.astype(float)
    data.z = data.z.astype(float)

    # create dataframes with only x,y,z coordinates of each atom type of interest
    h1s_only = (data[data['atom type'] == 'H1']).drop(['atom type'], axis = 1) # AB nitrogen-bound hydrogen
    oes_only = (data[data['atom type'] == 'O1']).drop(['atom type'], axis = 1) # O1 == Oe
    ocs_only = (data[data['atom type'] == 'O2']).drop(['atom type'], axis = 1) # O2 == Oc
    c5s_only = (data[data['atom type'] == 'C1']).drop(['atom type'], axis = 1) # C1 in simulation is C5 in nomenclature (unimportant for example purposes - just know that C3, C4, and C5 are the diglyme carbons)
    c4s_only = (data[data['atom type'] == 'C2']).drop(['atom type'], axis = 1) # C2 in simulation is C4 in nomenclature
    c3s_only = (data[data['atom type'] == 'C3']).drop(['atom type'], axis = 1)

    ###################################################################################################

    # next step is to record which diglyme molecules bond to AB molecules, storing the indexes of each
    # start by finding all O2s that hydrogen bond to H1
    h1_oc_dist_df = pd.DataFrame(data = distance_matrix(h1s_only, ocs_only))
    does_h1_oc_bond = h1_oc_dist_df[h1_oc_dist_df <= 2.8]
    h1_oc_bond_list = list(does_h1_oc_bond[does_h1_oc_bond.notnull()].stack().index)

    # repeat for all O1s that bond to H1
    h1_oe_dist_df = pd.DataFrame(data = distance_matrix(h1s_only, oes_only))
    h1_oe_bonds = h1_oe_dist_df[h1_oe_dist_df <= 3]
    h1_oe_bond_list = list(h1_oe_bonds[h1_oe_bonds.notnull()].stack().index)

    # extract list of which ABs bond to which diglymes via H1-O2 bonds
    ab_oc_bond_list = [list(x) for x in h1_oc_bond_list] # convert from list of immutable tuples to list of mutable lists

    for i in range(0, len(ab_oc_bond_list)):
        ab_oc_bond_list[i][0] = ab_oc_bond_list[i][0] // 3 # converts from which H1 to which AB
        # no need to convert from which O2 to which diglyme, as there is only one O2 per diglyme

    # do same for H1-O1
    ab_dig_bond_list = [list(x) for x in h1_oe_bond_list]

    for i in range(0, len(ab_dig_bond_list)):
        ab_dig_bond_list[i][0] = ab_dig_bond_list[i][0] // 3 # converts from which H1 to which AB
        ab_dig_bond_list[i][1] = ab_dig_bond_list[i][1] // 2 # converts from which O1 to which diglyme

    # convert back to lists of tuples to prevent accidental changes
    ab_oc_bond_list = [tuple(x) for x in ab_oc_bond_list]
    ab_dig_bond_list = [tuple(x) for x in ab_dig_bond_list]

    # remove duplicates caused by bifurcated bonds (where two hydrogens on the same AB hydrogen bond to the same oxygen)
    ab_oc_bond_list = list(set(ab_oc_bond_list))
    ab_dig_bond_list = list(set(ab_dig_bond_list))

    # extract only diglymes
    abs_oc, digs_oc = zip(*ab_oc_bond_list)
    abs_oe, digs_oe = zip(*ab_dig_bond_list)

    # remove duplicate diglymes caused by multiple ABs bonding to same oxygen (or other Oe)
    digs_oc = tuple(set(digs_oc))
    digs_oe = tuple(set(digs_oe))

    # extract list of diglymes that hydrogen bond to ABs via both Oc and at least one Oe
    digs_both = [x for x in digs_oc if x in digs_oe]

    ######################################################################################################

    # second, determine the dihedral angles
    # get scipy distance matrices for each relevant bond length (needed even though distances are intramolecular, as the simulation varies them slightly)
    c5_oe_dist_df = pd.DataFrame(data = distance_matrix(c5s_only, oes_only))
    oe_c4_dist_df = pd.DataFrame(data = distance_matrix(oes_only, c4s_only))
    c4_c3_dist_df = pd.DataFrame(data = distance_matrix(c4s_only, c3s_only))
    c3_oc_dist_df = pd.DataFrame(data = distance_matrix(c3s_only, ocs_only))

    # initialise list to contain the dihedral angles for all the diglymes in this iteration/frame of the simulation
    all_dih_both = []

    # loop over all extracted diglymes
    for i in digs_both:
        # first, find each relevant atom index in this diglyme
        # index if one atom per diglyme (Oc)
        opd = i

        # indices if two atoms per diglyme (Oe)
        tpd_1 = i*2
        tpd_2 = i*2+1

        # so 1 and 2 are the two sides/ends of the diglyme (which are mirrored around Oc in terms of atom type)
        # important to keep consistent so we aren't calling distance between one side of the diglyme, and the other

        # extract oxygen and carbon atom coordinates
        oc = tuple(ocs_only.iloc[opd])
        oe_1 = tuple(oes_only.iloc[tpd_1])
        oe_2 = tuple(oes_only.iloc[tpd_2])
        c5_1 = tuple(c5s_only.iloc[tpd_1])
        c5_2 = tuple(c5s_only.iloc[tpd_2])
        c4_1 = tuple(c4s_only.iloc[tpd_1])
        c4_2 = tuple(c4s_only.iloc[tpd_2])
        c3_1 = tuple(c3s_only.iloc[tpd_1])
        c3_2 = tuple(c3s_only.iloc[tpd_2])

        # next, determine vectors representing bonds through vector subtraction
        # again, using theory for calculating dihedral angles from https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates

        # unzip coordinates, in order by the side/end of diglyme
        oc_x, oc_y, oc_z = oc[0], oc[1], oc[2]
        oe_1_x, oe_1_y, oe_1_z = oe_1[0], oe_1[1], oe_1[2]
        c5_1_x, c5_1_y, c5_1_z = c5_1[0], c5_1[1], c5_1[2]
        c4_1_x, c4_1_y, c4_1_z = c4_1[0], c4_1[1], c4_1[2]
        c3_1_x, c3_1_y, c3_1_z = c3_1[0], c3_1[1], c3_1[2]
        c3_2_x, c3_2_y, c3_2_z = c3_2[0], c3_2[1], c3_2[2]
        c4_2_x, c4_2_y, c4_2_z = c4_2[0], c4_2[1], c4_2[2]
        c5_2_x, c5_2_y, c5_2_z = c5_2[0], c5_2[1], c5_2[2]
        oe_2_x, oe_2_y, oe_2_z = oe_2[0], oe_2[1], oe_2[2]

        # subtract end position from start to get vector representing the bond
        c5_1_to_oe_1 = (oe_1_x-c5_1_x, oe_1_y-c5_1_y, oe_1_z-c5_1_z) # c5 to Oe vec, equals end - start
        oe_1_to_c4_1 = (c4_1_x-oe_1_x, c4_1_y-oe_1_y, c4_1_z-oe_1_z)
        c4_1_to_c3_1 = (c3_1_x-c4_1_x, c3_1_y-c4_1_y, c3_1_z-c4_1_z)
        c3_1_to_oc = (oc_x-c3_1_x, oc_y-c3_1_y, oc_z-c3_1_z)
        oc_to_c3_2 = (c3_2_x-oc_x, c3_2_y-oc_y, c3_2_z-oc_z)
        c3_2_to_c4_2 = (c4_2_x-c3_2_x, c4_2_y-c3_2_y, c4_2_z-c3_2_z)
        c4_2_to_oe_2 = (oe_2_x-c4_2_x, oe_2_y-c4_2_y, oe_2_z-c4_2_z)
        oe_2_to_c5_2 = (c5_2_x-oe_2_x, c5_2_y-oe_2_y, c5_2_z-oe_2_z)

        # calculate the vectors normal to the planes containing b1/b2, and b2/b3, where bx are the atoms in each dihedral angle
        # angle between these two normal vectors is the same as the dihedral
        dih1_n1 = tuple(np.cross(c5_1_to_oe_1, oe_1_to_c4_1))
        dih1_n2 = tuple(np.cross(oe_1_to_c4_1, c4_1_to_c3_1))
        dih2_n1 = dih1_n2
        dih2_n2 = tuple(np.cross(c4_1_to_c3_1, c3_1_to_oc))
        dih3_n1 = dih2_n2
        dih3_n2 = tuple(np.cross(c3_1_to_oc, oc_to_c3_2))
        dih4_n1 = dih3_n2
        dih4_n2 = tuple(np.cross(oc_to_c3_2, c3_2_to_c4_2))
        dih5_n1 = dih4_n2
        dih5_n2 = tuple(np.cross(c3_2_to_c4_2, c4_2_to_oe_2))
        dih6_n1 = dih5_n2
        dih6_n2 = tuple(np.cross(c4_2_to_oe_2, oe_2_to_c5_2))

        # calculate the vector m1, which forms orthonormal frame with n1 and b2
        dih1_m1 = tuple(np.cross(dih1_n1, oe_1_to_c4_1))
        dih2_m1 = tuple(np.cross(dih2_n1, c4_1_to_c3_1))
        dih3_m1 = tuple(np.cross(dih3_n1, c3_1_to_oc))
        dih4_m1 = tuple(np.cross(dih4_n1, oc_to_c3_2))
        dih5_m1 = tuple(np.cross(dih5_n1, c3_2_to_c4_2))
        dih6_m1 = tuple(np.cross(dih6_n1, c4_2_to_oe_2))

        # calculate the coordinates of n2 in that frame
        dih1_x = np.dot(dih1_n1, dih1_n2)
        dih1_y = np.dot(dih1_m1, dih1_n2)
        dih2_x = np.dot(dih2_n1, dih2_n2)
        dih2_y = np.dot(dih2_m1, dih2_n2)
        dih3_x = np.dot(dih3_n1, dih3_n2)
        dih3_y = np.dot(dih3_m1, dih3_n2)
        dih4_x = np.dot(dih4_n1, dih4_n2)
        dih4_y = np.dot(dih4_m1, dih4_n2)
        dih5_x = np.dot(dih5_n1, dih5_n2)
        dih5_y = np.dot(dih5_m1, dih5_n2)
        dih6_x = np.dot(dih6_n1, dih6_n2)
        dih6_y = np.dot(dih6_m1, dih6_n2)

        # finally, calculate the dihedral angles
        dih1_ang = np.degrees(np.arctan2(dih1_y, dih1_x))
        dih2_ang = np.degrees(np.arctan2(dih2_y, dih2_x))
        dih3_ang = np.degrees(np.arctan2(dih3_y, dih3_x))
        dih4_ang = np.degrees(np.arctan2(dih4_y, dih4_x))
        dih5_ang = np.degrees(np.arctan2(dih5_y, dih5_x))
        dih6_ang = np.degrees(np.arctan2(dih6_y, dih6_x))

        # append the dihedral angles for this diglyme to the list, in order (important this order is immutable, to preserve identity of overall diglyme conformer)
        dihedrals = (dih1_ang, dih2_ang, dih3_ang, dih4_ang, dih5_ang, dih6_ang)
        all_dih_both.append(dihedrals)

    ####################################################################################################

    # next, bin the angles into gauche+, gauche-, trans, cis, eclipsed+, eclipsed-
    # -ve sign represents direction of the dihedral angle relative to the one before it, when needed
    # see https://en.wikipedia.org/wiki/Conformational_isomerism for formal definitions of these bins
    # angles run from -180 to +180 degrees, bin by closest +/-30 degrees

    # initialise list to store all binned conformers
    all_dih_both_conf = []

    # loop over all conformers
    for i in all_dih_both:
        # initialise list to store angles as they are binned
        conformer = []

        # loop over all the dihedral angles within a conformer
        for j in range(0, len(i)):
            # bin by closest +/- 30 degrees
            binning = round(i[j] / 60, 0)

            # replace number with formal designation (gauche+, gauche-, trans, cis, eclipsed+, eclipsed-)
            if binning == 3.0 or binning == -3.0:
                binned = "t"
            elif binning == 2.0:
                binned = "e+"
            elif binning == -2.0:
                binned = "e-"
            elif binning == 1.0:
                binned = "g+"
            elif binning == -1.0:
                binned = "g-"
            elif binning == 0.0 or binning == -0.0:
                binned = "c"

            conformer.append(binned)

        # append the final binned conformer (for example (t, c, t, t, e+, t)) to the overall list
        all_dih_both_conf.append(conformer)

    # convert to pandas series and count how many of each conformer were present in this frame/iteration
    all_dih_both_conf_tuples = [tuple(x) for x in all_dih_both_conf]
    all_dih_both_series = pd.Series(data = all_dih_both_conf_tuples)
    dih_both_series_count = all_dih_both_series.value_counts()

    # initialise result across iterations/frames as a pandas series, or add to it if not the first iteration
    if it_count == 0:
        all_dih_both_series_count = dih_both_series_count
    else:
        all_dih_both_series_count = all_dih_both_series_count.add(dih_both_series_count, fill_value = 0)

    # progress tracker
    it_count += 1
    if it_count % 500 == 0:
        print(it_count)

    ###############################################################################################

# finally, determine the relative probabilities of each conformer
all_dih_both_series_prob = all_dih_both_series_count.divide(np.sum(all_dih_both_series_count))
all_dih_both_series_prob = all_dih_both_series_prob.multiply(100).sort_index()
all_dih_both_series_prob = all_dih_both_series_prob.round(1)

# print results
print("Relative probabilities of a diglyme molecule adopting each conformation when ABs bond to Oc and at least one Oe are:")
print(all_dih_both_series_prob)
