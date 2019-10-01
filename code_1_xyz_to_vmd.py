# After every 'block' of atom coordinates in each EPSR iteration, there is an extra line used in converting the .xyz file to .HISu format (for dlputils analysis)
# This extra line prevents the .xyz file from being correctly read into VMD
# This simple code removes that extra line after the end of the block of atom coordinates, for each EPSR iteration

import pandas as pd

chunksize = 3834 # number of individual atomic coordinates per EPSR iteration, as well as two extra rows
it_count = 0
url = 'https://raw.githubusercontent.com/Dingram23/PhD-Codes/master/example_abnh3_3_frames.xyz'

# read chunk-by-chunk, as typical simulation files are far too big to be read into memory all at once
for chunk in pd.read_csv(url, names = ['first', 'x', 'y', 'z', 'atom type'], delim_whitespace = True, chunksize = chunksize): # using test simulation file with only 3 frames

    data = chunk

    if it_count == 0:
        data.to_csv('line_removed_abnh3.xyz', sep='\t', header=False, index=False, float_format='%.3f') # creates new .xyz file
        it_count += 1
    else:
        data.to_csv('line_removed_abnh3.xyz', sep='\t', header=False, index=False, float_format='%.3f', mode='a') # appends further iterations to same .xyz file
