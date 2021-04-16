import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import pandas as pd
from inference.walker_inference import BiasedPersistentInferer, prepare_paths



# Converts the csv file into a panda dataframe and updates to include the distance from the
# wound edge for spatial slicing

## C and E files are internal controls
"""
Functions for extracting the tracks from the raw csv files outputted by ImageJ.
Function definitions:
---------------------
LoadFile = loads the ImageJ csv file with the raw data sets
dataframe = extracts the data needed for the inference, i.e. x,y,trackID,t
To calculate the polar coordinates of the immune cells with respect to the wound we use the following lambda functions:

wound = (lambda x,y: np.sqrt(x**2 + y**2))(df['x'],df['y'])
angles = (lambda x,y: np.arctan2(x,y))(df['x'],df['y'])

Finally we readjust the angles to be between 0 and 2π, due to the output from the arctan2 giving an angle between
-π and π
"""

def LoadFile(name,x,y):
    file = '/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/ImageJcsvs/Two_wound_mutant.csv'
    file  = pd.read_csv(file,header=0)
    Tracks = dataframe(file,x,y)
    return Tracks

def dataframe(file,x,y):
    x = x
    y = y
    df1 = file
    df = pd.DataFrame({'trackID':df1['TRACK_ID'],'t':df1['POSITION_T'],'x':df1['POSITION_X'],'y':df1['POSITION_Y']})
    df['y'] = 353 - df['y']
    df['x'] = df['x'] - x
    df['y'] = df['y'] - y
    df['r'] = (lambda x,y: np.sqrt(x**2 + y**2))(df['x'],df['y'])
    df['angle'] = (lambda x,y: np.arctan2(x,y))(df['x'],df['y'])
    df['angle'] = df['angle'].apply(lambda x: (2*np.pi) + x if x < 0 else x)
    df['trackID'] = df['trackID'].astype(str)
    return df

# Two wound locations are set and the midpoint calculated

x1,x2= 186,188
xmp =  (x1+x2)/2
y1,y2 = 115,262
ymp = (y1+y2)/2
x_wound_m = [x1]
y_wound_m = [y1]

# Readjusts the y-coordinate system from imageJ to python friendly
y_wound_m[:] = [353 - number for number in y_wound_m ]


#Creates the dataframe for the mutant cell types
df = LoadFile("1",x_wound_m ,y_wound_m)


# The inference mechanism is dependent on the length of the array, if there are not enough tracks
# available in the array it cannot converge
"""
The functions bellow allow for the dataframe to be sliced into its respective, angular, radial and temporal bins
Function definitions:
---------------------
angle_slice = produces a new dataframe in which the tracks are sliced depending on their angular orientation
space_slice = produces a new dataframe in which the tracks are sliced depending on their radial orientation
time_slice = produces a new dataframe in which the tracks are sliced temporally
"""
def angle_slice(df):
    a1 = df[(df['angle'] >= (0)) & (df['angle'] <((5*np.pi)/4)) | (df['angle'] >((7*np.pi)/4))]
    #a1 = df[(df['angle'] >= (np.pi/4)) & (df['angle'] <((3*np.pi)/4))] # Bin for wound 2 (Mutant)
    #a2 = df[(df['angle'] >= ((3*np.pi)/4)) & (df['angle'] < ((5*np.pi)/4))]
    #a3 = df[(df['angle'] >= ((5*np.pi)/4)) & (df['angle'] < ((7*np.pi)/4))]
    a4 = df[(df['angle'] >= ((7*np.pi)/4)) | (df['angle'] < ((np.pi)/4))] # Bin for wound 1 (Wild Type)
    return [a1,a4]#,a3,a4]

def space_slice(theta):
    s25 = theta[(theta['r'] >= 5)  & (theta['r'] <= 45)]
    s50 = theta[(theta['r'] >= 25)  & (theta['r'] <= 75)]
    s75 = theta[(theta['r'] >= 45)  & (theta['r'] <= 105)]
    s100 = theta[(theta['r'] >= 65)  & (theta['r'] <= 135)]
    s125 = theta[(theta['r'] >= 85)  & (theta['r'] <= 165)]
    s150 = theta[(theta['r'] >= 105)  & (theta['r'] <= 195)]
    s175 = theta[(theta['r'] >= 125)  & (theta['r'] <= 225)]
    return [s25,s50,s75,s100,s125,s150,s175]

def time_slice(space):
    t5 = space[(space['t'] >=  0)  & (space['t'] <=600)]
    t15 = space[(space['t'] >=  180)  & (space['t'] <=1620)]
    t30 = space[(space['t'] >= 900)  & (space['t'] <= 2700)]
    #t50 = space[(space['t'] >= 2100)  & (space['t'] <= 3900)]
    times = [t5,t15,t30]#,t50]
    return times



Ag = angle_slice(df)
distance = [space_slice(Ag[i]) for i in range(len(Ag))]

# Here the final dataset for inference to be run on is produced with the data split angularly,radially, and temporally

Final_DataSet =[]

for i in range(len(Ag)):
   innerlist =[]
   for j in range(7):
       innerlist.append(time_slice(distance[i][j]))
   Final_DataSet.append(innerlist)
# This will run the inference method iteratively for each angular, radial and temporal bin. Then save
# them as a numpy array for analysis in the data analysis Python script
from in_silico.sources import PointSource
source = PointSource(position=np.array([0, 0]))
# Variables needed for Ensemble Monte Carlo
niter = 1300
nwalkers = 70

# Variables needed for Metroplis-Hastings Monte Carlo

t = 0
times = 4
for i in range(2):
    for j in range(7):
        for k in range(3):
            t += 1 # Tracks the number of bins
            print('analysing bin {}/{}'.format(t,2*7*3))# to give an overall sense of progress
            inferer = BiasedPersistentInferer(prepare_paths([paths[['x', 'y']].values for id, paths in Final_DataSet[i][j][k].groupby('trackID')],include_t=False),source)
            inf_out = inferer.Ensembleinfer(nwalkers,niter)
            np.save('/Users/danieltudor/Documents/Wood group/ImmuneCellMigrationAnalysis/data/WalkerData/PosterData/AngleTwoWoundMutantloc1{}{}{}_3'.format(i,j,k),inf_out)
