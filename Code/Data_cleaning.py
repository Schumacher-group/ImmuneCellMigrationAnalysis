"""
This takes in the csv files outputted by imageJ and applies all the processing needed to perform the
biased persistence random walk model. Eventually, this file will also allow for any data file type to be processed
but currently we run the imageJ trackMate plugin to extract cell x,y co-ordinates

The current file format from the csv are:

Label|ID|Track_ID|Quality|Position_x|Position_y|Position_z|Position_t|Frame|

For the inference pipeline and model we need to re-format to this:

Track_ID|Position_x|Position_y|Time|Distance from wound (radius)|

The data can then be binned into different spatial and temporal bins, and the model can be run on each of these bins
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import pandas as pd
from datetime import datetime


# Note need to make this generic so csv_file can take either control or mutant as a parameter
def csv_to_dataframe(file_number):
    """
    Converts csv file to panda dataframe
    Parameters
    ----------
    file_number : specific file number

    Returns
    -------
    Panda dataframe
    """
    csv_file = f'../data/ImageJcsvs/Control_{file_number}_new.csv'
    dataframe_from_file = pd.read_csv(csv_file, header=0)
    return dataframe_from_file


def reformat_file(dataframe, xw, yw, loc):
    """
    reformat_file takes in the new dataframe created from the csv and the x-y wound location and reshapes the dataframe
    to clean up the column labels and add a new column called 'r' which holds the radial distance from the wound.
    Parameters
    ----------
    dataframe
    loc
    yw
    xw

    dataframe: dataframe converted from csv
    xw = x location of wound
    yw = y location of wound
    loc = location of file in folder (i.e. file 1 is A, file 2 is B, etc.), allows for multiple files to be concatenated
    Returns
    -------
    reshaped_dataframe: reshaped dataframe which has the correct column labels and radial_location parameters
    """

    reshaped_dataframe = pd.DataFrame(
        {'trackID': dataframe['TRACK_ID'], 't': dataframe['POSITION_T'], 'x': dataframe['POSITION_X'],
         'y': dataframe['POSITION_Y']})
    reshaped_dataframe['x'] = reshaped_dataframe['x'] - xw
    reshaped_dataframe['y'] = reshaped_dataframe['y'] - yw
    reshaped_dataframe['r'] = (lambda x, y: np.sqrt(x ** 2 + y ** 2))(reshaped_dataframe['x'], reshaped_dataframe['y'])
    reshaped_dataframe['trackID'] = reshaped_dataframe['trackID'].astype(str)
    reshaped_dataframe.trackID = reshaped_dataframe.trackID + "{}".format(
        loc)  # creates a label for the tracks to be organised by, only needed if there are multiple files to run on

    return reshaped_dataframe


def wound_locations(control=True):
    """
    wound_location takes in a True or false and returns the correct wound locations depending which wound location are
    needed, e.g. control or mutant
    Parameters
    ----------
    control: True/False

    Returns
    -------
    either:
    x_wound_control,y_wound_control : wound_location for control
    x_wound_mutant,y_wound_mutant : wound_location for mutant
    """
    if control:
        x_wound_control = [199, 175, 170, 184, 170, 163, 184]
        y_wound_control = [(353 - 234), (353 - 150), (353 - 107), (353 - 110), (353 - 238), (353 - 226), (353 - 220)]
        return x_wound_control, y_wound_control
    else:
        x_wound_mutant = [224, 217, 221]
        y_wound_mutant = [226, 121, 174]
        return x_wound_mutant, y_wound_mutant


def concatenated_dataframes(num_files, control=True):
    """
    concatenate_dataframes takes in the number of files which need processing (i.e. 7 for control and 3 for wound), and
    then runs csv_to_dataframe and reform_file to convert this files to a pandas dataframe and produce correct column
    names. The wound_locations function is also used which uses the parameter control to decide if the wound locations
    need to come from the control or mutant dataset. It finally returns a concatenated dataframe which the inference
    pipeline can use.
    Parameters
    ----------
    num_files = number of files needing to be converted and reformatted
    control = control or mutant wounds locations, automatically set to True for control wound locations

    Returns
    -------
    Concatenated dataframe with all files correctly formatted
    """
    wound_xy = wound_locations(control)
    dataframes_list = [csv_to_dataframe(i+1) for i in range(num_files)]
    reformat_dataframes_list = [reformat_file(dataframes_list[i], wound_xy[0][i], wound_xy[1][0], chr(ord('@') + (i + 1))) for i in range(num_files)]
    return reformat_dataframes_list  # pd.concat(dataframes_list)


trajectory = concatenated_dataframes(7)
date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
import pickle
with open(f"../data/Trajectory_dataframes/{date}_control", "wb") as fp:   #Pickling
    pickle.dump(trajectory, fp)


