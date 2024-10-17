# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-09-30 -*-
# -*- Last revision: 2024-10-17 by Vincent Roduit -*-
# -*- python version : 3.9.19 -*-
# -*- Description: Util functions *-

#import libraries
import os

#import files
from constants import *
import pickle as pkl

def save_data(data: any, file_name: str, folder: str = os.path.join(DATA_FOLDER, "pickles")):
    """Save the data to a file
    
    Args:
        * data (any): the data to save

        * file_name (str): the name of the file

        * folder (str): the folder where to save the file
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, file_name)

    with open(file_path, 'wb') as handle:
        pkl.dump(data, handle)

def load_data(file_name: str, folder: str = os.path.join(DATA_FOLDER, "pickles")) -> any:
    """Load the data from a file

    Args:
        * file_name (str): the name of the file

        * folder (str): the folder where to save the file

    Returns:
        * any: the data
    """
    file_path = os.path.join(folder, file_name)

    with open(file_path, 'rb') as handle:
        data = pkl.load(handle)

    return data