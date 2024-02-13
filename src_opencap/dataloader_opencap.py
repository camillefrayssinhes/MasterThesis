import numpy as np
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
import natsort
import math
import pathlib
import os


def download_parameters(subject):
    """
    Download the gait parameters, the trajectories of all the markers, the joint angles and the forces of the selected trial of the selected subject.

    Inputs:
        * subject (string): name of the subject, e.g. 'BO2ST_101'
        
    Outputs:
        * trajectories (list):
        * angles (list):
        * forces (list): 
    """
        
    path = pathlib.Path("BO2STTrial/OpenCap/"+subject+"/Vicon")
    list_ = list(path.glob(subject+" Trial*"))
    print(list_)

    # marker trajectories
    trajectories = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 0, header = [1,2,3], skiprows = 1)
        trajectories.append(tuple((file_name, exc_trial)))  

    # joint angles
    angles = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 1, header = [1,2,3], skiprows = 1)
        angles.append(tuple((file_name, exc_trial)))   

    # forces    
    forces = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 2, header = [1,2,3], skiprows = 1)
        forces.append(tuple((file_name, exc_trial)))
              
    return trajectories, angles, forces 

