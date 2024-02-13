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


def download_parameters(subject, assessment):
    """
    Download the gait parameters, the trajectories of all the markers, the joint angles and the forces of the selected trial of the selected subject.

    Inputs:
        * subject (string): name of the subject, e.g. 'BO2ST_101'
        * assessment (string): date and category of assessment, e.g. '20230310_BL'
        
    Outputs:
        * gait_params (list): 
        * trajectories (list):
        * angles (list):
        * forces (list): 
    """
        
    path = pathlib.Path("BO2STTrial/"+subject+"/"+assessment+"/xcl")
    list_ = list(path.glob(subject+" Trial 0*"))

    # gait cycle parameters
    gait_params = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 0, header = [2])
        gait_params.append(tuple((file_name, exc_trial)))

    # marker trajectories
    trajectories = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 1, header = [1,2,3], skiprows = 1)
        trajectories.append(tuple((file_name, exc_trial)))  

    # joint angles
    angles = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 2, header = [1,2,3], skiprows = 1)
        angles.append(tuple((file_name, exc_trial)))   

    # forces    
    forces = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 3, header = [1,2,3], skiprows = 1)
        forces.append(tuple((file_name, exc_trial)))
              
    return gait_params, trajectories, angles, forces 

def download_parameters_new_ss_trials(subject, assessment):
    """
    Download the gait parameters, the trajectories of all the markers, the joint angles and the forces of the selected trial of the selected subject.

    Inputs:
        * subject (string): name of the subject, e.g. 'BO2ST_101'
        * assessment (string): date and category of assessment, e.g. '20230310_BL'
        
    Outputs:
        * gait_params (list): 
        * trajectories (list):
        * angles (list):
        * forces (list): 
    """
        
    path = pathlib.Path("BO2STTrial/"+subject+"/"+assessment+"/xcl")
    list_ = list(path.glob(subject+" New_ss*"))

    # gait cycle parameters
    gait_params = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 0, header = [2])
        gait_params.append(tuple((file_name, exc_trial)))

    # marker trajectories
    trajectories = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 1, header = [1,2,3], skiprows = 1)
        trajectories.append(tuple((file_name, exc_trial)))  

    # joint angles
    angles = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 2, header = [1,2,3], skiprows = 1)
        angles.append(tuple((file_name, exc_trial)))   

    # forces    
    forces = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        exc_trial = pd.read_excel(l, sheet_name = 3, header = [1,2,3], skiprows = 1)
        forces.append(tuple((file_name, exc_trial)))
              
    return gait_params, trajectories, angles, forces 



def download_parameters_force_calibration(subject, assessment):
    """
    Download the gait parameters, the trajectories of all the markers, the joint angles and the forces of the selected trial of the selected subject.
    By default, Trial 10 is the calibration for weight on the left force plate, Trial 11 is the calibration for weight on the right force plate. 

    Inputs:
        * subject (string): name of the subject, e.g. 'BO2ST_101'
        * assessment (string): date and category of assessment, e.g. '20230310_BL'
        
    Outputs:
        * trajectories (list):
        * angles (list):
        * forces (list): 
    """
        
    path = pathlib.Path("BO2STTrial/"+subject+"/"+assessment+"/xcl")
    list_ = list(path.glob(subject+" Cal*"))


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


def download_WBDSascii(subject):
    """
    Download the gait parameters, the trajectories of all the markers, the joint angles and the forces of the selected trial of the selected subject.

    Inputs:
        * subject (string): number of the AB subject, e.g. '03'
           
    Outputs:
        * trajectories (list):
        * angles (list):
        * forces (list): 
    """
        
    path = pathlib.Path("BO2STTrial/"+"WBDSascii/AB"+subject)
    list_ = list(path.glob("WBDS"+subject+"walk"+"T0*"))

    # marker trajectories
    trajectories = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        if 'mkr' in file_name:
            exc_trial = pd.read_csv(l, sep="\t") 
            trajectories.append(tuple((file_name, exc_trial)))  

    # joint angles
    angles = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        if 'ang' in file_name:
            exc_trial = pd.read_csv(l, sep="\t") 
            angles.append(tuple((file_name, exc_trial)))   

    # forces    
    forces = []
    for l in list_:
        file_name = os.path.splitext(os.path.basename(l))[0]
        if 'grf' in file_name:
            exc_trial = pd.read_csv(l, sep="\t") 
            forces.append(tuple((file_name, exc_trial)))
              
    return trajectories, angles, forces 


