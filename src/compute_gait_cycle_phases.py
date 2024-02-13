import numpy as np
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
import natsort
import math
import pathlib
from src.dataloader import *
from src.gait_cycle import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_stance_swing_durations(trajectories, ID, side, new_ss=False, T0 = False, T1 = False, T2 = False):
    """
    Compute the duration of the stance and swing phases in percentage of the gait cycle at self-selected speed
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): "L" for left or "R" for right side
        
    Outputs:
        * stance_perc_mean (list): mean of the duration of the stance phase in percentage of the gait cycle  
        * swing_perc_mean (list): mean of the duration of the swing phase in percentage of the gait cycle 
        * phase_std (list): std of the duration of the swing/stance phase in percentage of the gait cycle 
    """

    # compute mean and std of stance and swing percentage of gait cycle
    stance_perc_mean = []; stance_perc_std = []; swing_perc_mean = []; swing_perc_std = []
    stance_perc_ss = []; swing_perc_ss = []
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    new_ss_index = [0,1,2]
    
    if (new_ss==True):
        EV_new_ss_index_file_name = ("BO2STTrial/new_ss_trials.xlsx")
        EV_new_ss_index_xcl = pd.read_excel(EV_new_ss_index_file_name, header = [0], index_col = [0])
        if (T0==True):
            new_ss_index = list(range(int(EV_new_ss_index_xcl.loc[ID][['new_ss_index_T0']].values.squeeze())))
        if (T1==True):
            new_ss_index = list(range(int(EV_new_ss_index_xcl.loc[ID][['new_ss_index_T1']].values.squeeze())))
        if (T2==True):
            new_ss_index = list(range(int(EV_new_ss_index_xcl.loc[ID][['new_ss_index_T2']].values.squeeze())))
    
    for i in new_ss_index: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                # extract heel trajectory
                trajectory_heel = trajectories[j][1][ID+":"+side+"HEE"].Y
                trajectory_toe = trajectories[j][1][ID+":"+side+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                if (ID=="BO2ST_109" and T1 ==True and new_ss == False):
                    heel_strike.pop(2)
                # compute stance and swing and total gait cycle durations
                stance = []; swing = []; gait_cycle = []
                min_ = min(len(heel_strike), len(toe_off))
                for k in range(min_-1):
                    a = heel_strike[k]
                    b = toe_off[k]
                    if (b<a):
                        b = toe_off[k+1]
                    c = heel_strike[k+1]
                    # stance duration
                    stance.append(trajectory_heel.index.values[b] - trajectory_heel.index.values[a])
                    # swing duration 
                    swing.append(trajectory_heel.index.values[c] - trajectory_heel.index.values[b])
                    # gait cycle duration
                    gait_cycle.append(trajectory_heel.index.values[c] - trajectory_heel.index.values[a])
                # compute stance and swing percentage of gait cycle 
                stance_perc = []; swing_perc = []
                for l in range(len(stance)):
                    stance_perc_ss.append(stance[l]/gait_cycle[l]*100)
                    swing_perc_ss.append(swing[l]/gait_cycle[l]*100)

    stance_perc_mean = np.mean(stance_perc_ss)
    swing_perc_mean = np.mean(swing_perc_ss) 
    phase_std = np.std(swing_perc_ss) # same as np.std(stance_perc_ss)
       
                    
    return stance_perc_mean, swing_perc_mean, phase_std            


def compute_stance_swing_durations_AB(trajectories_AB, number_AB, side):
    """
    Compute the duration of the stance and swing phases in percentage of the gait cycle at self-selected speed
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): "L" for left or "R" for right side
        
    Outputs:
        * stance_perc_mean (list): mean of the duration of the stance phase in percentage of the gait cycle  
        * swing_perc_mean (list): mean of the duration of the swing phase in percentage of the gait cycle 
        * phase_std (list): std of the duration of the swing/stance phase in percentage of the gait cycle 
    """

    # compute mean and std of stance and swing percentage of gait cycle
    stance_perc_mean = []; stance_perc_std = []; swing_perc_mean = []; swing_perc_std = []
    stance_perc_ss = []; swing_perc_ss = []
    
    for i in range(len(trajectories_AB)): # ss speed
        # sort the trials
        if (trajectories_AB[i][0] == 'WBDS'+number_AB+'walkT05mkr'):
            # extract heel trajectory
            trajectory_heel = trajectories_AB[i][1][side + '.HeelX']
            trajectory_toe = trajectories_AB[i][1][side + '.MT1X']
            # extract heel_strike and toe_off events
            heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
            # compute stance and swing and total gait cycle durations
            stance = []; swing = []; gait_cycle = []
            min_ = min(len(heel_strike), len(toe_off))
            for k in range(min_-1):
                a = heel_strike[k]
                b = toe_off[k]
                if (b<a):
                    b = toe_off[k+1]
                c = heel_strike[k+1]
                # stance duration
                stance.append(trajectory_heel.index.values[b] - trajectory_heel.index.values[a])
                # swing duration 
                swing.append(trajectory_heel.index.values[c] - trajectory_heel.index.values[b])
                # gait cycle duration
                gait_cycle.append(trajectory_heel.index.values[c] - trajectory_heel.index.values[a])
            # compute stance and swing percentage of gait cycle 
            stance_perc = []; swing_perc = []
            for l in range(len(stance)):
                stance_perc_ss.append(stance[l]/gait_cycle[l]*100)
                swing_perc_ss.append(swing[l]/gait_cycle[l]*100)

    stance_perc_mean = np.mean(stance_perc_ss)
    swing_perc_mean = np.mean(swing_perc_ss) 
    phase_std = np.std(swing_perc_ss) # same as np.std(stance_perc_ss)
       
                    
    return stance_perc_mean, swing_perc_mean, phase_std            
                    