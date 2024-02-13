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


def centeroidnp(arr):
    """
    Compute the centroid of an array.
    Input:
        * arr (list): array containing the marker positions for which you want to compute the centroid (e.g. LPSI, RPSI, LASI, RASI).
    Output:
        * centroid (float): centroid 
    """
    length = len(arr)
    sum_x = np.sum(arr[:])
    centroid = sum_x/length
    return centroid

def compute_step_length(trajectories, ID, new_ss = False, T0=False, T1=False, T2=False):
    """
    Compute the mean step length for all the trials at self-selected speed.
    Since both feet are on the ground at the same time during a heel strike, the feet are stationary relative to one other, so we can have a direct calculation of step length by subtracting the location of one heel from the other. Thus we calculate step length as the distance between the right and left heels at the moment of HEEL STRIKE.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_step_lengths (float): mean value of the right step lengths at self-selected speed
        * mean_left_step_lengths (float): mean value of the left step lengths at self-selected speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    mean_right_step_lengths_ss = []; mean_left_step_lengths_ss = []
    std_right_step_lengths_ss = []; std_left_step_lengths_ss = []

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
                # extract heel and toe trajectories
                trajectory_heel_L = trajectories[j][1][ID+":"+'L'+"HEE"].Y
                trajectory_toe_L = trajectories[j][1][ID+":"+'L'+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_heel_L, trajectory_toe_L)
                # extract right heel trajectory
                trajectory_heel_R = trajectories[j][1][ID+":"+'R'+"HEE"].Y
                trajectory_toe_R = trajectories[j][1][ID+":"+'R'+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_heel_R, trajectory_toe_R)
               
                # compute left and right step lengths
                right_step_lengths = []
                left_step_lengths = []
                for l in range(len(heel_strike_R)):
                    right_step_lengths.append(float(trajectory_heel_R.iloc[heel_strike_R[l]].values) - float(trajectory_heel_L.iloc[heel_strike_R[l]].values))
                for m in range(len(heel_strike_L)):
                    left_step_lengths.append(float(trajectory_heel_L.iloc[heel_strike_L[m]].values) - float(trajectory_heel_R.iloc[heel_strike_L[m]].values))
                
                mean_right_step_lengths_ss.append(np.mean(right_step_lengths))
                mean_left_step_lengths_ss.append(np.mean(left_step_lengths))
                std_right_step_lengths_ss.append(np.std(right_step_lengths))
                std_left_step_lengths_ss.append(np.std(left_step_lengths))


    # compute mean and std of right/left step lengths
    mean_right_step_lengths = np.nanmean(mean_right_step_lengths_ss)
    mean_left_step_lengths = np.nanmean(mean_left_step_lengths_ss)  
    std_right_step_lengths = np.nanmean(std_right_step_lengths_ss)
    std_left_step_lengths = np.nanmean(std_left_step_lengths_ss)  
                                                                                                                        
    return mean_right_step_lengths, std_right_step_lengths, mean_left_step_lengths, std_left_step_lengths


def compute_stride_length(trajectories, ID, new_ss = False, T0=False, T1=False, T2=False):
    """
    Compute the mean stride length for all the trials at self-selected speed.
    Since both feet are on the ground at the same time during a heel strike, the feet are stationary relative to one other, so we can have a direct calculation of step length by subtracting the location of one heel from the other. Thus we calculate step length as the distance between the right and left heels at the moment of HEEL STRIKE.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_stride_lengths (float): mean value of the right stride lengths at self-selected speed
        * mean_left_stride_lengths (float): mean value of the left stride lengths at self-selected speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    mean_stride_lengths_ss = []
    std_stride_lengths_ss = []
   
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
                # extract heel and toe trajectories
                trajectory_heel_L = trajectories[j][1][ID+":"+'L'+"HEE"].Y
                trajectory_toe_L = trajectories[j][1][ID+":"+'L'+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_heel_L, trajectory_toe_L)
                # extract right heel trajectory
                trajectory_heel_R = trajectories[j][1][ID+":"+'R'+"HEE"].Y
                trajectory_toe_R = trajectories[j][1][ID+":"+'R'+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_heel_R, trajectory_toe_R)
               
                # compute left and right step lengths
                right_step_lengths = []
                left_step_lengths = []
                for l in range(len(heel_strike_R)):
                    right_step_lengths.append(float(trajectory_heel_R.iloc[heel_strike_R[l]].values) - float(trajectory_heel_L.iloc[heel_strike_R[l]].values))
                for m in range(len(heel_strike_L)):
                    left_step_lengths.append(float(trajectory_heel_L.iloc[heel_strike_L[m]].values) - float(trajectory_heel_R.iloc[heel_strike_L[m]].values))
                
                # compute stride length by summing two consecutive step lengths 
                min_ = min(len(right_step_lengths), len(left_step_lengths))
                mean_stride_lengths = []
                for p in range(min_):
                    mean_stride_lengths.append(right_step_lengths[p]+left_step_lengths[p])
                mean_stride_lengths_ss.append(np.mean(mean_stride_lengths))
                std_stride_lengths_ss.append(np.std(mean_stride_lengths))    

    # compute mean and std of stride lengths
    mean_stride_lengths = np.nanmean(mean_stride_lengths_ss)
    std_stride_lengths = np.nanmean(std_stride_lengths_ss)
                                                                                                                        
    return mean_stride_lengths, std_stride_lengths


def compute_step_length_AB(trajectories_AB, number_AB):
    """
    Compute the mean step length for all the trials at self-selected speed.
    Since both feet are on the ground at the same time during a heel strike, the feet are stationary relative to one other, so we can have a direct calculation of step length by subtracting the location of one heel from the other. Thus we calculate step length as the distance between the right and left heels at the moment of HEEL STRIKE.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_step_lengths (float): mean value of the right step lengths at self-selected speed
        * mean_left_step_lengths (float): mean value of the left step lengths at self-selected speed
    """
    
    mean_right_step_lengths_ss = []; mean_left_step_lengths_ss = []
    std_right_step_lengths_ss = []; std_left_step_lengths_ss = []
    
    for i in range(len(trajectories_AB)): # ss speed
            # sort the trials
            if (trajectories_AB[i][0] == 'WBDS'+number_AB+'walkT05mkr'):
                # extract heel and toe trajectories
                trajectory_heel_L = trajectories_AB[i][1]['L' + '.HeelX']
                trajectory_toe_L = trajectories_AB[i][1]['L' + '.MT1X']
                # extract heel_strike and toe_off events
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_heel_L, trajectory_toe_L)
                # extract right heel trajectory
                trajectory_heel_R = trajectories_AB[i][1]['R' + '.HeelX']
                trajectory_toe_R = trajectories_AB[i][1]['R' + '.MT1X']
                # extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_heel_R, trajectory_toe_R)
               
                # compute left and right step lengths
                right_step_lengths = []
                left_step_lengths = []
                for l in range(len(heel_strike_R)):
                    right_step_lengths.append(float(trajectory_heel_R.iloc[heel_strike_R[l]]) - float(trajectory_heel_L.iloc[heel_strike_R[l]]))
                for m in range(len(heel_strike_L)):
                    left_step_lengths.append(float(trajectory_heel_L.iloc[heel_strike_L[m]]) - float(trajectory_heel_R.iloc[heel_strike_L[m]]))
                
                mean_right_step_lengths_ss.append(np.mean(right_step_lengths))
                mean_left_step_lengths_ss.append(np.mean(left_step_lengths))
                std_right_step_lengths_ss.append(np.std(right_step_lengths))
                std_left_step_lengths_ss.append(np.std(left_step_lengths))


    # compute mean and std of right/left step lengths
    mean_right_step_lengths = np.nanmean(mean_right_step_lengths_ss)
    mean_left_step_lengths = np.nanmean(mean_left_step_lengths_ss)  
    std_right_step_lengths = np.nanmean(std_right_step_lengths_ss)
    std_left_step_lengths = np.nanmean(std_left_step_lengths_ss)  
                                                                                                                        
    return mean_right_step_lengths, std_right_step_lengths, mean_left_step_lengths, std_left_step_lengths


def compute_stride_length_AB(trajectories_AB, number_AB):
    """
    Compute the mean stride length for all the trials at self-selected speed.
    Since both feet are on the ground at the same time during a heel strike, the feet are stationary relative to one other, so we can have a direct calculation of step length by subtracting the location of one heel from the other. Thus we calculate step length as the distance between the right and left heels at the moment of HEEL STRIKE.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_stride_lengths (float): mean value of the right stride lengths at self-selected speed
        * mean_left_stride_lengths (float): mean value of the left stride lengths at self-selected speed
    """
    
    mean_stride_lengths_ss = []
    std_stride_lengths_ss = []
    
    for i in range(len(trajectories_AB)): # ss speed
            # sort the trials
            if (trajectories_AB[i][0] == 'WBDS'+number_AB+'walkT05mkr'):
                # extract heel and toe trajectories
                trajectory_heel_L = trajectories_AB[i][1]['L' + '.HeelX']
                trajectory_toe_L = trajectories_AB[i][1]['L' + '.MT1X']
                # extract heel_strike and toe_off events
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_heel_L, trajectory_toe_L)
                # extract right heel trajectory
                trajectory_heel_R = trajectories_AB[i][1]['R' + '.HeelX']
                trajectory_toe_R = trajectories_AB[i][1]['R' + '.MT1X']
                # extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_heel_R, trajectory_toe_R)
               
                # compute left and right step lengths
                right_step_lengths = []
                left_step_lengths = []
                for l in range(len(heel_strike_R)):
                    right_step_lengths.append(float(trajectory_heel_R.iloc[heel_strike_R[l]]) - float(trajectory_heel_L.iloc[heel_strike_R[l]]))
                for m in range(len(heel_strike_L)):
                    left_step_lengths.append(float(trajectory_heel_L.iloc[heel_strike_L[m]]) - float(trajectory_heel_R.iloc[heel_strike_L[m]]))
                
                mean_stride_lengths = []
                # compute stride length by summing two consecutive step lengths 
                min_ = min(len(right_step_lengths), len(left_step_lengths))
                mean_stride_lengths = []
                for p in range(min_):
                    mean_stride_lengths.append(right_step_lengths[p]+left_step_lengths[p])
                mean_stride_lengths_ss.append(np.mean(mean_stride_lengths))
                std_stride_lengths_ss.append(np.std(mean_stride_lengths))    

    # compute mean and std of stride lengths
    mean_stride_lengths = np.nanmean(mean_stride_lengths_ss)
    std_stride_lengths = np.nanmean(std_stride_lengths_ss)  
                                                                                                                        
    return mean_stride_lengths, std_stride_lengths


def compute_step_width(trajectories, ID, new_ss=False, T0=False, T1=False, T2=False):
    """
    Compute the mean step width for all the trials at self-selected speed.
    The left/right step width is computed by computing the distance along the x-axis between the position of the left/right heel marker and the position of the centroid defined by the LPSI/RPSI/LASI/RASI markers at heel strike. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_step_widths (float): mean value of the right step widths at self-selected speed
        * mean_left_step_widths (float): mean value of the left step widths at self-selected speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    mean_right_step_widths_ss = []; mean_left_step_widths_ss = []
    std_right_step_widths_ss = []; std_left_step_widths_ss = []

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
                # extract trajectories 
                trajectory_LPSI = trajectories[j][1][ID+':LPSI'].X
                trajectory_RPSI = trajectories[j][1][ID+':RPSI'].X
                trajectory_RASI = trajectories[j][1][ID+':RASI'].X
                trajectory_LASI = trajectories[j][1][ID+':LASI'].X
                trajectory_LHEE = trajectories[j][1][ID+':LHEE'].X
                trajectory_RHEE = trajectories[j][1][ID+':RHEE'].X
                trajectory_LHEE_Y = trajectories[j][1][ID+':LHEE'].Y
                trajectory_RHEE_Y = trajectories[j][1][ID+':RHEE'].Y
                trajectory_LTOE_Y = trajectories[j][1][ID+':LTOE'].Y
                trajectory_RTOE_Y = trajectories[j][1][ID+':RTOE'].Y
                
                # compute centroid
                centroid = []
                for n in range(len(trajectory_LPSI)):
                    arr = [float(trajectory_LPSI.iloc[n].values), float(trajectory_RPSI.iloc[n].values), float(trajectory_RASI.iloc[n].values), float(trajectory_LASI.iloc[n].values)]
                    centroid.append(centeroidnp(arr))
                
                # compute gait cycle thanks to left heel trajectory and extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_RHEE_Y, trajectory_RTOE_Y)
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_LHEE_Y, trajectory_LTOE_Y)
                
                # compute left and right step widths
                right_step_widths = []
                left_step_widths = []
                for l in range(len(heel_strike_R)):
                    right_step_widths.append(math.dist([float(trajectory_RHEE.iloc[heel_strike_R[l]].values)], [centroid[heel_strike_R[l]]]))
                for m in range(len(heel_strike_L)):
                    left_step_widths.append(math.dist([float(trajectory_LHEE.iloc[heel_strike_L[m]].values)], [centroid[heel_strike_L[m]]]))

                mean_right_step_widths_ss.append(np.mean(right_step_widths))
                mean_left_step_widths_ss.append(np.mean(left_step_widths))
                std_right_step_widths_ss.append(np.std(right_step_widths))
                std_left_step_widths_ss.append(np.std(left_step_widths))
                    
   
    # compute mean and std of right/left step widths
    mean_right_step_widths = np.nanmean(mean_right_step_widths_ss)
    mean_left_step_widths = np.nanmean(mean_left_step_widths_ss)    
    std_right_step_widths = np.nanmean(std_right_step_widths_ss)
    std_left_step_widths = np.nanmean(std_left_step_widths_ss)    
        
    return mean_right_step_widths, std_right_step_widths, mean_left_step_widths, std_left_step_widths


def compute_stride_width(trajectories, ID, new_ss=False, T0=False, T1=False, T2=False):
    """
    Compute the mean stride width for all the trials at self-selected speed.
    The left/right stride width is computed by computing the distance along the x-axis between the position of the left/right heel marker and the position of the centroid defined by the LPSI/RPSI/LASI/RASI markers at heel strike. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_stride_widths (float): mean value of the right stride widths at self-selected speed
        * mean_left_stride_widths (float): mean value of the left stride widths at self-selected speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    mean_stride_widths_ss = []
    std_stride_widths_ss = []

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
                # extract trajectories 
                trajectory_LPSI = trajectories[j][1][ID+':LPSI'].X
                trajectory_RPSI = trajectories[j][1][ID+':RPSI'].X
                trajectory_RASI = trajectories[j][1][ID+':RASI'].X
                trajectory_LASI = trajectories[j][1][ID+':LASI'].X
                trajectory_LHEE = trajectories[j][1][ID+':LHEE'].X
                trajectory_RHEE = trajectories[j][1][ID+':RHEE'].X
                trajectory_LHEE_Y = trajectories[j][1][ID+':LHEE'].Y
                trajectory_RHEE_Y = trajectories[j][1][ID+':RHEE'].Y
                trajectory_LTOE_Y = trajectories[j][1][ID+':LTOE'].Y
                trajectory_RTOE_Y = trajectories[j][1][ID+':RTOE'].Y
                
                # compute centroid
                centroid = []
                for n in range(len(trajectory_LPSI)):
                    arr = [float(trajectory_LPSI.iloc[n].values), float(trajectory_RPSI.iloc[n].values), float(trajectory_RASI.iloc[n].values), float(trajectory_LASI.iloc[n].values)]
                    centroid.append(centeroidnp(arr))
                
                # compute gait cycle thanks to left heel trajectory and extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_RHEE_Y, trajectory_RTOE_Y)
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_LHEE_Y, trajectory_LTOE_Y)
                
                # compute left and right step widths
                right_step_widths = []
                left_step_widths = []
                for l in range(len(heel_strike_R)):
                    right_step_widths.append(math.dist([float(trajectory_RHEE.iloc[heel_strike_R[l]].values)], [centroid[heel_strike_R[l]]]))
                for m in range(len(heel_strike_L)):
                    left_step_widths.append(math.dist([float(trajectory_LHEE.iloc[heel_strike_L[m]].values)], [centroid[heel_strike_L[m]]]))

                mean_stride_widths = []
                # compute stride length by summing two consecutive step lengths 
                min_ = min(len(right_step_widths), len(left_step_widths))
                for p in range(min_):
                    mean_stride_widths.append(right_step_widths[p]+left_step_widths[p])
                mean_stride_widths_ss.append(np.nanmean(mean_stride_widths))
                std_stride_widths_ss.append(np.nanstd(mean_stride_widths))    

    # compute mean and std of stride lengths
    mean_stride_widths = np.nanmean(mean_stride_widths_ss)
    std_stride_widths = np.nanmean(std_stride_widths_ss)   
        
    return mean_stride_widths, std_stride_widths



def compute_step_width_AB(trajectories_AB, number_AB):
    """
    Compute the mean step width for all the trials at self-selected speed.
    The left/right step width is computed by computing the distance along the x-axis between the position of the left/right heel marker and the position of the centroid defined by the LPSI/RPSI/LASI/RASI markers at heel strike. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_step_widths (float): mean value of the right step widths at self-selected speed
        * mean_left_step_widths (float): mean value of the left step widths at self-selected speed
    """

    # recover and sort the trials in ascending order
    
    mean_right_step_widths_ss = []; mean_left_step_widths_ss = []
    std_right_step_widths_ss = []; std_left_step_widths_ss = []
    
    for i in range(len(trajectories_AB)): # ss speed
            # sort the trials
            if (trajectories_AB[i][0] == 'WBDS'+number_AB+'walkT05mkr'):
                # extract trajectories 
                trajectory_LPSI = trajectories_AB[i][1]['L' + '.PSISZ']
                trajectory_RPSI = trajectories_AB[i][1]['R' + '.PSISZ']
                trajectory_RASI = trajectories_AB[i][1]['R' + '.ASISZ']
                trajectory_LASI = trajectories_AB[i][1]['L' + '.ASISZ']
                trajectory_LHEE = trajectories_AB[i][1]['L' + '.HeelZ']
                trajectory_RHEE = trajectories_AB[i][1]['R' + '.HeelZ']
                trajectory_LHEE_Y = trajectories_AB[i][1]['L' + '.HeelX']
                trajectory_RHEE_Y = trajectories_AB[i][1]['R' + '.HeelX']
                trajectory_LTOE_Y = trajectories_AB[i][1]['L' + '.MT1X']
                trajectory_RTOE_Y = trajectories_AB[i][1]['R' + '.MT1X']
                
                # compute centroid
                centroid = []
                for n in range(len(trajectory_LPSI)):
                    arr = [float(trajectory_LPSI.iloc[n]), float(trajectory_RPSI.iloc[n]), float(trajectory_RASI.iloc[n]), float(trajectory_LASI.iloc[n])]
                    centroid.append(centeroidnp(arr))
                
                # compute gait cycle thanks to left heel trajectory and extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_RHEE_Y, trajectory_RTOE_Y)
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_LHEE_Y, trajectory_LTOE_Y)
                
                # compute left and right step widths
                right_step_widths = []
                left_step_widths = []
                for l in range(len(heel_strike_R)):
                    right_step_widths.append(math.dist([float(trajectory_RHEE.iloc[heel_strike_R[l]])], [centroid[heel_strike_R[l]]]))
                for m in range(len(heel_strike_L)):
                    left_step_widths.append(math.dist([float(trajectory_LHEE.iloc[heel_strike_L[m]])], [centroid[heel_strike_L[m]]]))

                mean_right_step_widths_ss.append(np.mean(right_step_widths))
                mean_left_step_widths_ss.append(np.mean(left_step_widths))
                std_right_step_widths_ss.append(np.std(right_step_widths))
                std_left_step_widths_ss.append(np.std(left_step_widths))
                    
   
    # compute mean and std of right/left step widths
    mean_right_step_widths = np.nanmean(mean_right_step_widths_ss)
    mean_left_step_widths = np.nanmean(mean_left_step_widths_ss)    
    std_right_step_widths = np.nanmean(std_right_step_widths_ss)
    std_left_step_widths = np.nanmean(std_left_step_widths_ss)    
        
    return mean_right_step_widths, std_right_step_widths, mean_left_step_widths, std_left_step_widths


def compute_stride_width_AB(trajectories_AB, number_AB):
    """
    Compute the mean stride width for all the trials at self-selected speed.
    The left/right stride width is computed by computing the distance along the x-axis between the position of the left/right heel marker and the position of the centroid defined by the LPSI/RPSI/LASI/RASI markers at heel strike. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * mean_right_stride_widths (float): mean value of the right stride widths at self-selected speed
        * mean_left_stride_widths (float): mean value of the left stride widths at self-selected speed
    """

    # recover and sort the trials in ascending order
    
    mean_stride_widths_ss = []
    std_stride_widths_ss = []
    
    for i in range(len(trajectories_AB)): # ss speed
            # sort the trials
            if (trajectories_AB[i][0] == 'WBDS'+number_AB+'walkT05mkr'):
                # extract trajectories 
                trajectory_LPSI = trajectories_AB[i][1]['L' + '.PSISZ']
                trajectory_RPSI = trajectories_AB[i][1]['R' + '.PSISZ']
                trajectory_RASI = trajectories_AB[i][1]['R' + '.ASISZ']
                trajectory_LASI = trajectories_AB[i][1]['L' + '.ASISZ']
                trajectory_LHEE = trajectories_AB[i][1]['L' + '.HeelZ']
                trajectory_RHEE = trajectories_AB[i][1]['R' + '.HeelZ']
                trajectory_LHEE_Y = trajectories_AB[i][1]['L' + '.HeelX']
                trajectory_RHEE_Y = trajectories_AB[i][1]['R' + '.HeelX']
                trajectory_LTOE_Y = trajectories_AB[i][1]['L' + '.MT1X']
                trajectory_RTOE_Y = trajectories_AB[i][1]['R' + '.MT1X']
                
                # compute centroid
                centroid = []
                for n in range(len(trajectory_LPSI)):
                    arr = [float(trajectory_LPSI.iloc[n]), float(trajectory_RPSI.iloc[n]), float(trajectory_RASI.iloc[n]), float(trajectory_LASI.iloc[n])]
                    centroid.append(centeroidnp(arr))
                
                # compute gait cycle thanks to left heel trajectory and extract heel_strike and toe_off events
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_RHEE_Y, trajectory_RTOE_Y)
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_LHEE_Y, trajectory_LTOE_Y)
                
                # compute left and right step widths
                right_step_widths = []
                left_step_widths = []
                for l in range(len(heel_strike_R)):
                    right_step_widths.append(math.dist([float(trajectory_RHEE.iloc[heel_strike_R[l]])], [centroid[heel_strike_R[l]]]))
                for m in range(len(heel_strike_L)):
                    left_step_widths.append(math.dist([float(trajectory_LHEE.iloc[heel_strike_L[m]])], [centroid[heel_strike_L[m]]]))

                mean_stride_widths = []
                # compute stride length by summing two consecutive step lengths 
                min_ = min(len(right_step_widths), len(left_step_widths))
                for p in range(min_):
                    mean_stride_widths.append(right_step_widths[p]+left_step_widths[p])
                mean_stride_widths_ss.append(np.mean(mean_stride_widths))
                std_stride_widths_ss.append(np.std(mean_stride_widths))    

    # compute mean and std of stride lengths
    mean_stride_widths = np.nanmean(mean_stride_widths_ss)
    std_stride_widths = np.nanmean(std_stride_widths_ss)    
        
    return mean_stride_widths, std_stride_widths


def compute_cadence(trajectories, ID, new_ss=False, T0=False, T1=False, T2=False):
    """
    Compute the cadence (number steps/min) for the selected participant averaged over the 3 trials at self-selected speed.
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Output:
        * cadence (float): cadence of the participant 
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    cadences = []
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
                # extract heel and toe trajectories 
                trajectory_L_TOE_Y = trajectories[j][1][ID+':' + 'LTOE'].Y
                trajectory_L_HEE_Y = trajectories[j][1][ID+':' + 'LHEE'].Y
                trajectory_R_TOE_Y = trajectories[j][1][ID+':' + 'RTOE'].Y
                trajectory_R_HEE_Y = trajectories[j][1][ID+':' + 'RHEE'].Y
                # extract heel_strike and toe_off events of left and right leg
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_L_HEE_Y, trajectory_L_TOE_Y) 
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_R_HEE_Y, trajectory_R_TOE_Y) 

                # compute total number of heel strikes
                total_heel_strikes = len(heel_strike_L)+len(heel_strike_R)

                # find time frame of 1st HS
                if (heel_strike_L[0]<heel_strike_R[0]):
                    first_HS = heel_strike_L[0]
                else: 
                    first_HS = heel_strike_R[0]
                    
                # find time frame of last HS
                if (heel_strike_L[-1]>heel_strike_R[-1]):
                    last_HS = heel_strike_L[-1]
                else: 
                    last_HS = heel_strike_R[-1]

                # compute delta time frames between first and last HS
                delta_time_frame = last_HS - first_HS

                # compute delta seconds using the sampling frequency
                sf = 250 # Hz
                delta_time = delta_time_frame/sf

                # compute cadence
                cad = total_heel_strikes/delta_time
                cadences.append(cad)

    # compute mean cadence averaged over the 3 trials at ss speed 
    cadence = np.mean(cadences)*60
    
    return cadence

def compute_cadence_AB(trajectories_AB, number_AB):
    """
    Compute the cadence (number steps/min) for the selected participant averaged over the 3 trials at self-selected speed.
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Output:
        * cadence (float): cadence of the participant 
    """
    
    for i in range(len(trajectories_AB)): # ss speed
            # sort the trials
            if (trajectories_AB[i][0] == 'WBDS'+number_AB+'walkT05mkr'):
                # extract heel and toe trajectories 
                trajectory_L_TOE_Y = trajectories_AB[i][1]['L' + '.MT1X']
                trajectory_L_HEE_Y = trajectories_AB[i][1]['L' + '.HeelX']
                trajectory_R_TOE_Y = trajectories_AB[i][1]['R' + '.MT1X']
                trajectory_R_HEE_Y = trajectories_AB[i][1]['R' + '.HeelX']
                # extract heel_strike and toe_off events of left and right leg
                heel_strike_L, toe_off_L = compute_gait_cycle_2(trajectory_L_HEE_Y, trajectory_L_TOE_Y) 
                heel_strike_R, toe_off_R = compute_gait_cycle_2(trajectory_R_HEE_Y, trajectory_R_TOE_Y) 

                # compute total number of heel strikes
                total_heel_strikes = len(heel_strike_L)+len(heel_strike_R)

                # find time frame of 1st HS
                if (heel_strike_L[0]<heel_strike_R[0]):
                    first_HS = heel_strike_L[0]
                else: 
                    first_HS = heel_strike_R[0]
                    
                # find time frame of last HS
                if (heel_strike_L[-1]>heel_strike_R[-1]):
                    last_HS = heel_strike_L[-1]
                else: 
                    last_HS = heel_strike_R[-1]

                # compute delta time frames between first and last HS
                delta_time_frame = last_HS - first_HS

                # compute delta seconds using the sampling frequency
                sf = 150 # Hz
                delta_time = delta_time_frame/sf

                # compute cadence
                cadence = total_heel_strikes/delta_time

    # convert into number of steps/minute
    cadence = cadence*60
    
    return cadence