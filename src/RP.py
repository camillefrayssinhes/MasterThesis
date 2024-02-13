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
from scipy.signal import hilbert
from scipy import signal
import plotly.express as px
from src.gait_cycle import *

def compute_relative_phase_contralateral(trajectories, angles, ID, MA_side, LA_side, shift, BL = True, T2 = False, plot=True):
    """
    Compute the (discrete) relative phase of the contralateral hip-shoulder pair for the three trials at the self-selected speed. Takes the hip of the MA_side and the shoulder of the LA_side (reverse the order in the attributes given to the function if you want the reverse hip-shoulder pair). 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * shift (int): shift for the minimum distance between two peaks that will be substracted to the minimum duration of a gait cycle. The shift parameter is determined by individual examination of the plots checking that we select all the maxima.
        * plot (bool): if True plot the trajectories and the peaks for the three trials at self-selected speed
    Outputs:
        * RPs (list): list of relative phases of the three trials at the self-selected speed. 
        * std (float): stp of relative phases
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(angles)):
        list_.append(angles[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    RPs = []
    
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
    
                # doing that just because there are some kinks with some trials 
                if (('ID'=='BO2ST_102' and MA_side =='L')):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[1000:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[1000:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[1000:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[1000:]
                    
                elif ((angles[j][0] == "BO2ST_102 Trial 03" and MA_side =='R' and BL == True) or (angles[j][0] == "BO2ST_102 Trial 01" and MA_side =='L' and BL == True)):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[100:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[100:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[100:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[100:] 
                    
                elif (((angles[j][0] == "BO2ST_103 Trial 01" or angles[j][0] == "BO2ST_103 Trial 03") and MA_side =='L' and BL == True) or (angles[j][0] == "BO2ST_103 Trial 02" and MA_side =='L' and BL == False and T2 == False)):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[400:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[400:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[400:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[400:]
                    
                elif ((angles[j][0] == "BO2ST_104 Trial 01" and MA_side =='R' and BL == True) or (angles[j][0] == "BO2ST_103 Trial 02" and T2 == True)):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[200:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[200:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[200:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[200:]
                    
                elif ((angles[j][0] == 'BO2ST_104 Trial 01' or angles[j][0] == 'BO2ST_104 Trial 02') and T2==True):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[200:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[200:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[200:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[200:]
                    
                    
                elif (angles[j][0] == "BO2ST_102 Trial 01" and MA_side =='R' and BL == False):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[1100:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[1100:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[1100:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[1100:]    
  
                elif (angles[j][0] == "BO2ST_101 Trial 03"):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[1200:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[1200:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[1200:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[1200:]    

                
                else:   
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y

                # compute minimum duration of a gait cycle 
                HS, TO = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                dur = []
                for m in range(len(HS) -1):
                    dur.append( HS[m+1] - HS[m] )
                min_dur = np.min(dur) - shift # this is the min distance between two peaks, so it is the minimum
                # duration of a gait cycle minus a shift defined for each participant 
                #print(np.min(dur))

                if (ID == "BO2ST_103" and T2 == True):
                    min_dur = 400
                if (ID == "BO2ST_104" or ID == "BO2ST_105"):
                    height_ = -10
                elif ((ID == "BO2ST_101" and MA_side == 'R') or (ID == "BO2ST_101" and T2 == True)):
                    height_ = -10
                else:
                    height_ = 0
                # take max extension of the hip and shoulder in each gait cycle 
                ilocs_max_hip = find_peaks(hip_angles_L.values.ravel(), height=height_, distance=min_dur)[0]
                ilocs_max_shoulder = find_peaks(shoulder_angles_L.values.ravel(), height=height_, distance = min_dur)[0]

                # compute number of gait cycles 
                len_ = min(ilocs_max_hip.size, ilocs_max_shoulder.size)

                # compute relative phase 
                for l in range(len_-1):
                    a = ilocs_max_shoulder[l]
                    #print(a)
                    b = ilocs_max_hip[l]
                    #print(b)
                    c = ilocs_max_hip[l+1]
                    #print(c)
                    phi = 360*(b-a)/(c-b)
                    #print(phi)
                    RPs.append(phi) # append relative phase 

                # plot
                if (plot):
                    plt.figure(figsize=(20,8))
                    plt.plot(hip_angles_L, label='hip')
                    plt.plot(hip_angles_L.iloc[ilocs_max_hip], '.', lw=10, color='red', marker="v")
                    plt.plot(shoulder_angles_L, label='shoulder')
                    plt.plot(shoulder_angles_L.iloc[ilocs_max_shoulder], '.', lw=10, color='green', marker="v")
                    plt.title('Hip angles and shoulder angles centered'); plt.ylabel('Angle [deg]'); plt.xlabel('Time points')
                    plt.tight_layout()
                    plt.legend()
                    plt.show();
                    
    std = np.std(RPs)
        
    return RPs, std    


def compute_relative_phase_contralateral_AB(trajectories_AB, angles_AB, ID_AB, shift, plot=True):
    """
    Compute the (discrete) relative phase of the contralateral hip-shoulder pair for the three trials at the self-selected speed. Takes the hip of the MA_side and the shoulder of the LA_side (reverse the order in the attributes given to the function if you want the reverse hip-shoulder pair). 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * shift (int): shift for the minimum distance between two peaks that will be substracted to the minimum duration of a gait cycle. The shift parameter is determined by individual examination of the plots checking that we select all the maxima.
        * plot (bool): if True plot the trajectories and the peaks for the three trials at self-selected speed
    Outputs:
        * RPs (list): list of relative phases of the three trials at the self-selected speed. 
        * std (float): stp of relative phases
    """

    RPs = []
    
    for i in range(len(angles_AB)):
        # ss speed
        if (angles_AB[i][0] == 'WBDS'+number_AB+'walkT05ang'):
            # extract left and right knee angles
            hip_angle_AB = angles_AB[i][1]['LHipAngleZ']
            ankle_angle_AB = angles_AB[i][1][side+'AnkleAngleZ']
    
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
    
                # doing that just because there are some kinks with some trials 
                if (('ID'=='BO2ST_102' and MA_side =='L')):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[1000:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[1000:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[1000:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[1000:]
                    
                elif ((angles[j][0] == "BO2ST_102 Trial 03" and MA_side =='R' and BL == True) or (angles[j][0] == "BO2ST_102 Trial 01" and MA_side =='L' and BL == True)):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[100:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[100:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[100:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[100:] 
                
                else:   
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y

                # compute minimum duration of a gait cycle 
                HS, TO = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                dur = []
                for m in range(len(HS) -1):
                    dur.append( HS[m+1] - HS[m] )
                min_dur = np.min(dur) - shift # this is the min distance between two peaks, so it is the minimum
                # duration of a gait cycle minus a shift defined for each participant 
                #print(np.min(dur))

                if (ID == "BO2ST_103" and T2 == True):
                    min_dur = 400
                else:
                    height_ = 0
                # take max extension of the hip and shoulder in each gait cycle 
                ilocs_max_hip = find_peaks(hip_angles_L.values.ravel(), height=height_, distance=min_dur)[0]
                ilocs_max_shoulder = find_peaks(shoulder_angles_L.values.ravel(), height=height_, distance = min_dur)[0]

                # compute number of gait cycles 
                len_ = min(ilocs_max_hip.size, ilocs_max_shoulder.size)

                # compute relative phase 
                for l in range(len_-1):
                    a = ilocs_max_shoulder[l]
                    #print(a)
                    b = ilocs_max_hip[l]
                    #print(b)
                    c = ilocs_max_hip[l+1]
                    #print(c)
                    phi = 360*(b-a)/(c-b)
                    #print(phi)
                    RPs.append(phi) # append relative phase 

                # plot
                if (plot):
                    plt.figure(figsize=(20,8))
                    plt.plot(hip_angles_L, label='hip')
                    plt.plot(hip_angles_L.iloc[ilocs_max_hip], '.', lw=10, color='red', marker="v")
                    plt.plot(shoulder_angles_L, label='shoulder')
                    plt.plot(shoulder_angles_L.iloc[ilocs_max_shoulder], '.', lw=10, color='green', marker="v")
                    plt.title('Hip angles and shoulder angles centered'); plt.ylabel('Angle [deg]'); plt.xlabel('Time points')
                    plt.tight_layout()
                    plt.legend()
                    plt.show();
                    
    std = np.std(RPs)
        
    return RPs, std    




def compute_relative_phase_contralateral_T1(trajectories, angles, ID, MA_side, LA_side, shift, plot=True):
    """
    Compute the (discrete) relative phase of the contralateral hip-shoulder pair for the three trials at the self-selected speed. Takes the hip of the MA_side and the shoulder of the LA_side (reverse the order in the attributes given to the function if you want the reverse hip-shoulder pair). 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * shift (int): shift for the minimum distance between two peaks that will be substracted to the minimum duration of a gait cycle. The shift parameter is determined by individual examination of the plots checking that we select all the maxima.
        * plot (bool): if True plot the trajectories and the peaks for the three trials at self-selected speed
    Outputs:
        * RPs (list): list of relative phases of the three trials at the self-selected speed. 
        * std (float): stp of relative phases
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(angles)):
        list_.append(angles[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    RPs = []
    
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
  
                if (angles[j][0] == 'BO2ST_101 Trial 01'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[0:3400]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[0:3400]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[0:3400]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[0:3400]
                    
                elif (angles[j][0] == 'BO2ST_104 Trial 01'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[0:3000]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[0:3000]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[0:3000]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[0:3000]    
                    
                elif (angles[j][0] == 'BO2ST_106 Trial 02'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[10:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[10:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[10:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[10:]
                    
                elif (angles[j][0] == 'BO2ST_104 Trial 02'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[1000:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[1000:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[1000:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[1000:] 
                    
                elif (angles[j][0] == 'BO2ST_105 Trial 02'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[1000:4000]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[1000:4000]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[1000:4000]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[1000:4000]     
                    
                elif (angles[j][0] == 'BO2ST_104 Trial 03' or angles[j][0] == 'BO2ST_105 Trial 03'):
                    break
                    
                else: 
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y

                # compute minimum duration of a gait cycle 
                HS, TO = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                dur = []
                for m in range(len(HS) -1):
                    dur.append( HS[m+1] - HS[m] )
                min_dur = np.min(dur) - shift # this is the min distance between two peaks, so it is the minimum
                # duration of a gait cycle minus a shift defined for each participant 
                #print(np.min(dur))

                if (ID=="BO2ST_104" or ID=="BO2ST_105"):
                    height_ = -5
                else:    
                    height_ = 0
                # take max extension of the hip and shoulder in each gait cycle 
                ilocs_max_hip = find_peaks(hip_angles_L.values.ravel(), height=height_, distance=min_dur)[0]
                ilocs_max_shoulder = find_peaks(shoulder_angles_L.values.ravel(), height=height_, distance = min_dur)[0]

                # compute number of gait cycles 
                len_ = min(ilocs_max_hip.size, ilocs_max_shoulder.size)

                # compute relative phase 
                for l in range(len_-1):
                    a = ilocs_max_shoulder[l]
                    #print(a)
                    b = ilocs_max_hip[l]
                    #print(b)
                    c = ilocs_max_hip[l+1]
                    #print(c)
                    phi = 360*(b-a)/(c-b)
                    #print(phi)
                    RPs.append(phi) # append relative phase 

                # plot
                if (plot):
                    plt.figure(figsize=(20,8))
                    plt.plot(hip_angles_L, label='hip')
                    plt.plot(hip_angles_L.iloc[ilocs_max_hip], '.', lw=10, color='red', marker="v")
                    plt.plot(shoulder_angles_L, label='shoulder')
                    plt.plot(shoulder_angles_L.iloc[ilocs_max_shoulder], '.', lw=10, color='green', marker="v")
                    plt.title('Hip angles and shoulder angles centered'); plt.ylabel('Angle [deg]'); plt.xlabel('Time points')
                    plt.tight_layout()
                    plt.legend()
                    plt.show();

    std = np.std(RPs)
        
    return RPs, std    

def compute_relative_phase_contralateral_F1(trajectories, angles, ID, MA_side, LA_side, shift, plot=True):
    """
    Compute the (discrete) relative phase of the contralateral hip-shoulder pair for the three trials at the self-selected speed. Takes the hip of the MA_side and the shoulder of the LA_side (reverse the order in the attributes given to the function if you want the reverse hip-shoulder pair). 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * shift (int): shift for the minimum distance between two peaks that will be substracted to the minimum duration of a gait cycle. The shift parameter is determined by individual examination of the plots checking that we select all the maxima.
        * plot (bool): if True plot the trajectories and the peaks for the three trials at self-selected speed
    Outputs:
        * RPs (list): list of relative phases of the three trials at the self-selected speed. 
        * std (float): stp of relative phases
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(angles)):
        list_.append(angles[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    RPs = []
    
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
  
                if (angles[j][0] == 'BO2ST_103 Trial 02'):
                    break
            
                elif (angles[j][0] == 'BO2ST_104 Trial 01'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[250:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[250:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[250:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[250:]
            
                else: 
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y

                # compute minimum duration of a gait cycle 
                HS, TO = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                dur = []
                for m in range(len(HS) -1):
                    dur.append( HS[m+1] - HS[m] )
                min_dur = np.min(dur) - shift # this is the min distance between two peaks, so it is the minimum
                # duration of a gait cycle minus a shift defined for each participant 
                #print(np.min(dur))

                if (ID=="BO2ST_105"):
                    height_ = -5
                else:    
                    height_ = 0
                # take max extension of the hip and shoulder in each gait cycle 
                ilocs_max_hip = find_peaks(hip_angles_L.values.ravel(), height=height_, distance=min_dur)[0]
                ilocs_max_shoulder = find_peaks(shoulder_angles_L.values.ravel(), height=height_, distance = min_dur)[0]

                # compute number of gait cycles 
                len_ = min(ilocs_max_hip.size, ilocs_max_shoulder.size)

                # compute relative phase 
                for l in range(len_-1):
                    a = ilocs_max_shoulder[l]
                    #print(a)
                    b = ilocs_max_hip[l]
                    #print(b)
                    c = ilocs_max_hip[l+1]
                    #print(c)
                    phi = 360*(b-a)/(c-b)
                    #print(phi)
                    RPs.append(phi) # append relative phase 

                # plot
                if (plot):
                    plt.figure(figsize=(20,8))
                    plt.plot(hip_angles_L, label='hip')
                    plt.plot(hip_angles_L.iloc[ilocs_max_hip], '.', lw=10, color='red', marker="v")
                    plt.plot(shoulder_angles_L, label='shoulder')
                    plt.plot(shoulder_angles_L.iloc[ilocs_max_shoulder], '.', lw=10, color='green', marker="v")
                    plt.title('Hip angles and shoulder angles centered'); plt.ylabel('Angle [deg]'); plt.xlabel('Time points')
                    plt.tight_layout()
                    plt.legend()
                    plt.show();

    std = np.std(RPs)
        
    return RPs, std    


def compute_relative_phase_contralateral_F4(trajectories, angles, ID, MA_side, LA_side, shift, plot=True):
    """
    Compute the (discrete) relative phase of the contralateral hip-shoulder pair for the three trials at the self-selected speed. Takes the hip of the MA_side and the shoulder of the LA_side (reverse the order in the attributes given to the function if you want the reverse hip-shoulder pair). 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * shift (int): shift for the minimum distance between two peaks that will be substracted to the minimum duration of a gait cycle. The shift parameter is determined by individual examination of the plots checking that we select all the maxima.
        * plot (bool): if True plot the trajectories and the peaks for the three trials at self-selected speed
    Outputs:
        * RPs (list): list of relative phases of the three trials at the self-selected speed. 
        * std (float): stp of relative phases
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(angles)):
        list_.append(angles[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    RPs = []
    
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
              
                if (angles[j][0] == 'BO2ST_105 Trial 02'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[450:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[450:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[450:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[450:]
            
                elif (angles[j][0] == 'BO2ST_103 Trial 02'):
                    break
            
                else: 
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y

                # compute minimum duration of a gait cycle 
                HS, TO = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                dur = []
                for m in range(len(HS) -1):
                    dur.append( HS[m+1] - HS[m] )
                min_dur = np.min(dur) - shift # this is the min distance between two peaks, so it is the minimum
                # duration of a gait cycle minus a shift defined for each participant 
                #print(np.min(dur))

                if (ID=="BO2ST_105" or ID=="BO2ST_104"):
                    height_ = -5
                elif (ID=="BO2ST_103"):
                    height_ = 9
                else:    
                    height_ = 0
                # take max extension of the hip and shoulder in each gait cycle 
                ilocs_max_hip = find_peaks(hip_angles_L.values.ravel(), height=height_, distance=min_dur)[0]
                ilocs_max_shoulder = find_peaks(shoulder_angles_L.values.ravel(), height=height_, distance = min_dur)[0]

                # compute number of gait cycles 
                len_ = min(ilocs_max_hip.size, ilocs_max_shoulder.size)

                # compute relative phase 
                for l in range(len_-1):
                    a = ilocs_max_shoulder[l]
                    #print(a)
                    b = ilocs_max_hip[l]
                    #print(b)
                    c = ilocs_max_hip[l+1]
                    #print(c)
                    phi = 360*(b-a)/(c-b)
                    #print(phi)
                    RPs.append(phi) # append relative phase 

                # plot
                if (plot):
                    plt.figure(figsize=(20,8))
                    plt.plot(hip_angles_L, label='hip')
                    plt.plot(hip_angles_L.iloc[ilocs_max_hip], '.', lw=10, color='red', marker="v")
                    plt.plot(shoulder_angles_L, label='shoulder')
                    plt.plot(shoulder_angles_L.iloc[ilocs_max_shoulder], '.', lw=10, color='green', marker="v")
                    plt.title('Hip angles and shoulder angles centered'); plt.ylabel('Angle [deg]'); plt.xlabel('Time points')
                    plt.tight_layout()
                    plt.legend()
                    plt.show();

    std = np.std(RPs)
        
    return RPs, std    

def compute_relative_phase_contralateral_F8(trajectories, angles, ID, MA_side, LA_side, shift, plot=True):
    """
    Compute the (discrete) relative phase of the contralateral hip-shoulder pair for the three trials at the self-selected speed. Takes the hip of the MA_side and the shoulder of the LA_side (reverse the order in the attributes given to the function if you want the reverse hip-shoulder pair). 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * shift (int): shift for the minimum distance between two peaks that will be substracted to the minimum duration of a gait cycle. The shift parameter is determined by individual examination of the plots checking that we select all the maxima.
        * plot (bool): if True plot the trajectories and the peaks for the three trials at self-selected speed
    Outputs:
        * RPs (list): list of relative phases of the three trials at the self-selected speed. 
        * std (float): stp of relative phases
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(angles)):
        list_.append(angles[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    RPs = []
    
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
              
                if (angles[j][0] == 'BO2ST_103 Trial 03'):
                    break
                 
                elif (angles[j][0] == 'BO2ST_103 Trial 01'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[0:3000]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[0:3000]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[0:3000]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[0:3000]
                    
                elif (angles[j][0] == 'BO2ST_104 Trial 01' or angles[j][0] == 'BO2ST_104 Trial 03' or angles[j][0] == 'BO2ST_105 Trial 02'):
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X[100:]
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X[100:]
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y[100:]
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y[100:]
                    
                else:     
                    # take angles and center them 
                    hip_angles_L = angles[j][1][ID+':' + MA_side + 'HipAngles'].X
                    hip_angles_L = hip_angles_L - np.nanmin(hip_angles_L) - (np.nanmax(hip_angles_L) - np.nanmin(hip_angles_L))/2
                    shoulder_angles_L = angles[j][1][ID+':' + LA_side + 'ShoulderAngles'].X
                    shoulder_angles_L = shoulder_angles_L - np.nanmin(shoulder_angles_L) - (np.nanmax(shoulder_angles_L) - np.nanmin(shoulder_angles_L))/2
                    # compute min distance between two peaks = min duration of a gait cycle
                    trajectory_heel = trajectories[j][1][ID+':'+MA_side+'HEE'].Y
                    trajectory_toe = trajectories[j][1][ID+':'+MA_side+'TOE'].Y

                # compute minimum duration of a gait cycle 
                HS, TO = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                dur = []
                for m in range(len(HS) -1):
                    dur.append( HS[m+1] - HS[m] )
                min_dur = np.min(dur) - shift # this is the min distance between two peaks, so it is the minimum
                # duration of a gait cycle minus a shift defined for each participant 
                #print(np.min(dur))

                if (ID=="BO2ST_105" or ID=="BO2ST_104"):
                    height_ = -5
                elif (ID=="BO2ST_103"):
                    height_ = 5
                else:    
                    height_ = 0
                # take max extension of the hip and shoulder in each gait cycle 
                ilocs_max_hip = find_peaks(hip_angles_L.values.ravel(), height=height_, distance=min_dur)[0]
                ilocs_max_shoulder = find_peaks(shoulder_angles_L.values.ravel(), height=height_, distance = min_dur)[0]

                # compute number of gait cycles 
                len_ = min(ilocs_max_hip.size, ilocs_max_shoulder.size)

                # compute relative phase 
                for l in range(len_-1):
                    a = ilocs_max_shoulder[l]
                    #print(a)
                    b = ilocs_max_hip[l]
                    #print(b)
                    c = ilocs_max_hip[l+1]
                    #print(c)
                    phi = 360*(b-a)/(c-b)
                    #print(phi)
                    RPs.append(phi) # append relative phase 

                # plot
                if (plot):
                    plt.figure(figsize=(20,8))
                    plt.plot(hip_angles_L, label='hip')
                    plt.plot(hip_angles_L.iloc[ilocs_max_hip], '.', lw=10, color='red', marker="v")
                    plt.plot(shoulder_angles_L, label='shoulder')
                    plt.plot(shoulder_angles_L.iloc[ilocs_max_shoulder], '.', lw=10, color='green', marker="v")
                    plt.title('Hip angles and shoulder angles centered'); plt.ylabel('Angle [deg]'); plt.xlabel('Time points')
                    plt.tight_layout()
                    plt.legend()
                    plt.show();

    std = np.std(RPs)
        
    return RPs, std    

