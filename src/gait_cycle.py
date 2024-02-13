import numpy as np
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
import natsort
import math
import pathlib

def compute_gait_cycle_2(trajectory_heel, trajectory_toe): 
    """
    Compute gait cycles using a velocity-based treadmill algorithm with the heel marker defined as the motion reference. Heel-strike (HS) events are approximated using a downward zero-crossing point of the fore-aft HEEL velocity. Toe-off (TO) corresponds to an upward zero-crossing point of fore-aft TOE velocity. Swing phase is defined as the time between TO to HS. 

    Inputs:
        * trajectory_heel (list): trajectory of the selected heel marker. We usually use the heel marker to compute the gait cycle heel-strike events using the velocity-based treadmill algorithm. 
        * trajectory_toe (list): trajectory of the selected toe marker. We usually use the toe marker to compute the gait cycle toe-off events using the velocity-based treadmill algorithm.  
        
    Outputs:
        * heel_strike (list): time points of the heel strike events
        * toe_off (list): time point of the toe off events
    """
    
    # compute velocity of the heel marker
    t_heel = np.linspace(0,len(trajectory_heel),len(trajectory_heel))
    dt_heel = t_heel[1]-t_heel[0]
    y_heel = trajectory_heel.values.squeeze()
    dydt_heel = np.gradient(y_heel, dt_heel)
    # zero crossing
    heel_strike = []
    for i in range(len(dydt_heel)-1):
        # downward crossing
        if( dydt_heel[i]>=0 and dydt_heel[i+1]<0):
            heel_strike.append(i)
    # compute velocity of the toe marker        
    t_toe = np.linspace(0,len(trajectory_toe),len(trajectory_toe))
    dt_toe = t_toe[1]-t_toe[0]
    y_toe = trajectory_toe.values.squeeze()
    dydt_toe = np.gradient(y_toe, dt_toe)
    # zero crossing
    toe_off = []  
    for i in range(len(dydt_toe)-3):
        # upward crossing
        if (dydt_toe[i]<=0 and dydt_toe[i+1]>0 and dydt_toe[i+2]>0 and dydt_toe[i+3]>0):
            toe_off.append(i) 
    
    return heel_strike, toe_off

def normalize_gait_cycle(heel_strike, toe_off, trajectory, safety_check=False):
    """
    Normalize each gait cycle in the selected trajectory to 250 frames: 150 frames for the stance phase (60% of the gait cycle) and 100 frames for the swing phase (40% of the gait cycle). Each normalized trajectory starts with the stance phase. 
    (Litterature Ref: The influence of increasing steady-state walking speed on muscle activity in below-knee amputees, Fey et al.
    Speed related changes in muscle activity from normal to very slow walking speeds, A.R den Otter et al.)

    Inputs:
        * heel_strike (list): time points of the heel strike events 
        * toe_off (list): time point of the toe off events
        * trajectory (list): angle trajectory of the selected marker
        * safety_check (bool): if True, print the total number of time frames and the total number of gait cycles. The printed length of the list should be the printed number of gait cycles times 250. 
        
    Outputs:
        * trajectory_normalized (list): normalized trajectory of the marker.
    """
    
    trajectory_normalized = []
    
    min_ = min(len(heel_strike), len(toe_off))
    for i in range(min_-1):
        a = heel_strike[i]
        b = toe_off[i]
        if (b<a):
            b = toe_off[i+1]
        c = heel_strike[i+1]
        # cut stance and swing phases
        stance = trajectory.reset_index(drop=True)[a:b]
        swing = trajectory.reset_index(drop=True)[b:c]
        # normalize stance phase from 0 to 149 (150 frames)
        stance['percent'] = (np.arange(len(stance))+1)/len(stance)*150
        stance.set_index('percent',inplace =True)
        stanceresampled = np.linspace(0,149,150)
        stance_normalized = stance.reindex(stance.index.union(stanceresampled)).interpolate('values').loc[stanceresampled]
        # normalize swing phase from 0 to 99 (100 frames)
        swing['percent'] = (np.arange(len(swing))+1)/len(swing)*100
        swing.set_index('percent',inplace =True)
        swingresampled = np.linspace(0,99,100)
        swing_normalized = swing.reindex(swing.index.union(swingresampled)).interpolate('values').loc[swingresampled]
        # append to array
        for i in range(150):
            trajectory_normalized.append(stance_normalized.deg.values.squeeze()[i])
        for i in range(100):
            trajectory_normalized.append(swing_normalized.deg.values.squeeze()[i])
    
    if (safety_check):
        # safety check
        print('total number of time frames: ' + str(len(trajectory_normalized)))
        print('total number of gait cycles: ' + str(len(trajectory_normalized)/250))
    
    return trajectory_normalized   

       
  
    