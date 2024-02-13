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
#from src.step_variability import*
from src.gait_cycle import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compute_ACC(trajectory_normalized_joint1, trajectory_normalized_joint2):
    """
    Compute the angular component of the coefficient of correspondence (ACC). If the relative motion between joint1 and joint2 is in perfect agreement over multiple cycles, then the ACC = 1, indicating maximal consistency between gait cycles.

    Inputs:
        * trajectory_normalized_joint1 (list): normalized trajectory of the joint1. Each gait cycle in the trajectory should be normalized to 150 frames: 100 frames for the stance phase and 50 frames for the swing phase.
        * trajectory_normalized_joint1 (list): normalized trajectory of the joint2. 
        
    Outputs:
        * ACC (float): angular component of the coefficient of correspondence
    """
    
    # compute the angular direction of the line segment between 2 consecutive frames for each frame-to-frame interval within each cycle
    angular_direction_l = []
    for i in range(len(trajectory_normalized_joint1)-1):
        l = np.sqrt( (trajectory_normalized_joint1[i] - trajectory_normalized_joint1[i+1])**2 +
                   (trajectory_normalized_joint2[i] - trajectory_normalized_joint2[i+1])**2)
        angular_direction_l.append(l)
    
    # compute the cosine and sine of the angular direction for each frame-to-frame interval within each cycle
    sin_l = []
    for i in range(len(trajectory_normalized_joint2)-1):
        sin = (trajectory_normalized_joint2[i] - trajectory_normalized_joint2[i+1])/angular_direction_l[i]
        sin_l.append(sin)

    cos_l = []
    for i in range(len(trajectory_normalized_joint1)-1):
        cos = (trajectory_normalized_joint1[i] - trajectory_normalized_joint1[i+1])/angular_direction_l[i]
        cos_l.append(cos)     
    
    # compute the mean cosine and the mean sine for a given frame-to-frame interval over multiple cycles (eg, frame 1–2 of cycles 1– 6)
    mean_sin_l = []
    mean_cos_l = []

    for i in range(int((len(cos_l)+1)/250)):
        #print(str(249*i+i) + ':' + str(249*(i+1)+i))
        mean_cos_l.append(cos_l[249*i+i:249*(i+1)+i])
    mean_cos_l = np.mean(mean_cos_l, axis=0)  
    
    for i in range(int((len(sin_l)+1)/250)):
        mean_sin_l.append(sin_l[249*i+i:249*(i+1)+i])
    mean_sin_l = np.mean(mean_sin_l, axis=0) 
    
    # compute the mean vector length for each frame-to-frame interval within a cycle. The length of the mean vector denotes the degree of dispersion of the joint1-joint2 values about the mean over multiple cycles for that particular frame. The larger the value (between 0 and 1), the less variable (i.e., less randomly distributed, more consistent) is the joint1-joint2 relationship.
    alpha_l = []
    for i in range(len(mean_sin_l)):
        alpha = np.sqrt(mean_cos_l[i]**2 + mean_sin_l[i]**2)
        alpha_l.append(alpha)
    alpha_l = [x for x in alpha_l if str(x) != 'nan']
    #print(alpha_l)

    # compute the ACC (i.e., the arithmetic average of all the mean vector lengths) which indicates the overall variability of the joint1-joint2 relationship for all cycles.
    ACC = np.mean(alpha_l)
    
    return ACC


def compute_mean_std_ACC(trajectories, angles, ID, side, new_ss=False, T0=False, T1=False, T2=False):
    """
    Compute the mean and the standard deviation of the ACCs for the self-selected speed. 

    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the joint angles of all the 9 trials for the selected subject, as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): "L" for left or "R" for right side
        * ss_speed (float): self-selected speed of the subject
        * plot (bool): True if you want to plot the ACC versus speed 
        
    Outputs:
        * means (list): return the mean ACC for the self-selected speed, fast speed and slow speed
        * std (list): return the std of the ACC for the self-selected speed, fast speed and slow speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    ACCs = []
       
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
                trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y
                trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_HEE_Y, trajectory_TOE_Y)
                if (ID=="BO2ST_109" and T1 ==True):
                    heel_strike.pop(2)
                # extract knee and hip angles
                knee_angles = angles[j][1][ID+":"+side+'KneeAngles'].X
                hip_angles = angles[j][1][ID+":"+side+'HipAngles'].X
                # normalize knee and hip angles
                knee_angles_normalized = normalize_gait_cycle(heel_strike, toe_off, knee_angles)
                hip_angles_normalized = normalize_gait_cycle(heel_strike, toe_off, hip_angles)
                # compute ACC
                ACCs.append(compute_ACC(knee_angles_normalized, hip_angles_normalized))

    #print(ACCs)
    mean = np.nanmean(ACCs)
    std = np.nanstd(ACCs)
    
    
    return mean, std


def individual_progress_ACC(ID, side):
    """
    Plot the progress of the participant in terms of ACC according to time.
    Inputs:
        * ID (string): ID of the participant, e.g. "BO2ST_101"
        * side (string): most affected side of the participant, 'L' or 'R'
    """
    
    # read xcl files
    ACCs_file_name = ("BO2STTrial/ACCs.xlsx")
    ACCs_xcl = pd.read_excel(ACCs_file_name, header = [0], index_col = [0])
    
    # matched subject files
    #matched_subjects = pd.read_excel("BO2STTrial/matched_subjects.xlsx", index_col=0)
    #matched_control = matched_subjects.loc[ID]['Matched_control'] # find the matched healthy control to the SCI participant
    #walking_mechanics_performances_control_file_name = ("BO2STTrial/walking_mechanics_performances_control.xlsx") # find the walking mechanics of the matched control
    #walking_mechanics_performances_control_xcl = pd.read_excel(walking_mechanics_performances_control_file_name, header = [0], index_col = [0])
        
    x = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8']
    ACC_y_L = ACCs_xcl.loc[ID][['ACC_L_BL', 'ACC_L_T0', 'ACC_L_T1', 'ACC_L_T2', 'ACC_L_F1', 'ACC_L_F4', 'ACC_L_F8']] # relative to T0?
    ACC_y_R = ACCs_xcl.loc[ID][['ACC_R_BL', 'ACC_R_T0', 'ACC_R_T1', 'ACC_R_T2', 'ACC_R_F1', 'ACC_R_F4', 'ACC_R_F8']] # relative to T0?
    
    #EV_AB = walking_mechanics_performances_control_xcl.loc[matched_control][['EV_BL', 'EV_T0', 'EV_T1', 'EV_T2', 'EV_F1', 'EV_F4', 'EV_F8']]
    #A_AB = walking_mechanics_performances_control_xcl.loc[matched_control][['A_BL', 'A_T0', 'A_T1', 'A_T2', 'A_F1', 'A_F4', 'A_F8']]
    
    if (side=='L'):
        weak_ACC = ACC_y_L
        strong_ACC = ACC_y_R
    elif (side=='R'):
        weak_ACC = ACC_y_R
        strong_ACC = ACC_y_L
    else:
        print('ERR: side should be L or R')
        return
    

    fig = make_subplots(rows=1, cols=1, x_title = 'Time', horizontal_spacing=0.115)

    # ACC less affected leg (strong leg)
    fig.add_trace(
        go.Scatter(name='LA side', x=x, y=strong_ACC, line=dict(color='black')),
        row=1, col=1)
    # ACC most affected leg (weak leg)
    fig.add_trace(
        go.Scatter(name='MA side', x=x, y=weak_ACC, line=dict(color='red')),
        row=1, col=1)


    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=450, width=600, showlegend=True)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    #fig.add_hline(y=0, line_color='black', line_dash="dash")
    fig['layout']['xaxis1'].update(
        tickmode = 'array',
        tickvals = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'],
        ticktext = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8']
    )
    fig.update_traces(connectgaps=True)
    # Update yaxis properties
    fig.update_yaxes(title_text="ACC", range=[0.7,1], row=1, col=1)
    fig.show()
    

