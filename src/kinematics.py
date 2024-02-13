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


def plot_kinematics(ID, side, angles, trajectories):
    """
    Plot the mean ± std ROM of the hip, knee and ankle angles for the selected subject at self-selected speed in the 3 axis.
    Inputs:
        * ID (string): ID of the participant e.g. "BO2ST_101"
        * side (string): left or right leg: "L" or "R"
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
    """
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    mean_knee_X = []; mean_knee_Y = []; mean_knee_Z = []
    mean_hip_X = []; mean_hip_Y = []; mean_hip_Z = []
    mean_ankle_X = []; mean_ankle_Y = []; mean_ankle_Z = []

    knee_X = []; knee_Y = []; knee_Z = []
    hip_X = []; hip_Y = []; hip_Z = []
    ankle_X = []; ankle_Y = []; ankle_Z = []
    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                    # extract heel trajectory
                    trajectory_heel = trajectories[j][1][ID+":"+side+"HEE"].Y
                    trajectory_toe = trajectories[j][1][ID+":"+side+"TOE"].Y
                    # extract heel_strike and toe_off events
                    heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                    # extract knee, hip and ankle angles
                    knee_angles_X = angles[j][1][ID+":"+side+'KneeAngles'].X
                    knee_angles_Y = angles[j][1][ID+":"+side+'KneeAngles'].Y
                    knee_angles_Z = angles[j][1][ID+":"+side+'KneeAngles'].Z
                    hip_angles_X = angles[j][1][ID+":"+side+'HipAngles'].X
                    hip_angles_Y = angles[j][1][ID+":"+side+'HipAngles'].Y
                    hip_angles_Z = angles[j][1][ID+":"+side+'HipAngles'].Z
                    ankle_angles_X = angles[j][1][ID+":"+side+'AnkleAngles'].X
                    ankle_angles_Y = angles[j][1][ID+":"+side+'AnkleAngles'].Y
                    ankle_angles_Z = angles[j][1][ID+":"+side+'AnkleAngles'].Z
                    # normalize knee, hip and ankle angles
                    knee_X.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_X))
                    knee_Y.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_Y))
                    knee_Z.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_Z))
                    hip_X.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_X))
                    hip_Y.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_Y))
                    hip_Z.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_Z))
                    ankle_X.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_X))
                    ankle_Y.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_Y))
                    ankle_Z.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_Z))

        # plot
        for k in range(int(len(knee_X[i])/250)):
            a = 250*k
            b = 250*(k+1)-1
            mean_knee_X.append(knee_X[i][a:b])
            mean_knee_Y.append(knee_Y[i][a:b])
            mean_knee_Z.append(knee_Z[i][a:b])
            mean_hip_X.append(hip_X[i][a:b])
            mean_hip_Y.append(hip_Y[i][a:b])
            mean_hip_Z.append(hip_Z[i][a:b])
            mean_ankle_X.append(ankle_X[i][a:b])
            mean_ankle_Y.append(ankle_Y[i][a:b])
            mean_ankle_Z.append(ankle_Z[i][a:b])
    
    x = np.linspace(0,100,249)

    fig = make_subplots(rows=3, cols=3, x_title = 'Percentage of gait cycle', y_title = 'ROM [°]', subplot_titles=("Knee Flexion/Extension", "Knee Adduction/Abduction", "Knee Internal/External Rotation", "Hip Flexion/Extension", "Hip Adduction/Abduction", "Hip Internal/External Rotation", "Ankle Flexion/Extension", "Ankle Adduction/Abduction", "Ankle Internal/External Rotation"))
    
    # KNEE
    fig.add_trace(
        go.Scatter(name="Mean Knee X", x=x, y=np.mean(mean_knee_X, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_knee_X, axis=0)+np.std(mean_knee_X, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=1)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_knee_X, axis=0)-np.std(mean_knee_X, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=1) 
    fig.add_hline(y=0, row=1, col=1, line_dash='dash')
    fig.add_vline(x=60, row=1, col=1, line_dash="dash", line_color="red")
    
    fig.add_trace(
        go.Scatter(name="Mean Knee Y", x=x, y=np.mean(mean_knee_Y, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=1, col=2)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_knee_Y, axis=0)+np.std(mean_knee_Y, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=2)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_knee_Y, axis=0)-np.std(mean_knee_Y, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=2)   
    
    fig.add_trace(
        go.Scatter(name="Mean Knee Z", x=x, y=np.mean(mean_knee_Z, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=1, col=3)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_knee_Z, axis=0)+np.std(mean_knee_Z, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=3)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_knee_Z, axis=0)-np.std(mean_knee_Z, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=3)   
    
    # HIP
    fig.add_trace(
        go.Scatter(name="Mean Hip X", x=x, y=np.mean(mean_hip_X, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=2, col=1)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_hip_X, axis=0)+np.std(mean_knee_X, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=2, col=1)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_hip_X, axis=0)-np.std(mean_knee_X, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=2, col=1)  
    fig.add_hline(y=0, row=2, col=1, line_dash='dash')
    fig.add_vline(x=60, row=2, col=1, line_dash="dash", line_color="red")
    
    fig.add_trace(
        go.Scatter(name="Mean Hip Y", x=x, y=np.mean(mean_hip_Y, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=2, col=2)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_hip_Y, axis=0)+np.std(mean_hip_Y, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=2, col=2)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_hip_Y, axis=0)-np.std(mean_hip_Y, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=2, col=2)   
    
    fig.add_trace(
        go.Scatter(name="Mean Hip Z", x=x, y=np.mean(mean_hip_Z, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=2, col=3)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_hip_Z, axis=0)+np.std(mean_hip_Z, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=2, col=3)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_hip_Z, axis=0)-np.std(mean_hip_Z, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=2, col=3)   
    
    # ANKLE
    fig.add_trace(
        go.Scatter(name="Mean Ankle X", x=x, y=np.mean(mean_ankle_X, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=3, col=1)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_ankle_X, axis=0)+np.std(mean_knee_X, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=3, col=1)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_ankle_X, axis=0)-np.std(mean_knee_X, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=3, col=1)   
    fig.add_hline(y=0, row=3, col=1, line_dash='dash')
    fig.add_vline(x=60, row=3, col=1, line_dash="dash", line_color="red")
    
    fig.add_trace(
        go.Scatter(name="Mean Ankle Y", x=x, y=np.mean(mean_ankle_Y, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=3, col=2)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_ankle_Y, axis=0)+np.std(mean_ankle_Y, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=3, col=2)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_ankle_Y, axis=0)-np.std(mean_ankle_Y, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=3, col=2)   
    
    fig.add_trace(
        go.Scatter(name="Mean Ankle Z", x=x, y=np.mean(mean_ankle_Z, axis=0), mode = 'lines', showlegend=False, line=dict(color='black')),
        row=3, col=3)
    fig.add_trace(
        go.Scatter(name="Upper Bound", x=x, y=np.mean(mean_ankle_Z, axis=0)+np.std(mean_ankle_Z, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=3, col=3)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.mean(mean_ankle_Z, axis=0)-np.std(mean_ankle_Z, axis=0),mode='lines',marker=dict(color="#444"), fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=3, col=3)   
    
    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=1000, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.update_traces(connectgaps=True)
    fig.show()
    
    
    
def compute_ROM(trajectories, angles, ID, side, plot=False, new_ss=False, T0=False, T1=False, T2=False):
    """
    Compute the maximal ROM for the knee flexion, hip flexion, and ankle flexion in the sagittal plane for the selected subject.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' or 'R' for the left or right leg
        * plot (boolean): 'True' if you want to plot the joint ROMs
        
    Outputs:
        * mean_ROM_knee (float): mean maximal ROM of the knee angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_hip (float): mean maximal ROM of the hip angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_ankle (float): mean maximal ROM of the ankle angles averaged over all gait cycles of the 3 trials at self-selected speed

    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    knee_X = []; mean_knee_X = []; mean_ROM_knee = []
    hip_X = []; mean_hip_X = []; mean_ROM_hip = []
    ankle_X = []; mean_ankle_X = []; mean_ROM_ankle = []

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
    
    for i in new_ss_index: ## ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                # extract heel and toe trajectories
                trajectory_heel = trajectories[j][1][ID+":"+side+"HEE"].Y
                trajectory_toe = trajectories[j][1][ID+":"+side+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                
                # extract joint angles
                knee_angles_X = angles[j][1][ID+":"+side+'KneeAngles'].X
                hip_angles_X = angles[j][1][ID+":"+side+'HipAngles'].X
                ankle_angles_X = angles[j][1][ID+":"+side+'AnkleAngles'].X
                # normalize joint angles
                knee_X.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_X))
                hip_X.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_X))
                ankle_X.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_X))
                
                # compute ROM for each gait cycle
                for k in range(int(len(knee_X[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_knee_X.append(knee_X[i][a:b])
                    #mean_ROM_knee.append(np.nanmax(knee_X[i][a:b]))
                    mean_ROM_knee.append(np.nanmax(knee_X[i][a:b]) - np.nanmin(knee_X[i][a:b]))
                for k in range(int(len(hip_X[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_hip_X.append(hip_X[i][a:b])  
                    #mean_ROM_hip.append(np.nanmax(hip_X[i][a:b]))
                    mean_ROM_hip.append(np.nanmax(hip_X[i][a:b]) - np.nanmin(hip_X[i][a:b]))
                for k in range(int(len(ankle_X[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_ankle_X.append(ankle_X[i][a:b])   
                    #mean_ROM_ankle.append(np.nanmax(ankle_X[i][a:b]))
                    mean_ROM_ankle.append(np.nanmax(ankle_X[i][a:b]) - np.nanmin(ankle_X[i][a:b]))
                    
   
    # compute mean angles and mean ROMs
    x_ = np.linspace(0,100,249)
    y_knee = np.mean(mean_knee_X, axis=0) 
    mean_ROM_knee = np.mean(mean_ROM_knee)
    y_hip = np.mean(mean_hip_X, axis=0) 
    mean_ROM_hip = np.mean(mean_ROM_hip)
    y_ankle = np.mean(mean_ankle_X, axis=0)
    mean_ROM_ankle = np.mean(mean_ROM_ankle)
    
    # plot
    if (plot):
        fig = make_subplots(rows=1, cols=3, x_title = 'Percentage of gait cycle', y_title = 'ROM [°]', subplot_titles=("Knee Flexion/Extension", "Hip Flexion/Extension", "Ankle Flexion/Extension"))

        # KNEE
        fig.add_trace(
            go.Scatter(name="Mean Knee X", x=x_, y = y_knee, mode = 'lines', showlegend=False, line=dict(color='black')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(name="Mean Hip X", x=x_, y = y_hip, mode = 'lines', showlegend=False, line=dict(color='black')),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(name="Mean Ankle X", x=x_, y = y_ankle, mode = 'lines', showlegend=False, line=dict(color='black')),
            row=1, col=3)
        
        # fig params
        fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=400, width=1000, showlegend=False)
        fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
        fig.update_traces(connectgaps=True)
        fig.show()

    return mean_ROM_knee, mean_ROM_hip, mean_ROM_ankle


    
def compute_ROM_AB(trajectories_AB, angles_AB, number_AB, side):
    """
    Compute the maximal ROM for the knee flexion, hip flexion, and ankle flexion in the sagittal plane for the selected subject.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' or 'R' for the left or right leg
        * plot (boolean): 'True' if you want to plot the joint ROMs
        
    Outputs:
        * mean_ROM_knee (float): mean maximal ROM of the knee angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_hip (float): mean maximal ROM of the hip angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_ankle (float): mean maximal ROM of the ankle angles averaged over all gait cycles of the 3 trials at self-selected speed

    """
    
    ROM_knee = []; ROM_hip = []; ROM_ankle = []
    
    # AB
    for i in range(len(angles_AB)):
        # ss speed
        if (angles_AB[i][0] == 'WBDS'+number_AB+'walkT05ang'):
            # extract left and right knee angles
            knee_angles_AB = angles_AB[i][1][side+'KneeAngleZ']
            hip_angle_AB = angles_AB[i][1][side+'HipAngleZ']
            ankle_angle_AB = angles_AB[i][1][side+'AnkleAngleZ']
    
    # center angles
    knee_angles_AB = knee_angles_AB - np.nanmin(knee_angles_AB) - (np.nanmax(knee_angles_AB) - np.nanmin(knee_angles_AB))/2
    hip_angle_AB = hip_angle_AB - np.nanmin(hip_angle_AB) - (np.nanmax(hip_angle_AB) - np.nanmin(hip_angle_AB))/2
    ankle_angle_AB = ankle_angle_AB- np.nanmin(ankle_angle_AB) - (np.nanmax(ankle_angle_AB) - np.nanmin(ankle_angle_AB))/2

    # compute ROM
    ROM_knee = (np.nanmax(knee_angles_AB) - np.nanmin(knee_angles_AB))    
    ROM_hip = (np.nanmax(hip_angle_AB) - np.nanmin(hip_angle_AB))    
    ROM_ankle = (np.nanmax(ankle_angle_AB) - np.nanmin(ankle_angle_AB))    
    
    return ROM_knee, ROM_hip, ROM_ankle
   
    
def compute_ROM_left_and_right(trajectories, angles, ID, plot=False, new_ss=False, T0=False, T1=False, T2=False):
    """
    Compute the maximal ROM for the knee flexion, hip flexion, and ankle flexion in the sagittal plane for the selected subject.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' or 'R' for the left or right leg
        * plot (boolean): 'True' if you want to plot the joint ROMs
        
    Outputs:
        * mean_ROM_knee (float): mean maximal ROM of the knee angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_hip (float): mean maximal ROM of the hip angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_ankle (float): mean maximal ROM of the ankle angles averaged over all gait cycles of the 3 trials at self-selected speed

    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    knee_X_left = []; mean_knee_X_left = []; mean_ROM_knee_left = []
    hip_X_left = []; mean_hip_X_left = []; mean_ROM_hip_left = []
    ankle_X_left = []; mean_ankle_X_left = []; mean_ROM_ankle_left = []
    
    knee_X_right = []; mean_knee_X_right = []; mean_ROM_knee_right = []
    hip_X_right = []; mean_hip_X_right = []; mean_ROM_hip_right = []
    ankle_X_right = []; mean_ankle_X_right = []; mean_ROM_ankle_right = []
    
    
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

    for i in new_ss_index: ## ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                # extract heel and toe trajectories
                trajectory_heel = trajectories[j][1][ID+":"+'L'+"HEE"].Y
                trajectory_toe = trajectories[j][1][ID+":"+'L'+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                
                # extract joint angles
                knee_angles_X = angles[j][1][ID+":"+'L'+'KneeAngles'].X
                hip_angles_X = angles[j][1][ID+":"+'L'+'HipAngles'].X
                ankle_angles_X = angles[j][1][ID+":"+'L'+'AnkleAngles'].X
                # normalize joint angles
                knee_X_left.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_X))
                hip_X_left.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_X))
                ankle_X_left.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_X))
                
                # compute ROM for each gait cycle
                for k in range(int(len(knee_X_left[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_knee_X_left.append(knee_X_left[i][a:b])
                    #mean_ROM_knee.append(np.nanmax(knee_X[i][a:b]))
                    mean_ROM_knee_left.append(np.nanmax(knee_X_left[i][a:b]) - np.nanmin(knee_X_left[i][a:b]))
                for k in range(int(len(hip_X_left[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_hip_X_left.append(hip_X_left[i][a:b])  
                    #mean_ROM_hip.append(np.nanmax(hip_X[i][a:b]))
                    mean_ROM_hip_left.append(np.nanmax(hip_X_left[i][a:b]) - np.nanmin(hip_X_left[i][a:b]))
                for k in range(int(len(ankle_X_left[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_ankle_X_left.append(ankle_X_left[i][a:b])   
                    #mean_ROM_ankle.append(np.nanmax(ankle_X[i][a:b]))
                    mean_ROM_ankle_left.append(np.nanmax(ankle_X_left[i][a:b]) - np.nanmin(ankle_X_left[i][a:b]))
                    
   
    # compute mean angles and mean ROMs
    x_ = np.linspace(0,100,249)
    y_knee_left = np.mean(mean_knee_X_left, axis=0) 
    mean_ROM_knee_left = np.mean(mean_ROM_knee_left)
    y_hip_left = np.mean(mean_hip_X_left, axis=0) 
    mean_ROM_hip_left = np.mean(mean_ROM_hip_left)
    y_ankle_left = np.mean(mean_ankle_X_left, axis=0)
    mean_ROM_ankle_left = np.mean(mean_ROM_ankle_left)
    
    
    for i in new_ss_index: ## ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                # extract heel and toe trajectories
                trajectory_heel = trajectories[j][1][ID+":"+'R'+"HEE"].Y
                trajectory_toe = trajectories[j][1][ID+":"+'R'+"TOE"].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)
                
                # extract joint angles
                knee_angles_X = angles[j][1][ID+":"+'R'+'KneeAngles'].X
                hip_angles_X = angles[j][1][ID+":"+'R'+'HipAngles'].X
                ankle_angles_X = angles[j][1][ID+":"+'R'+'AnkleAngles'].X
                # normalize joint angles
                knee_X_right.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_X))
                hip_X_right.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_X))
                ankle_X_right.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_X))
                
                # compute ROM for each gait cycle
                for k in range(int(len(knee_X_right[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_knee_X_right.append(knee_X_right[i][a:b])
                    #mean_ROM_knee.append(np.nanmax(knee_X[i][a:b]))
                    mean_ROM_knee_right.append(np.nanmax(knee_X_right[i][a:b]) - np.nanmin(knee_X_right[i][a:b]))
                for k in range(int(len(hip_X_right[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_hip_X_right.append(hip_X_right[i][a:b])  
                    #mean_ROM_hip.append(np.nanmax(hip_X[i][a:b]))
                    mean_ROM_hip_right.append(np.nanmax(hip_X_right[i][a:b]) - np.nanmin(hip_X_right[i][a:b]))
                for k in range(int(len(ankle_X_right[i])/250)):
                    a = 250*k
                    b = 250*(k+1)-1
                    mean_ankle_X_right.append(ankle_X_right[i][a:b])   
                    #mean_ROM_ankle.append(np.nanmax(ankle_X[i][a:b]))
                    mean_ROM_ankle_right.append(np.nanmax(ankle_X_right[i][a:b]) - np.nanmin(ankle_X_right[i][a:b]))
    
    # compute mean angles and mean ROMs
    x_ = np.linspace(0,100,249)
    y_knee_right = np.mean(mean_knee_X_right, axis=0) 
    mean_ROM_knee_right = np.mean(mean_ROM_knee_right)
    y_hip_right = np.mean(mean_hip_X_right, axis=0) 
    mean_ROM_hip_right = np.mean(mean_ROM_hip_right)
    y_ankle_right = np.mean(mean_ankle_X_right, axis=0)
    mean_ROM_ankle_right = np.mean(mean_ROM_ankle_right)
    
    
    # plot
    if (plot):
        fig = make_subplots(rows=1, cols=3, x_title = 'Percentage of gait cycle', y_title = 'ROM [°]', subplot_titles=("Knee Flexion/Extension", "Hip Flexion/Extension", "Ankle Flexion/Extension"))

        # KNEE
        fig.add_trace(
            go.Scatter(name="Mean Knee X left", x=x_, y = y_knee_left, mode = 'lines', showlegend=False, line=dict(color='black')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(name="Mean Knee X right", x=x_, y = y_knee_right, mode = 'lines', showlegend=False, line=dict(color='red')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(name="Mean Hip X left", x=x_, y = y_hip_left, mode = 'lines', showlegend=False, line=dict(color='black')),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(name="Mean Hip X right", x=x_, y = y_hip_right, mode = 'lines', showlegend=False, line=dict(color='red')),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(name="Mean Ankle X left", x=x_, y = y_ankle_left, mode = 'lines', showlegend=False, line=dict(color='black')),
            row=1, col=3)
        fig.add_trace(
            go.Scatter(name="Mean Ankle X right", x=x_, y = y_ankle_right, mode = 'lines', showlegend=False, line=dict(color='red')),
            row=1, col=3)
        
        # fig params
        fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=400, width=1000, showlegend=False)
        fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
        fig.update_traces(connectgaps=True)
        fig.show()

    return mean_ROM_knee_left, mean_ROM_knee_right, mean_ROM_hip_left, mean_ROM_hip_right, mean_ROM_ankle_left, mean_ROM_ankle_right


def compute_ROM_time_evolution(traj, ang, ID, side, ang_AB, number_AB, plot=False):
    """
    Compute the maximal ROM for the knee flexion, hip flexion, and ankle flexion in the sagittal plane for the selected subject.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' or 'R' for the left or right leg
        * plot (boolean): 'True' if you want to plot the joint ROMs
        
    Outputs:
        * mean_ROM_knee (float): mean maximal ROM of the knee angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_hip (float): mean maximal ROM of the hip angles averaged over all gait cycles of the 3 trials at self-selected speed
        * mean_ROM_ankle (float): mean maximal ROM of the ankle angles averaged over all gait cycles of the 3 trials at self-selected speed

    """

    # plot
    if (plot):
        fig = make_subplots(rows=1, cols=3, x_title = 'Percentage of gait cycle', y_title = 'ROM [°]', subplot_titles=("Knee Flexion/Extension", "Hip Flexion/Extension", "Ankle Flexion/Extension"))
        
    for p, _ in enumerate(traj):
        
        trajectories = traj[p]
        angles = ang[p]
        
        # recover and sort the trials in ascending order
        list_ = []
        for i in range(len(trajectories)):
            list_.append(trajectories[i][0])
        list_sorted = natsort.natsorted(list_,reverse=False)

        knee_X = []; mean_knee_X = []; mean_ROM_knee = []
        hip_X = []; mean_hip_X = []; mean_ROM_hip = []
        ankle_X = []; mean_ankle_X = []; mean_ROM_ankle = []

        if (ID=='BO2ST_101' or ID=='BO2ST_102' or ID=='BO2ST_105'):
            new_ss_index = [0,1]
        else:  
            new_ss_index = [0,1,2]
        
        for i in new_ss_index: ## ss speed
            for j in range(len(list_sorted)):
                # sort the trials
                if (trajectories[j][0] == list_sorted[i]):
                    # extract heel and toe trajectories
                    trajectory_heel = trajectories[j][1][ID+":"+side+"HEE"].Y
                    trajectory_toe = trajectories[j][1][ID+":"+side+"TOE"].Y
                    # extract heel_strike and toe_off events
                    heel_strike, toe_off = compute_gait_cycle_2(trajectory_heel, trajectory_toe)

                    # extract joint angles
                    knee_angles_X = angles[j][1][ID+":"+side+'KneeAngles'].X
                    hip_angles_X = angles[j][1][ID+":"+side+'HipAngles'].X
                    ankle_angles_X = angles[j][1][ID+":"+side+'AnkleAngles'].X
                    # normalize joint angles
                    knee_X.append(normalize_gait_cycle(heel_strike, toe_off, knee_angles_X))
                    hip_X.append(normalize_gait_cycle(heel_strike, toe_off, hip_angles_X))
                    ankle_X.append(normalize_gait_cycle(heel_strike, toe_off, ankle_angles_X))

                    # compute ROM for each gait cycle
                    for k in range(int(len(knee_X[i])/250)):
                        a = 250*k
                        b = 250*(k+1)-1
                        mean_knee_X.append(knee_X[i][a:b])
                        #mean_ROM_knee.append(np.nanmax(knee_X[i][a:b]))
                        mean_ROM_knee.append(np.nanmax(knee_X[i][a:b]) - np.nanmin(knee_X[i][a:b]))
                    for k in range(int(len(hip_X[i])/250)):
                        a = 250*k
                        b = 250*(k+1)-1
                        mean_hip_X.append(hip_X[i][a:b])  
                        #mean_ROM_hip.append(np.nanmax(hip_X[i][a:b]))
                        mean_ROM_hip.append(np.nanmax(hip_X[i][a:b]) - np.nanmin(hip_X[i][a:b]))
                    for k in range(int(len(ankle_X[i])/250)):
                        a = 250*k
                        b = 250*(k+1)-1
                        mean_ankle_X.append(ankle_X[i][a:b])   
                        #mean_ROM_ankle.append(np.nanmax(ankle_X[i][a:b]))
                        mean_ROM_ankle.append(np.nanmax(ankle_X[i][a:b]) - np.nanmin(ankle_X[i][a:b]))


        # compute mean angles and mean ROMs
        x_ = np.linspace(0,100,101)
        y_knee = np.mean(mean_knee_X, axis=0) 
        y_knee = y_knee - np.nanmin(y_knee) - (np.nanmax(y_knee) - np.nanmin(y_knee))/2
        mean_ROM_knee = np.mean(mean_ROM_knee)
        y_hip = np.mean(mean_hip_X, axis=0) 
        y_hip = y_hip - np.nanmin(y_hip) - (np.nanmax(y_hip) - np.nanmin(y_hip))/2
        mean_ROM_hip = np.mean(mean_ROM_hip)
        y_ankle = np.mean(mean_ankle_X, axis=0)
        y_ankle = y_ankle - np.nanmin(y_ankle) - (np.nanmax(y_ankle) - np.nanmin(y_ankle))/2
        mean_ROM_ankle = np.mean(mean_ROM_ankle)

        # resample to match AB sampling frequency
        y_knee = resample_by_interpolation(y_knee, 249, 101)
        y_hip = resample_by_interpolation(y_hip, 249, 101)
        y_ankle = resample_by_interpolation(y_ankle, 249, 101)

        # plot
        if (plot):
            # KNEE
            fig.add_trace(
                go.Scatter(name="Mean Knee X "+str(p), x=x_, y = y_knee, mode = 'lines', showlegend=False, line=dict(color='rgb('+str(128-p*40)+','+str(128-p*40)+','+str(128-p*40)+')')),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(name="Mean Hip X "+str(p), x=x_, y = y_hip, mode = 'lines', showlegend=False, line=dict(color='rgb('+str(128-p*40)+','+str(128-p*40)+','+str(128-p*40)+')')),
                row=1, col=2)
            fig.add_trace(
                go.Scatter(name="Mean Ankle X "+str(p), x=x_, y = y_ankle, mode = 'lines', showlegend=False, line=dict(color='rgb('+str(128-p*40)+','+str(128-p*40)+','+str(128-p*40)+')')),
                row=1, col=3)

    
    # AB
    for i in range(len(ang_AB)):
        # ss speed
        if (ang_AB[i][0] == 'WBDS'+number_AB+'walkT05ang'):
            # extract left and right knee angles
            knee_angles_AB = ang_AB[i][1][side+'KneeAngleZ']
            hip_angle_AB = ang_AB[i][1][side+'HipAngleZ']
            ankle_angle_AB = ang_AB[i][1][side+'AnkleAngleZ']
    
    # center angles
    knee_angles_AB = knee_angles_AB - np.nanmin(knee_angles_AB) - (np.nanmax(knee_angles_AB) - np.nanmin(knee_angles_AB))/2
    hip_angle_AB = hip_angle_AB - np.nanmin(hip_angle_AB) - (np.nanmax(hip_angle_AB) - np.nanmin(hip_angle_AB))/2
    ankle_angle_AB = ankle_angle_AB- np.nanmin(ankle_angle_AB) - (np.nanmax(ankle_angle_AB) - np.nanmin(ankle_angle_AB))/2
    
    if (plot):
        # KNEE
        fig.add_trace(
            go.Scatter(name="Mean Knee X "+str(p), x=x_, y = knee_angles_AB, mode = 'lines', showlegend=False, line = dict(color='royalblue', dash='dash')),
            row=1, col=1)
        fig.add_trace(
            go.Scatter(name="Mean Hip X "+str(p), x=x_, y = hip_angle_AB, mode = 'lines', showlegend=False, line = dict(color='royalblue', dash='dash')),
            row=1, col=2)
        fig.add_trace(
            go.Scatter(name="Mean Ankle X "+str(p), x=x_, y = ankle_angle_AB, mode = 'lines', showlegend=False, line = dict(color='royalblue', dash='dash')),
            row=1, col=3)
    
    
    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=400, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.update_traces(connectgaps=True)
    fig.show()



    return mean_ROM_knee, mean_ROM_hip, mean_ROM_ankle

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def compute_symmetry_index(Xr, Xl):
    """ 
    Compute the symmetry index (SI) according to Robinson et al.
    A value of zero for SI indicates that there is no difference between the variables Xr and Xl, and therefore
    that there is perfect gait symmetry. 
    A positive value for SI indicates the the magnitude of Xr is larger than that of Xl.
    A negative value for SI indicates the the magnitude of Xr is smaller than that of Xl.

    Inputs:
        * Xr (float): gait variable recorded for the right leg (mean variable over the trial)
        * Xl (float): gait variable recorded for the left leg (mean variable over the trial)
    Output:
        * SI (float): symmetry index
        
    """
    SI = np.abs((Xr-Xl)/ (1/2*(Xr+Xl)) * 100)
    return SI  
