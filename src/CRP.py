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


def display_frequencies(signal, Fs=250, xlim=None):
    """
    Plot the frequencies of the signal obtained by FFT.
    
    Inputs:
        * signal (array): signal you want to obtain the frequencies
        * Fs (float): sampling frequency of the signal
    """
    plt.figure(figsize=(20,8))
    plt.magnitude_spectrum(signal, Fs=Fs, color='C1')
    if (xlim!=None):
        plt.xlim(0,xlim)
    

def compute_CRP_ss_speed_threetrials(angles, ID, crp, LA_side, MA_side, MA = True, plot=False):
    """
    Compute the continuous relative phase for the three trials at the self-selected speed. 
    
    Inputs:
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * crp (string): "IPSILATERAL" for ipsilateral shoulder-hip pairs, "CONTRALATERAL" for contralateral shoulder-hip pairs, "BILATERAL_HIP" OR "BILATERAL_SHOULDER" for bilateral shoulder-shoulder and hip-hip pairs
        * LA_side (string): less affected side 'L' for left and 'R' for right
        * MA_side (string): most affected side 'L' for left and 'R' for right
        * MA (bool): True if you want the crp for the MA side
        * plot (bool): if True plot the CRP
    Outputs:
        * mean_CRPs (float): mean CRP averaged over the three trials at the self-selected speed. 
        * std_CRPs (float): standard deviation of the CRP over the three trials at the self-selected speed. 
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(angles)):
        list_.append(angles[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    CRPs = []
    fig, ax = plt.subplots(1,1,figsize=(20,8))
    
    if (MA):
        side = MA_side
        contra_side = LA_side
    else:
        side = LA_side
        contra_side = MA_side

    for i in [0,1,2]: # ss speed
        for j in range(len(list_sorted)):
            # sort the trials
            if (angles[j][0] == list_sorted[i]):
                
                if (crp=="IPSILATERAL"):
                    # recover hip and shoulder angle time series
                    angle_1 = angles[j][1][ID+':' + side + 'HipAngles'].X
                    angle_2 = angles[j][1][ID+':' + side + 'ShoulderAngles'].X
                    
                if (crp=="CONTRALATERAL"):
                    # recover hip and shoulder angle time series
                    angle_1 = angles[j][1][ID+':' + side + 'HipAngles'].X
                    angle_2 = angles[j][1][ID+':' + contra_side + 'ShoulderAngles'].X   
                 
                if (crp=="BILATERAL_HIP"):
                    # recover hip and hip angle time series
                    angle_1 = angles[j][1][ID+':' + side + 'HipAngles'].X
                    angle_2 = angles[j][1][ID+':' + contra_side + 'HipAngles'].X
                    
                if (crp=="BILATERAL_SHOULDER"):
                    # recover shoulder and shoulder angle time series
                    angle_1 = angles[j][1][ID+':' + side + 'ShoulderAngles'].X
                    angle_2 = angles[j][1][ID+':' + contra_side + 'ShoulderAngles'].X   
                
                # center angle time series
                angle_2_centered = angle_2 - np.nanmin(angle_2) - (np.nanmax(angle_2) - np.nanmin(angle_2))/2
                angle_1_centered = angle_1 - np.nanmin(angle_1) - (np.nanmax(angle_1) - np.nanmin(angle_1))/2
                
                # remove NA values
                min_ = max(angle_1_centered.first_valid_index(), angle_2_centered.first_valid_index())
                max_ = min(angle_1_centered.last_valid_index(), angle_2_centered.last_valid_index())
                angle_1_centered = angle_1_centered.iloc[min_:max_].deg.values
                angle_2_centered = angle_2_centered.iloc[min_:max_].deg.values
                #if (len(angle_1_centered.dropna().deg.values) < len(angle_2_centered.dropna().deg.values)):
                 #   angle_2_centered = angle_2_centered.iloc[angle_1_centered.dropna().index[0]:angle_1_centered.dropna().index[-1]+1].deg.values
                #    angle_1_centered = angle_1_centered.dropna().deg.values
                #else:
                 #   angle_1_centered = angle_1_centered.iloc[angle_2_centered.dropna().index[0]:angle_2_centered.dropna().index[-1]+1].deg.values
                 #   angle_2_centered = angle_2_centered.dropna().deg.values
                
                # filter signals 
                b,a = signal.butter(4, 12, 'lowpass',fs=250)
                filtered_angle_1 = signal.filtfilt(b, a, angle_1_centered, padlen=150)
                filtered_angle_2 = signal.filtfilt(b, a, angle_2_centered, padlen=150)

                # compute phase angles
                phase_2 = np.angle(hilbert(filtered_angle_2), deg=True)
                phase_1 = np.angle(hilbert(filtered_angle_1), deg=True)
                # compute continuous relative phase 
                crp = phase_1 - phase_2
                crp = np.rad2deg(np.unwrap(np.radians(crp)))
                
                # add the crp to the list
                for k in range(len(crp)):
                    CRPs.append(crp[k])
                    
                
    # compute mean and std of the crp over the three trials
    CRPs = [x%360 for x in CRPs] # 0 <= x <= 360
    CRPs = [(x + 360) % 360 for x in CRPs] # 0 <= x < 360
    CRPs = [normalizeAngle(x) for x in CRPs] # -180 <= x < 180
    
    mean_CRPs = np.nanmean((CRPs))
    std_CRPs = np.nanstd((CRPs))

    if (plot):
        plt.plot(CRPs, color='grey')
        plt.axhline(y=mean_CRPs, color='red', linestyle = 'dotted')
        # figure parameters    
        plt.xlabel('Time'); plt.ylabel('CRP [°]')
        ax.spines[['right', 'top']].set_visible(False)
        #ax.set_ylim(-180,180)
        plt.show()
        plt.tight_layout(); 

    plt.close()
    
    return mean_CRPs, std_CRPs


def normalizeAngle(angle):

    if (angle > 180):
        angle -=360
        
    return angle


def compute_CRP_ss_plot(angles_BL, angles_T0, IDs, crp, LA_sides, MA_sides, MA=True):
    
    fig, ax = plt.subplots(1,figsize= (7,5))

    for i in range(len(IDs)):
        mean_CRP_BL, std_CRP_BL = compute_CRP_ss_speed_threetrials(angles_BL[i], IDs[i], crp, LA_sides[i], MA_sides[i], MA, plot=False)
        mean_CRP_T0, std_CRP_T0 = compute_CRP_ss_speed_threetrials(angles_T0[i], IDs[i], crp, LA_sides[i], MA_sides[i], MA, plot=False)
        data = [mean_CRP_BL, mean_CRP_T0]
        e = [std_CRP_BL, std_CRP_T0]
        ax.plot(["BL","T0"],data, label=IDs[i], color="C"+str(i))
        ax.plot(["BL","T0"],data, 'o',color="C"+str(i))
        #ax.errorbar(["BL","T0"], data, e, marker='o', label=IDs[i])
    ax.legend(title = "Participants")
    ax.spines[['right', 'top']].set_visible(False)

    # figure parameters
    fig.show()
    fig.tight_layout();
    
def compute_CRP_polar_plot(angles_BL, angles_T0, IDs, crp, title, LA_sides, MA_sides, MA=True, legend=False):  
    
    df_CRP = pd.DataFrame(columns=['ID', 'CRP', 'Time', 'R', 'S'])
    for i in range(len(IDs)):
            mean_CRP_BL, std_CRP_BL = compute_CRP_ss_speed_threetrials(angles_BL[i], IDs[i], crp, LA_sides[i], MA_sides[i], MA, plot=False)
            mean_CRP_T0, std_CRP_T0 = compute_CRP_ss_speed_threetrials(angles_T0[i], IDs[i], crp, LA_sides[i], MA_sides[i], MA, plot=False)
            df_CRP.loc[len(df_CRP)] = [IDs[i], mean_CRP_BL, 'BL', 1, 2.0]
            df_CRP.loc[len(df_CRP)] = [IDs[i], mean_CRP_T0, 'T0', 2, 2.0]

    print(df_CRP)        
    fig = px.scatter_polar(df_CRP, theta="CRP", symbol='Time', color='ID', r='R', size='S')
    fig.update_layout(title = title, showlegend = legend, polar = dict(
          radialaxis = dict(visible = False, range = [0, 2])))
    fig.show()
    