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
from sklearn.metrics import mean_absolute_error


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



def synchronize_data(hip_X_opencap, hip_X_vicon, angle_opencap, angle_name_opencap, angle_vicon, height_vicon, height_opencap, plot=False):
    """
    Synchronise the opencap and vicon angle time series according to the first left hip extension. 
    
    OCS01:
    Vicon data @ 250Hz
    Trial01: height_hip_left_X_opencap = 20; height_hip_left_X_vicon = 0
    Trial02: height_hip_left_X_opencap = 30; height_hip_left_X_vicon = 33
    Trial03: height_hip_left_X_opencap = 25; height_hip_left_X_vicon = 20
    
    OCA01:
    Vicon data @ 240Hz!!

    """
    # hip
    ilocs_max_hip_X_opencap = find_peaks(hip_X_opencap, height=height_opencap)[0] # the height has to be modified for different trials
    hip_X_vicon_resample = resample_by_interpolation(hip_X_vicon, 240, 60)
    ilocs_max_hip_X_vicon = find_peaks(hip_X_vicon_resample, height=height_vicon)[0] # the height has to be modified for different trials
    
    angle_opencap_synchro = angle_opencap[ilocs_max_hip_X_opencap[0]:].reset_index()[angle_name_opencap]
    angle_vicon_synchro = angle_vicon[ilocs_max_hip_X_vicon[0]:]
    
    if (plot):
        plt.figure(figsize=(20,6))
        plt.plot(angle_opencap_synchro[1000:1800].values)
        plt.plot(angle_vicon_synchro[1000:1800])
    
    return angle_opencap_synchro, angle_vicon_synchro


def get_joint_angles(hip_X_opencap, hip_X_vicon, angle_opencap, angle_opencap_name, angle_vicon, height_vicon, height_opencap):
    """
    Resample, synchronise, and plot the joint angles of opencap and vicon. 
    """
    
    # adjust sampling frequency 
    angle_vicon = resample_by_interpolation(angle_vicon, 250, 60) # OpenCap is set at 60Hz of sampling frequency. Vicon is set at 250Hz of sampling frequency. I have to resample the Vicon data at 60Hz.

    # synchronize data
    angle_opencap, angle_vicon = synchronize_data(hip_X_opencap, hip_X_vicon, angle_opencap, angle_opencap_name, angle_vicon, height_vicon, height_opencap)
    
    # extract 15sec @ self-selected speed 
    angle_opencap_ss = angle_opencap[100:1000].values
    angle_vicon_ss = angle_vicon[100:1000]
    
    # plot
    plt.figure(figsize=(20,6))
    plt.plot(angle_opencap_ss, label='opencap')
    plt.plot(angle_vicon_ss, label='vicon')
    plt.legend()
    
    return angle_opencap_ss, angle_vicon_ss


def compute_MAE_joint_angles(joint_angle_OpenCap, joint_angle_Vicon):
    """
    Compute the mean absolute error (MAE) between the two synchronized joint angle time series from the OpenCap software and the Vicon motion capture system.
    
    Inputs: 
        * joint_angle_OpenCap (list): selected synchronized joint angle estimated by the OpenCap software 
        * joint_angle_Vicon (list): selected synchronized joint angle computed by the Vicon motion capture system
    Outputs: 
        * MAE (float): MAE across the trial
    """
    
    y_true = joint_angle_Vicon # gold-standard
    y_pred = joint_angle_OpenCap # estimated joint angles
    MAE = mean_absolute_error(y_true, y_pred)
    
    return MAE

