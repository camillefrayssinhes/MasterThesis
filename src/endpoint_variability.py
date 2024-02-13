import numpy as np
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats
import natsort
import math
import pathlib
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from numpy import linalg as LA
from src.gait_cycle import *

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    Adapted from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    scaled_radii : array_like, shape (1, 2)
        Radii of the scaled ellipse
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),width=ell_radius_x * 2,height=ell_radius_y * 2,facecolor=facecolor,**kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
        
    eigenvalues, eigenvectors = LA.eig(cov)
    scaled_radii = np.array([np.sqrt(eigenvalues[0])*n_std, np.sqrt(eigenvalues[1])*n_std])
    
    return ellipse, scaled_radii   


def compute_EV(trajectories, ID, side):
    """
    Compute the endpoint variability. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' for left and 'R' for right
    Outputs:
        * EVs (list 1x3): EV of the self-selected, fast and slow speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    EVs = []
    areas_slow = []; areas_ss = []; areas_fast = []
    # store all vertical and fore-aft toe trajectories during swing phase of each gait cycle for each speed
    TOE_Y_swing_all_ss = []; TOE_Z_swing_all_ss = []
    TOE_Y_swing_all_fast = []; TOE_Z_swing_all_fast = []
    TOE_Y_swing_all_slow = []; TOE_Z_swing_all_slow = []
    fig, axs = plt.subplots(1,3,figsize=(20,6), sharex=True)

    for i in range(len(trajectories)):
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                # recover vertical and fore-aft trajectories of toe + heel trajectory
                trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z
                trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y
                trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_HEE_Y, trajectory_TOE_Y)
                # loop on each gait cycle
                if (ID == "BO2ST_106" and side == 'R' and i==8):
                    min_ = 12
                else: 
                    min_ = min(len(heel_strike), len(toe_off))
                for p in range(min_-1):
                    a = heel_strike[p]
                    b = toe_off[p]
                    if (b<a):
                        b = toe_off[p+1]
                    c = heel_strike[p+1]
                    # normalize vertical position of the toe during swing phase in [0:1]
                    TOE_Z_swing = (trajectory_TOE_Z[b:c] - trajectory_TOE_Z[b:c].min())/(trajectory_TOE_Z[b:c].max() - trajectory_TOE_Z[b:c].min())
                    # normalize fore-aft position of the toe during swing phase in [0:1]
                    TOE_Y_swing = (trajectory_TOE_Y[b:c] - trajectory_TOE_Y[b:c].min())/(trajectory_TOE_Y[b:c].max() - trajectory_TOE_Y[b:c].min())
                    # store all swing phase toe vertical and fore-aft trajectories
                    if (0<=i<3):
                        TOE_Y_swing_all_ss.append(TOE_Y_swing)
                        TOE_Z_swing_all_ss.append(TOE_Z_swing)
                    if (3<=i<6):
                        TOE_Y_swing_all_fast.append(TOE_Y_swing)
                        TOE_Z_swing_all_fast.append(TOE_Z_swing)
                    if (6<=i<9):
                        TOE_Y_swing_all_slow.append(TOE_Y_swing)
                        TOE_Z_swing_all_slow.append(TOE_Z_swing)    
                    # plot each swing phase toe vertical and fore-aft trajectory 
                    axs[math.floor(i/3)].plot(TOE_Y_swing, TOE_Z_swing, color='grey')


    # loop on each bin of 10% increments in the horizontal excursion of the toe   
    for k in range(10):
        x_tmp = []
        y_tmp = []
        for m in range(len(TOE_Z_swing_all_slow)):
            if (k==9): # if it is the last bin take all the values in the [90%:100%] range
                x = TOE_Y_swing_all_slow[m].loc[(np.logical_and(TOE_Y_swing_all_slow[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_slow[m].values.squeeze()<=np.round_(k/10+0.1, decimals = 1)))]
            else:
                # take all the fore-aft trajectories in the [j%:(j+1)%[ range with j in [0:8]
                x = TOE_Y_swing_all_slow[m].loc[(np.logical_and(TOE_Y_swing_all_slow[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_slow[m].values.squeeze()<np.round_(k/10+0.1, decimals = 1)))]
            # take all the vertical trajectories in the [j%:(j+1)%[ range with j in [0:8]
            y = TOE_Z_swing_all_slow[m].iloc[x.index[0] - TOE_Z_swing_all_slow[m].index[0]: x.index[-1] - TOE_Z_swing_all_slow[m].index[0] + 1]
            x = np.array(x.values.squeeze())
            y = np.array(y.values.squeeze())
            for z in range(len(x)):
                x_tmp.append(x[z])
                y_tmp.append(y[z])
        # plot a confidence ellipse enclosing 95% of the data points in that bin
        ell, radii = confidence_ellipse(np.asarray(x_tmp), np.asarray(y_tmp), axs[2], n_std = np.sqrt(5.991), edgecolor='black', facecolor='lightgrey', alpha=0.7, angle=0, zorder=2)
        axs[2].add_patch(ell)
        # compute area of each ellipse
        areas_slow.append(np.pi * np.product(radii))   
    
    
    # loop on each bin of 10% increments in the horizontal excursion of the toe   
    for k in range(10):
        x_tmp = []
        y_tmp = []
        for m in range(len(TOE_Z_swing_all_ss)):
            if (k==9): # if it is the last bin take all the values in the [90%:100%] range
                x = TOE_Y_swing_all_ss[m].loc[(np.logical_and(TOE_Y_swing_all_ss[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_ss[m].values.squeeze()<=np.round_(k/10+0.1, decimals = 1)))]
            else:
                # take all the fore-aft trajectories in the [j%:(j+1)%[ range with j in [0:8]
                x = TOE_Y_swing_all_ss[m].loc[(np.logical_and(TOE_Y_swing_all_ss[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_ss[m].values.squeeze()<np.round_(k/10+0.1, decimals = 1)))]
            # take all the vertical trajectories in the [j%:(j+1)%[ range with j in [0:8]
            y = TOE_Z_swing_all_ss[m].iloc[x.index[0] - TOE_Z_swing_all_ss[m].index[0]: x.index[-1] - TOE_Z_swing_all_ss[m].index[0] + 1]
            x = np.array(x.values.squeeze())
            y = np.array(y.values.squeeze())
            for z in range(len(x)):
                x_tmp.append(x[z])
                y_tmp.append(y[z])
        # plot a confidence ellipse enclosing 95% of the data points in that bin
        ell, radii = confidence_ellipse(np.asarray(x_tmp), np.asarray(y_tmp), axs[0], n_std = np.sqrt(5.991), edgecolor='black', facecolor='lightgrey', alpha=0.7, angle=0, zorder=2)
        axs[0].add_patch(ell)
        # compute area of each ellipse
        areas_ss.append(np.pi * np.product(radii))  
        
    # loop on each bin of 10% increments in the horizontal excursion of the toe   
    for k in range(10):
        x_tmp = []
        y_tmp = []
        for m in range(len(TOE_Z_swing_all_fast)):
            if (k==9): # if it is the last bin take all the values in the [90%:100%] range
                x = TOE_Y_swing_all_fast[m].loc[(np.logical_and(TOE_Y_swing_all_fast[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_fast[m].values.squeeze()<=np.round_(k/10+0.1, decimals = 1)))]
            else:
                # take all the fore-aft trajectories in the [j%:(j+1)%[ range with j in [0:8]
                x = TOE_Y_swing_all_fast[m].loc[(np.logical_and(TOE_Y_swing_all_fast[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_fast[m].values.squeeze()<np.round_(k/10+0.1, decimals = 1)))]
            # take all the vertical trajectories in the [j%:(j+1)%[ range with j in [0:8]
            y = TOE_Z_swing_all_fast[m].iloc[x.index[0] - TOE_Z_swing_all_fast[m].index[0]: x.index[-1] - TOE_Z_swing_all_fast[m].index[0] + 1]
            x = np.array(x.values.squeeze())
            y = np.array(y.values.squeeze())
            for z in range(len(x)):
                x_tmp.append(x[z])
                y_tmp.append(y[z])
        # plot a confidence ellipse enclosing 95% of the data points in that bin
        ell, radii  = confidence_ellipse(np.asarray(x_tmp), np.asarray(y_tmp), axs[1], n_std = np.sqrt(5.991), edgecolor='black', facecolor='lightgrey', alpha=0.7, angle=0, zorder=2)
        axs[1].add_patch(ell)
        # compute area of each ellipse
        areas_fast.append(np.pi * np.product(radii))   

    EVs.append(np.round(np.mean(areas_ss)*100,2)); EVs.append(np.round(np.mean(areas_fast)*100,2)); EVs.append(np.round(np.mean(areas_slow)*100,2))
    
    # figure parameters    
    axs[0].set_title('Self-selected'); axs[1].set_title('Fast'); axs[2].set_title('Slow')
    axs[0].set_xlabel('Normalized fore-aft position'); axs[0].set_ylabel('Normalized vertical position')
    axs[1].set_xlabel('Normalized fore-aft  position')
    axs[2].set_xlabel('Normalized fore-aft  position')
    axs[0].spines[['right', 'top']].set_visible(False)
    axs[1].spines[['right', 'top', 'left']].set_visible(False); axs[1].set_yticks([])
    axs[2].spines[['right', 'top', 'left']].set_visible(False); axs[2].set_yticks([])
    plt.show()
    plt.tight_layout(); 
        
    return EVs



def compute_EV_ss_speed_threetrials(trajectories, ID, side, new_ss = False, F1 = False, plot=False, F8 = False, F4=False, T0=False, T1=False, T2=False):
   
    """
    Compute the endpoint variability for the three trials at the self-selected speed. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' for left and 'R' for right
        * plot (bool): if True plot the EV
    Outputs:
        * EVs (list 1x3): EV of the self-selected, fast and slow speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    EVs = []
    areas_ss = []
    # store all vertical and fore-aft toe trajectories during swing phase of each gait cycle for each speed
    TOE_Y_swing_all_ss = []; TOE_Z_swing_all_ss = []
    fig, ax = plt.subplots(1,1,figsize=(14,8))
    
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
                #print(trajectories[j][0])
                if (trajectories[j][0] == 'BO2ST_103 Trial 02' and F1 == True):
                    break
                 
                if (trajectories[j][0] == 'BO2ST_102 Trial 02' and F8 == True):
                    break
                    
                if (trajectories[j][0] == 'BO2ST_109 Trial 03' and side=='R'):
                    #print('break')
                    break    
                
                elif ((trajectories[j][0] == 'BO2ST_103 Trial 02' and F4 == True) or (trajectories[j][0] == 'BO2ST_109 Trial 03' and T1==False and T0==True and T2==False)):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[0:4500]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[0:4500]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[0:4500]    
                    
                elif ((trajectories[j][0] == 'BO2ST_103 Trial 03' and F4 == True) or
                      (trajectories[j][0] == 'BO2ST_109 Trial 01' and side == 'R' and (T0 == True or T1==True)) or (trajectories[j][0] == 'BO2ST_109 Trial 03' and side == 'R' and T1 == True)):
                    #print('bla')
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[2000:]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[2000:]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[2000:] 
                       
                elif (trajectories[j][0] == 'BO2ST_103 Trial 01' and F8 == True):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[1500:]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[1500:]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[1500:] 
                    
                elif (trajectories[j][0] == 'BO2ST_109 Trial 02' and side == 'R'):
                    #print('here')
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[0:4000]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[0:4000]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[0:4000]
                    
                elif (trajectories[j][0] == 'BO2ST_109 Trial 02' and side == 'R' and T1==True):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[500:3000]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[500:3000]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[500:3000]    
                    
                elif (trajectories[j][0] == 'BO2ST_105 New_ss Trial 13'):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[3500:]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[3500:]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[3500:]   
                    
                elif (trajectories[j][0] == 'BO2ST_109 Trial 02' and side=='R' and T0==True):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[0:3000]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[0:3000]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[0:3000]       
                    
                elif (trajectories[j][0] == 'BO2ST_109 Trial 01' and T0==False):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[2000:7000]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[2000:7000]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[2000:7000]     
                    
                elif (trajectories[j][0] == 'BO2ST_103 Trial 03' or trajectories[j][0] == 'BO2ST_104 New_ss Trial 13' or (trajectories[j][0] == 'BO2ST_103 Trial 02' and F8 == True)):
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z[0:2500]
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y[0:2500]
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y[0:2500]
                else:    
                    # recover vertical and fore-aft trajectories of toe + heel trajectory
                    #print('else')
                    trajectory_TOE_Z = trajectories[j][1][ID+':' + side + 'TOE'].Z
                    trajectory_TOE_Y = trajectories[j][1][ID+':' + side + 'TOE'].Y
                    trajectory_HEE_Y = trajectories[j][1][ID+':' + side + 'HEE'].Y
                # extract heel_strike and toe_off events
                heel_strike, toe_off = compute_gait_cycle_2(trajectory_HEE_Y, trajectory_TOE_Y)

                # loop on each gait cycle
                if (trajectories == 'trajectories_106_BL' and side == 'R'):
                    min_ = 12
                else: 
                    min_ = min(len(heel_strike), len(toe_off))
                for p in range(min_-1):
                    a = heel_strike[p]
                    b = toe_off[p]
                    if (b<a):
                        b = toe_off[p+1]
                    c = heel_strike[p+1]
                    # normalize vertical position of the toe during swing phase in [0:1]
                    TOE_Z_swing = (trajectory_TOE_Z[b:c] - trajectory_TOE_Z[b:c].min())/(trajectory_TOE_Z[b:c].max() - trajectory_TOE_Z[b:c].min())
                    # normalize fore-aft position of the toe during swing phase in [0:1]
                    TOE_Y_swing = (trajectory_TOE_Y[b:c] - trajectory_TOE_Y[b:c].min())/(trajectory_TOE_Y[b:c].max() - trajectory_TOE_Y[b:c].min())
                    # store all swing phase toe vertical and fore-aft trajectories
                    TOE_Y_swing_all_ss.append(TOE_Y_swing)
                    TOE_Z_swing_all_ss.append(TOE_Z_swing)  
                    # plot each swing phase toe vertical and fore-aft trajectory 
                    if (plot):
                        ax.plot(TOE_Y_swing, TOE_Z_swing, color='grey')

    # loop on each bin of 10% increments in the horizontal excursion of the toe   
    for k in range(10):
        x_tmp = []
        y_tmp = []
        for m in range(len(TOE_Z_swing_all_ss)):
            if (k==9): # if it is the last bin take all the values in the [90%:100%] range
                x = TOE_Y_swing_all_ss[m].loc[(np.logical_and(TOE_Y_swing_all_ss[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_ss[m].values.squeeze()<=np.round_(k/10+0.1, decimals = 1)))]
            else:
                # take all the fore-aft trajectories in the [j%:(j+1)%[ range with j in [0:8]
                x = TOE_Y_swing_all_ss[m].loc[(np.logical_and(TOE_Y_swing_all_ss[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_ss[m].values.squeeze()<np.round_(k/10+0.1, decimals = 1)))]
            # take all the vertical trajectories in the [j%:(j+1)%[ range with j in [0:8]
            y = TOE_Z_swing_all_ss[m].iloc[x.index[0] - TOE_Z_swing_all_ss[m].index[0]: x.index[-1] - TOE_Z_swing_all_ss[m].index[0] + 1]
            x = np.array(x.values.squeeze())
            y = np.array(y.values.squeeze())
            for z in range(len(x)):
                x_tmp.append(x[z])
                y_tmp.append(y[z])
        # plot a confidence ellipse enclosing 95% of the data points in that bin
        ell, radii = confidence_ellipse(np.asarray(x_tmp), np.asarray(y_tmp), ax, n_std = np.sqrt(5.991), edgecolor='black', facecolor='lightgrey', alpha=0.7, angle=0, zorder=2)
        if (plot):
            ax.add_patch(ell)
        # compute area of each ellipse
        areas_ss.append(np.pi * np.product(radii))

    EVs.append(np.round(np.mean(areas_ss)*100,2))

    if (plot):
        # figure parameters    
        plt.xlabel('Normalized fore-aft position'); plt.ylabel('Normalized vertical position')
        ax.spines[['right', 'top']].set_visible(False)
        #ax.set_ylim(-0.2,1.4)
        plt.show()
        plt.tight_layout(); 

    plt.close()
    
    return EVs

def compute_EV_left_and_right(trajectories, ID, new_ss = False, F1 = False, plot=False, F8 = False, F4=False, T0=False, T1=False, T2=False):
    
    EV_left = compute_EV_ss_speed_threetrials(trajectories, ID, 'L', new_ss, F1, plot, F8, F4, T0, T1, T2)
    EV_right = compute_EV_ss_speed_threetrials(trajectories, ID, 'R', new_ss, F1, plot, F8, F4, T0, T1, T2)
    
    return EV_left, EV_right
    

def compute_EV_onetrial(trajectory_LTOE_Z, trajectory_LTOE_Y, trajectory_LHEE_Y):
    
    """
    Compute and print the endpoint variability for one trial. 
    
    Inputs:
        * trajectory_LTOE_Z (list): contains the trajectory of the TOE marker z-axis
        * trajectory_LTOE_Y (list): contains the trajectory of the TOE marker y-axis
        * trajectory_LHEE_Y (list): contains the trajectory of the HEEL marker y-axis
    """

    # compute gait cycle thanks to left heel trajectory and extract heel_strike and toe_off events
    heel_strike, toe_off = compute_gait_cycle_2(trajectory_LHEE_Y, trajectory_LTOE_Y)

    fig, ax = plt.subplots(1,1,figsize=(20,8))

    # store all vertical and fore-aft toe trajectories during swing phase of each gait cycle of one trial
    LTOE_Y_swing_all = []; LTOE_Z_swing_all = []

    min_ = min(len(heel_strike), len(toe_off))
    # loop on each gait cycle
    for i in range(min_-1):
        a = heel_strike[i]
        b = toe_off[i]
        if (b<a):
            b = toe_off[i+1]
        c = heel_strike[i+1]
        # normalize vertical position of the toe during swing phase in [0:1]
        LTOE_Z_swing = (trajectory_LTOE_Z[b:c] - trajectory_LTOE_Z[b:c].min())/(trajectory_LTOE_Z[b:c].max() - trajectory_LTOE_Z[b:c].min())
        # normalize fore-aft position of the toe during swing phase in [0:1]
        LTOE_Y_swing = (trajectory_LTOE_Y[b:c] - trajectory_LTOE_Y[b:c].min())/(trajectory_LTOE_Y[b:c].max() - trajectory_LTOE_Y[b:c].min())
        #1rint(LTOE_Y_swing)
        # store all swing phase toe vertical and fore-aft trajectories
        LTOE_Y_swing_all.append(LTOE_Y_swing)
        LTOE_Z_swing_all.append(LTOE_Z_swing)
        # plot each swing phase toe vertical and fore-aft trajectory 
        ax.plot(LTOE_Y_swing, LTOE_Z_swing, color='grey')


    # loop on each bin of 10% increments in the horizontal excursion of the toe   
    areas = []
    for j in range(10):
        x_tmp = []
        y_tmp = []
        for k in range(len(LTOE_Z_swing_all)):
            if (j==9): # if it is the last bin take all the values in the [90%:100%] range
                x = LTOE_Y_swing_all[k].loc[(np.logical_and(LTOE_Y_swing_all[k].values.squeeze()>=np.round_(j/10, decimals = 1), LTOE_Y_swing_all[k].values.squeeze()<=np.round_(j/10+0.1, decimals = 1)))]
            else:
                # take all the fore-aft trajectories in the [j%:(j+1)%[ range with j in [0:8]
                x = LTOE_Y_swing_all[k].loc[(np.logical_and(LTOE_Y_swing_all[k].values.squeeze()>=np.round_(j/10, decimals = 1), LTOE_Y_swing_all[k].values.squeeze()<np.round_(j/10+0.1, decimals = 1)))]
            # take all the vertical trajectories in the [j%:(j+1)%[ range with j in [0:8]
            #print(LTOE_Y_swing_all[k])
            y = LTOE_Z_swing_all[k].iloc[x.index[0] - LTOE_Z_swing_all[k].index[0]: x.index[-1] - LTOE_Z_swing_all[k].index[0] + 1]
            #print(len(x))
            x = np.array(x.values.squeeze())
            y = np.array(y.values.squeeze())
            for z in range(len(x)):
                x_tmp.append(x[z])
                y_tmp.append(y[z])
        # plot a confidence ellipse enclosing 95% of the data points in that bin
        ell, radii = confidence_ellipse(np.asarray(x_tmp), np.asarray(y_tmp), ax, n_std = np.sqrt(5.991), edgecolor='black', facecolor='lightgrey', angle=0)
        print(radii)
        ax.add_patch(ell)
        # compute area of each ellipse
        areas.append(np.pi * np.product(radii)*100)

    # figure parameters    
    plt.xlabel('Normalized fore-aft position'); plt.ylabel('Normalized vertical position')   
    plt.show()
    plt.tight_layout();   

    # print mean area
    print(np.round((areas),2))
    

def compute_EV_ss_plot(trajectories_BL, trajectories_T0, IDs, sides):
    

    fig, ax = plt.subplots(1,figsize= (7,5))

    for i in range(len(IDs)):
        EV_BL = compute_EV_ss_speed_threetrials(trajectories_BL[i], IDs[i], sides[i], plot=False)
        EV_T0 = compute_EV_ss_speed_threetrials(trajectories_T0[i], IDs[i], sides[i], plot=False)
        data = [EV_BL, EV_T0]
        ax.plot(["BL","T0"],data, label=IDs[i], color="C"+str(i))
        ax.plot(["BL","T0"],data, 'o',color="C"+str(i))
    #ax.legend(title = "Participants")
    ax.spines[['right', 'top']].set_visible(False)

    # figure parameters
    fig.show()
    fig.tight_layout();
    
    
def compute_EV_AB_ss(trajectories, number, side, plot=False):
    """
    Compute the endpoint variability. 
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * side (string): 'L' for left and 'R' for right
    Outputs:
        * EVs (list 1x3): EV of the self-selected, fast and slow speed
    """

    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    EVs = []
    areas_ss = []
    # store all vertical and fore-aft toe trajectories during swing phase of each gait cycle for each speed
    TOE_Y_swing_all_ss = []; TOE_Z_swing_all_ss = []
    fig, ax = plt.subplots(1,1,figsize=(20,6), sharex=True)

    for i in range(len(trajectories)):
        # sort the trials
        if (trajectories[i][0] == 'WBDS'+number+'walkT05mkr'):
            # recover vertical and fore-aft trajectories of toe + heel trajectory
            trajectory_TOE_Z = trajectories[i][1][side + '.MT1Y']
            trajectory_TOE_Y = trajectories[i][1][side + '.MT1X']
            trajectory_HEE_Y = trajectories[i][1][side + '.HeelX']
            # extract heel_strike and toe_off events
            heel_strike, toe_off = compute_gait_cycle_2(trajectory_HEE_Y, trajectory_TOE_Y)
            # loop on each gait cycle
            min_ = min(len(heel_strike), len(toe_off))
            for p in range(min_-1):
                a = heel_strike[p]
                b = toe_off[p]
                if (b<a):
                    b = toe_off[p+1]
                c = heel_strike[p+1]
                # normalize vertical position of the toe during swing phase in [0:1]
                TOE_Z_swing = (trajectory_TOE_Z[b:c] - trajectory_TOE_Z[b:c].min())/(trajectory_TOE_Z[b:c].max() - trajectory_TOE_Z[b:c].min())
                # normalize fore-aft position of the toe during swing phase in [0:1]
                TOE_Y_swing = (trajectory_TOE_Y[b:c] - trajectory_TOE_Y[b:c].min())/(trajectory_TOE_Y[b:c].max() - trajectory_TOE_Y[b:c].min())
                # store all swing phase toe vertical and fore-aft trajectories
                TOE_Y_swing_all_ss.append(TOE_Y_swing)
                TOE_Z_swing_all_ss.append(TOE_Z_swing) 
                if (plot):
                    # plot each swing phase toe vertical and fore-aft trajectory 
                    ax.plot(TOE_Y_swing, TOE_Z_swing, color='grey') 

    
    # loop on each bin of 10% increments in the horizontal excursion of the toe   
    for k in range(10):
        x_tmp = []
        y_tmp = []
        for m in range(len(TOE_Z_swing_all_ss)):
            if (k==9): # if it is the last bin take all the values in the [90%:100%] range
                x = TOE_Y_swing_all_ss[m].loc[(np.logical_and(TOE_Y_swing_all_ss[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_ss[m].values.squeeze()<=np.round_(k/10+0.1, decimals = 1)))]
            else:
                # take all the fore-aft trajectories in the [j%:(j+1)%[ range with j in [0:8]
                x = TOE_Y_swing_all_ss[m].loc[(np.logical_and(TOE_Y_swing_all_ss[m].values.squeeze()>=np.round_(k/10, decimals = 1), TOE_Y_swing_all_ss[m].values.squeeze()<np.round_(k/10+0.1, decimals = 1)))]
            # take all the vertical trajectories in the [j%:(j+1)%[ range with j in [0:8]
            y = TOE_Z_swing_all_ss[m].iloc[x.index[0] - TOE_Z_swing_all_ss[m].index[0]: x.index[-1] - TOE_Z_swing_all_ss[m].index[0] + 1]
            x = np.array(x.values.squeeze())
            y = np.array(y.values.squeeze())
            for z in range(len(x)):
                x_tmp.append(x[z])
                y_tmp.append(y[z])
        # plot a confidence ellipse enclosing 95% of the data points in that bin
        ell, radii = confidence_ellipse(np.asarray(x_tmp), np.asarray(y_tmp), ax, n_std = np.sqrt(5.991), edgecolor='black', facecolor='lightgrey', alpha=0.7, angle=0, zorder=2)
        if (plot):
            ax.add_patch(ell)
        # compute area of each ellipse
        areas_ss.append(np.pi * np.product(radii))  
         

    EVs.append(np.round(np.mean(areas_ss)*100,2))
   
    if (plot):
        # figure parameters    
        ax.set_title('Self-selected')
        ax.set_xlabel('Normalized fore-aft position'); ax.set_ylabel('Normalized vertical position')
        ax.spines[['right', 'top']].set_visible(False)
        plt.show()
        plt.tight_layout(); 
    
    plt.close()
    
    return EVs


def compute_EV_AB_left_and_right(trajectories, number):
    
    EV_left = float(np.squeeze(compute_EV_AB_ss(trajectories, number, 'L')))
    EV_right = float(np.squeeze(compute_EV_AB_ss(trajectories, number, 'R')))
    
    return EV_left, EV_right



