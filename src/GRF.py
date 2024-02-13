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
from src.gait_cycle import *
from shapely.geometry import Polygon

def normalize_stance(heel_strike, toe_off, force, safety_check=False):
    """
    Normalize each stance phase in the selected force to 100 frames.

    Inputs:
        * heel_strike (list): time points of the heel strike events 
        * toe_off (list): time point of the toe off events
        * force (list): ground reaction force
        * safety_check (bool): if True, print the total number of time frames and the total number of gait cycles. The printed length of the list should be the printed number of gait cycles times 250. 
        
    Outputs:
        * force_normalized (list): normalized force
    """
    
    force_normalized = []
    
    min_ = min(len(heel_strike), len(toe_off))
    for i in range(min_-1):
        a = heel_strike[i]
        b = toe_off[i]
        if (b<a):
            b = toe_off[i+1]
        c = heel_strike[i+1]
        # cut stance and swing phases
        stance = pd.DataFrame(force[a:b])
        # normalize stance phase from 0 to 99 (100 frames)
        stance['percent'] = (np.arange(len(stance))+1)/len(stance)*100
        stance.set_index('percent',inplace =True)
        stanceresampled = np.linspace(0,99,100)
        stance_normalized = stance.reindex(stance.index.union(stanceresampled)).interpolate('values').loc[stanceresampled]
        #print(stance_normalized)
        # append to array
        force_normalized.append(stance_normalized)
    
    if (safety_check):
        # safety check
        print('total number of time frames: ' + str(len(force_normalized)))
        print('total number of gait cycles: ' + str(len(force_normalized)/100))
    
    return force_normalized   


def compute_fore_aft_GRF(ID, side, trajectories, forces, forces_cal, plot=False, new_ss=False, T0=False, T1=False, T2=False, F8=False):
    """
    Compute the mean of the 3 parameters of the fore_aft GRF at the 3 trials at self-selected speed.
    Inputs:
        * ID (string): e.g. "BO2ST_101"
        * side (string): most affected side e.g. 'L'
        * trajectories (list): 
        * forces_cal (list): forces of the calibration trials extracted by the download_parameters_force_calibration()
    Outputs:
        * tp (float): The transition point was defined as the time point (% stance phase) that the force curve
        switches from negative (braking) to positive (propulsion)
        * braking_area (float):  Braking impulse was defined as the area under the negative force curve
        before the transition point.
        * propul_area (float): Propulsive impulse was defined as the area under the positive force curve
        after the transition point.
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    # extract weight of the participant (in Newton)
    weight_file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    weight_df = pd.read_excel(weight_file_name, header = [0], index_col = [0])
    weight = weight_df.loc[ID]["Weight"]*9.81
    
    # extract calibration weight
    if (side=='L'):
        cal_side = 'Right' # the participant is on the right force plate so the left force plate is empty
        weight_side = 'Left'
    elif (side=='R'):
        cal_side = 'Left'
        weight_side = 'Right'
    mean = 0    
    for b in range(len(forces_cal)):  
        if (forces_cal[b][0] == ID+' Cal '+cal_side): # i'm taking the calibratrion trial in which the participant is on the opposite force plate
            mean = np.mean(forces_cal[b][1][weight_side+' Force Plate - Force'].Fy/weight*100) # i'm taking the force plate with the null force
    mean_tp = []; mean_braking_area = []; mean_propul_area = []

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
                # extract trajectories of heel and toe 
                trajectory_heel = trajectories[j][1][ID+":"+side+"HEE"].Y
                trajectory_toe = trajectories[j][1][ID+":"+side+"TOE"].Y

                # compute heel strike and toe off events 
                hs, to = compute_gait_cycle_2(trajectory_heel, trajectory_toe)

                # extract and normalize fore-aft grf
                adjusted_force =(forces[j][1][weight_side+' Force Plate - Force'].Fy)/weight*100 - mean
                adjusted_force = - adjusted_force
                # resample 
                adjusted_force = resample_by_interpolation(adjusted_force.squeeze(), 2500, 250)

                # extract and normalize stance phases
                stances = normalize_stance(hs, to, adjusted_force)
                
                # compute the mean GRF across all gait cycles during the trials
                mean_stance = pd.Series(np.mean(stances, axis=0).squeeze())
                mean_stance = list(mean_stance.rolling(window=5).mean())# rolling window to smooth the curve
                
                
                if (plot):
                    fig, ax = plt.subplots(1, figsize=(20,6))
                    ax.plot(mean_stance, color='black')
                    ax.axhline(0, linestyle = '--', color='black')
                    ax.set_title(trajectories[j][0])
                
                # compute transition point
                tp = None
                for v in range(len(mean_stance)-3):
                    if(mean_stance[v]>=0 and mean_stance[v-1]<0 and mean_stance[v-2]<0 and mean_stance[v+1]>=0 and mean_stance[v+2]>=0 and mean_stance[v+3]>=0):
                        if (ID=='BO2ST_102' and new_ss ==True and T0==True and v<60):
                            tp = v
                        if (ID !='BO2ST_102' or (ID=='BO2ST_102' and new_ss ==False)):
                            tp = v
                # compute first point to compute area
                first = 5
                for w in range(len(mean_stance)-3):
                    if(mean_stance[w]<=0 and mean_stance[w-1]>0 and mean_stance[w-2]>0 and mean_stance[w+1]<0 and mean_stance[w+2]<0 and mean_stance[w+3]<0 and w<40):
                        if (ID=='BO2ST_104' and w<30):
                            first = w
                        if (ID!='BO2ST_104'):
                            first = w    
                # compute last point to compute area
                last = 99
                for x in range(len(mean_stance)-3):
                    if(mean_stance[x]<=0 and mean_stance[x-1]>0 and mean_stance[x-2]>0 and mean_stance[x+1]<0 and mean_stance[x+2]<0 and mean_stance[x+3]<0):
                        if (ID=='BO2ST_103' and x>50 and x<90):
                            last = x     
                        elif(ID!='BO2ST_103' and x>70):
                            last = x

                # plot
                if (tp!=None):
                    if ((ID=="BO2ST_103" and tp<75 and tp>10) or (ID!="BO2ST_103" and ID!="BO2ST_102") or (ID=="BO2ST_102" and tp>60 and new_ss==False) or (ID=="BO2ST_102" and tp<60 and new_ss==True) or (ID=="BO2ST_102" and F8==True)):
                        
                        if (plot):
                            print(trajectories[j][0])
                            print(first)
                            print(last)
                            print(tp)
                            ax.plot(tp, mean_stance[tp], 'ro')
                            ax.plot(first, mean_stance[first], 'bo')
                            ax.plot(last, mean_stance[last], 'go')

                        # compute braking area
                        braking_curve1 = [(m, np.abs(float(mean_stance[m]))) for m in np.arange(first,tp,1)] #these are your points for the grf curves
                        braking_curve2 = [(n, 0) for n in np.arange(first,tp,1)] #these are your points for 0 horizontal line

                        braking_polygon_points = []

                        for xyvalue in braking_curve1:
                            if not any(math.isnan(val) for val in xyvalue):
                                braking_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

                        for xyvalue in braking_curve2[::-1]:
                            if not any(math.isnan(val) for val in xyvalue):
                                braking_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

                        for xyvalue in braking_curve1[0:1]:
                            if not any(math.isnan(val) for val in xyvalue):
                                braking_polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

                        braking_polygon = Polygon(braking_polygon_points)
                        braking_area = braking_polygon.area

                        # compute propulsion area
                        propulsion_curve1 = [(o, np.abs(float(mean_stance[o]))) for o in np.arange(tp,last,1)] #these are your points for the grf curve
                        propulsion_curve2 = [(p, 0) for p in np.arange(tp,last,1)] #these are your points for 0 horizontal line

                        propulsion_polygon_points = []

                        for xyvalue in propulsion_curve1:
                            if not any(math.isnan(val) for val in xyvalue):
                                propulsion_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

                        for xyvalue in propulsion_curve2[::-1]:
                            if not any(math.isnan(val) for val in xyvalue):
                                propulsion_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

                        for xyvalue in propulsion_curve1[0:1]:
                            if not any(math.isnan(val) for val in xyvalue):
                                propulsion_polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

                        propulsion_polygon = Polygon(propulsion_polygon_points)
                        propul_area = propulsion_polygon.area

                        if (plot):
                            # Remove the top and right spines
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            #plt.ylim(-25,25)
                            plt.show()

                        # compute mean over all the gait cycles in the trial     
                        mean_tp.append(tp); mean_braking_area.append(braking_area); mean_propul_area.append(propul_area)    

    # compute mean over the 3 trials at self-selected speed
    mean_tp = np.nanmean(mean_tp); mean_braking_area = np.nanmean(mean_braking_area); mean_propul_area = np.nanmean(mean_propul_area)
            
    return mean_tp, mean_braking_area, mean_propul_area


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


def compute_fore_aft_GRF_AB(ID_AB, trajectories_AB, forces_AB, plot=False):
    """
    Compute the mean of the 3 parameters of the fore_aft GRF at the 3 trials at self-selected speed.
    Inputs:
        * ID_AB (string): ID of the AB control e.g. "AB43"
        * trajectories_AB (list): 
        * forces_AB (list):
    Outputs:
        * tp (float): The transition point was defined as the time point (% stance phase) that the force curve
        switches from negative (braking) to positive (propulsion)
        * braking_area (float):  Braking impulse was defined as the area under the negative force curve
        before the transition point.
        * propul_area (float): Propulsive impulse was defined as the area under the positive force curve
        after the transition point.
    """
    
    number_AB = ID_AB[-2:]
    
    # extract weight of the AB control (in Newton)
    weight_file_name = ("BO2STTrial/WBDSascii/AB_weights.xlsx")
    weight_df = pd.read_excel(weight_file_name, header = [0], index_col = [0])
    weight = weight_df.loc[ID_AB]["Weight"]*9.81

    mean_tp = []; mean_braking_area = []; mean_propul_area = []
    
    # find the trajectory at ss speed
    for k in range(len(trajectories_AB)): # ss speed
            # sort the trials
            if (trajectories_AB[k][0] == 'WBDS'+number_AB+'walkT05mkr'):
                trajectory = trajectories_AB[k][1]
    
    for i in range(len(forces_AB)): # ss speed
            # sort the trials
            if (forces_AB[i][0] == 'WBDS'+number_AB+'walkT05grf'):

                # extract trajectories of heel and toe 
                trajectory_heel = trajectory['L' + '.HeelX']
                trajectory_toe = trajectory['L' + '.MT1X']

                # compute heel strike and toe off events 
                hs, to = compute_gait_cycle_2(trajectory_heel, trajectory_toe)

                # extract and normalize fore-aft grf
                adjusted_force =(forces_AB[i][1].Fx1)/weight*100
                # resample TO ADJUST
                adjusted_force = resample_by_interpolation(adjusted_force.squeeze(), 300, 150)
                # f = 150Hz for trajectories; f = 300Hz for forces

                # extract and normalize stance phases
                stances = normalize_stance(hs, to, adjusted_force)
                
                # compute the mean GRF across all gait cycles during the trials
                mean_stance = np.mean(stances, axis=0)
                
                if (plot):
                    fig, ax = plt.subplots(1, figsize=(20,6))

                # compute transition point
                tp = None
                for v in range(len(mean_stance)-3):
                    if(mean_stance[v]>=0 and mean_stance[v-1]<0 and mean_stance[v-2]<0 and mean_stance[v+1]>0 and mean_stance[v+2]>0 and mean_stance[v+3]>0):
                        tp = v
                # compute first point to compute area
                first = 5
                for w in range(len(mean_stance)-3):
                    if(mean_stance[w]<=0 and mean_stance[w-1]>0 and mean_stance[w-2]>0 and mean_stance[w+1]<0 and mean_stance[w+2]<0 and mean_stance[w+3]<0 and w<40):
                            first = w             
                # compute last point to compute area
                last = 99
                for x in range(len(mean_stance)-3):
                    if(mean_stance[x]<=0 and mean_stance[x-1]>0 and mean_stance[x-2]>0 and mean_stance[x+1]<0 and mean_stance[x+2]<0 and mean_stance[x+3]<0 and x>60):
                            last = x           
                # plot
                if (tp!=None):
                    if (plot):
                        #plt.plot(stances[k])
                        ax.plot(mean_stance, color='black')
                        ax.axhline(0, linestyle = '--', color='black')
                        ax.plot(tp, mean_stance[tp], 'ro')
                        ax.plot(first, mean_stance[first], 'bo')
                        ax.plot(last, mean_stance[last], 'go')

                    # compute braking area
                    braking_curve1 = [(m, np.abs(float(mean_stance[m]))) for m in np.arange(first,tp,1)] #these are your points for the grf curves
                    braking_curve2 = [(n, 0) for n in np.arange(first,tp,1)] #these are your points for 0 horizontal line

                    braking_polygon_points = []

                    for xyvalue in braking_curve1:
                        if not any(math.isnan(val) for val in xyvalue):
                            braking_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

                    for xyvalue in braking_curve2[::-1]:
                        if not any(math.isnan(val) for val in xyvalue):
                            braking_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

                    for xyvalue in braking_curve1[0:1]:
                        if not any(math.isnan(val) for val in xyvalue):
                            braking_polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

                    braking_polygon = Polygon(braking_polygon_points)
                    braking_area = braking_polygon.area

                    # compute propulsion area
                    propulsion_curve1 = [(o, np.abs(float(mean_stance[o]))) for o in np.arange(tp,last,1)] #these are your points for the grf curve
                    propulsion_curve2 = [(p, 0) for p in np.arange(tp,last,1)] #these are your points for 0 horizontal line

                    propulsion_polygon_points = []

                    for xyvalue in propulsion_curve1:
                        propulsion_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 1

                    for xyvalue in propulsion_curve2[::-1]:
                        propulsion_polygon_points.append([xyvalue[0],xyvalue[1]]) #append all xy points for curve 2 in the reverse order (from last point to first point)

                    for xyvalue in propulsion_curve1[0:1]:
                        propulsion_polygon_points.append([xyvalue[0],xyvalue[1]]) #append the first point in curve 1 again, to it "closes" the polygon

                    propulsion_polygon = Polygon(propulsion_polygon_points)
                    propul_area = propulsion_polygon.area

                if (plot):
                    # Remove the top and right spines
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.ylim(-25,25)
                    plt.show()

                # compute mean over all the gait cycles in the trial     
                mean_tp.append(tp); mean_braking_area.append(braking_area); mean_propul_area.append(propul_area)    

    # compute mean over the 3 trials at self-selected speed
    mean_tp = np.mean(mean_tp); mean_braking_area = np.mean(mean_braking_area); mean_propul_area = np.mean(mean_propul_area)
            
    return mean_tp, mean_braking_area, mean_propul_area