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
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
import alphashape


def inverse_logit_after_log_norm(x):
    """
    To map (0, inf) into (-inf, inf), you can use y = log(x)
    To map (-inf, inf) into (0, 1), you can use y = 1 / (1 + exp(-x)) (inverse logit)
    To map (0, inf) into (0, 1), you can use y = x / (1 + x) (inverse logit after log)
    """
    
    x_norm = x/(1+x)
    
    return x_norm

def bilateral_cyclograms(angles, trajectories, ID, plot=False):
    """
    Plot bilateral cyclogram. 
    The area of the convex hull represents the area of the closed curve obtained from each angle–angle diagram. The smaller the area, the more symmetrical the gait.
    Inputs:
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
        * plot (bool): plot the bilateral cyclogram if True
    Outputs:
        * area (float): area of the convex hull within the cyclogram
        * angle (float): orientation of the cyclogram
        * ratio (float): the trend symmetry in %
    """
    
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)
    
    area_slow = []; area_ss = []; area_fast = []
    angle_slow = []; angle_ss = []; angle_fast = []
    ratio_slow = []; ratio_ss = []; ratio_fast = []
    
    for i in range(len(trajectories)):
        for j in range(len(list_sorted)):
            # sort the trials
            if (trajectories[j][0] == list_sorted[i]):
                # extract left and right knee angles
                knee_angles_left = angles[j][1][ID+":"+"L"+'KneeAngles'].X
                knee_angles_right = angles[j][1][ID+":"+"R"+'KneeAngles'].X
                # extract left and right heel and toe trajectories 
                trajectory_heel_left = trajectories[j][1][ID+":"+"L"+'HEE'].Y
                trajectory_toe_left = trajectories[j][1][ID+":"+"L"+'TOE'].Y
                trajectory_heel_right = trajectories[j][1][ID+":"+"R"+'HEE'].Y
                trajectory_toe_right = trajectories[j][1][ID+":"+"R"+'TOE'].Y
                # extract left and right heel strike and toe off events
                hs_r, to_r = compute_gait_cycle_2(trajectory_heel_right, trajectory_toe_right)
                hs_l, to_l = compute_gait_cycle_2(trajectory_heel_left, trajectory_toe_left)
                # normalize left and right knee angle trajectories
                knee_right_norm = normalize_gait_cycle(hs_r, to_r, knee_angles_right, safety_check=False)
                knee_left_norm = normalize_gait_cycle(hs_l, to_l, knee_angles_left, safety_check=False)
                # get minimum len of trajectory 
                min_ = min(len(knee_right_norm), len(knee_left_norm))
                # get maximum angle between left and right leg
                max_ = max(np.nanmax(knee_left_norm), np.nanmax(knee_right_norm))

                # get the set of points
                k = 0
                for m in range(len(knee_right_norm[0:min_])):
                    if (str(knee_right_norm[0:min_][m]) != 'nan' and str(knee_left_norm[0:min_][m]) != 'nan'):
                        k+=1
                points = np.empty(shape=(k, 2), dtype='object')
                
                k = 0
                for m in range(len(knee_right_norm[0:min_])):
                    if (str(knee_right_norm[0:min_][m]) != 'nan' and str(knee_left_norm[0:min_][m]) != 'nan'):
                        points[k,0] = knee_right_norm[0:min_][m]
                        points[k,1] = knee_left_norm[0:min_][m]
                        k+=1
                # compute convex hull --> NOT DOING THAT ANYMORE
                #hull = ConvexHull(points)
                #area = hull.volume # dimension = 2
                alpha = 0.2
                # Create alpha shape (concave hull)
                alpha_shape = alphashape.alphashape(points.astype(float), alpha)
                exterior_coords = alpha_shape.exterior.coords.xy
                # Create a Polygon object from the coordinates
                hull_polygon = Polygon(list(zip(exterior_coords[0], exterior_coords[1])))
                # Calculate the area of the concave hull
                area = hull_polygon.area

                # compute the covariance matrix
                cov = np.cov(points.astype(float), rowvar=False)
                # compute the eigenvalues and eigenvectors of the covariance matrix
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                # find the index of the maximum eigenvalue
                max_index = np.argmax(eigenvalues)
                # get the corresponding eigenvector
                principal_axis = eigenvectors[:, max_index]
                # compute cyclogram orientation
                centroid = np.mean(points, axis=0)
                slope = principal_axis[1] / principal_axis[0]
                y_intercept = centroid[1] - slope * centroid[0]
                # compute angle
                angle = np.abs(np.rad2deg(math.atan ((0-slope)/(1+0*slope))))

                # translate points
                points_translated = points - np.mean(points, axis=0)
                # rotate points
                theta = np.deg2rad(angle)
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                points_rotated = np.matmul(points_translated,R)
                # get variation along the eigen vector
                var_X = np.var(points_rotated[:,0])
                # get variation about the eigen vector
                var_Y = np.var(points_rotated[:,1])
                # compute ratio (trend symmetry)
                ratio = var_Y/var_X
                
                if (0<=i<3):
                    area_ss.append(area)
                    angle_ss.append(angle)
                    ratio_ss.append(ratio*100)
                if (3<=i<6):
                    area_fast.append(area)
                    angle_fast.append(angle)
                    ratio_fast.append(ratio*100)
                if (6<=i<9):
                    area_slow.append(area)
                    angle_slow.append(angle)
                    ratio_slow.append(ratio*100) 

                if (plot):
                    
                    # print
                    print('Area: ' + str(np.round(area,2)))
                    print('Angle: ' + str(np.round(angle,2)))
                    print('Ratio: ' + str(np.round(ratio*100,2)) +'%')

                    fig, ax = plt.subplots(1,1,figsize= (5,5))
                    # plot bilateral cyclogram 
                    ax.plot(knee_right_norm[0:min_], knee_left_norm[0:min_], color='black', alpha=0.2)
                    # plot symmetry line
                    ax.plot(np.arange(max_), np.arange(max_), linestyle='dashed', color='grey')
                    # plot convex hull
                    #for simplex in hull.simplices:
                        #plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
                    plt.plot(*alpha_shape.exterior.xy, 'k-', label='Concave Hull')     
                    # plot x-abcissa
                    ax.axhline(0, linestyle = 'dashed', color='grey')
                    # plot cyclogram orientation
                    ax.plot(np.arange(max_), slope*np.arange(max_)+y_intercept, color='red')
                    # figure parameters
                    ax.set_xlabel('Knee angle right [deg]'); ax.set_ylabel('Knee angle left [deg]')
                    ax.set_title('Knee angle bilateral cyclogram')
                    ax.spines[['right', 'top']].set_visible(False)
                    fig.tight_layout()
                    fig.show()

    asymmetry_ss = [area_ss, angle_ss, ratio_ss]
    asymmetry_fast = [area_fast, angle_fast, ratio_fast]
    asymmetry_slow = [area_slow, angle_slow, ratio_slow]
    
    return asymmetry_ss, asymmetry_fast, asymmetry_slow



def bilateral_cyclograms_single_trial(angle, trajectory, ID, joint, plot=False):
    """
    Inputs:
        * joint(string): "Knee", "Hip", or "Ankle"
    Output: 
        * A (float): asymmetry score 
    """

    # extract left and right joint angles
    joint_angles_left = angle[ID+":"+"L"+joint+'Angles'].X
    joint_angles_right = angle[ID+":"+"R"+joint+'Angles'].X
    # extract left and right heel and toe trajectories 
    trajectory_heel_left = trajectory[ID+":"+"L"+'HEE'].Y
    trajectory_toe_left = trajectory[ID+":"+"L"+'TOE'].Y
    trajectory_heel_right = trajectory[ID+":"+"R"+'HEE'].Y
    trajectory_toe_right = trajectory[ID+":"+"R"+'TOE'].Y
    # extract left and right heel strike and toe off events
    hs_r, to_r = compute_gait_cycle_2(trajectory_heel_right, trajectory_toe_right)
    hs_l, to_l = compute_gait_cycle_2(trajectory_heel_left, trajectory_toe_left)
    # normalize left and right knee angle trajectories
    joint_right_norm = normalize_gait_cycle(hs_r, to_r, joint_angles_right, safety_check=False)
    joint_left_norm = normalize_gait_cycle(hs_l, to_l, joint_angles_left, safety_check=False)
    # average the gait cycles
    joint_right_norm_avg = []; joint_left_norm_avg = []
    for i in range(int(len(joint_right_norm)/250)):
        joint_right_norm_avg.append(joint_right_norm[250*i:250*(i+1)])
    for i in range(int(len(joint_left_norm)/250)):
        joint_left_norm_avg.append(joint_left_norm[250*i:250*(i+1)])
    joint_right_norm_avg = np.nanmean(joint_right_norm_avg, axis=0)
    joint_left_norm_avg = np.nanmean(joint_left_norm_avg, axis=0)
       
    # get minimum len of trajectory (= minimum number of gait cycles)
    min_ = min(len(joint_right_norm_avg), len(joint_left_norm_avg))
    # get maximum angle between left and right leg
    max_ = max(np.nanmax(joint_left_norm_avg), np.nanmax(joint_right_norm_avg))

    # get the set of points
    k = 0
    for m in range(len(joint_right_norm_avg[0:min_])):
        if (str(joint_right_norm_avg[0:min_][m]) != 'nan' and str(joint_left_norm_avg[0:min_][m]) != 'nan'):
            k+=1
    points = np.empty(shape=(k, 2), dtype='object')

    k = 0
    for m in range(len(joint_right_norm_avg[0:min_])):
        if (str(joint_right_norm_avg[0:min_][m]) != 'nan' and str(joint_left_norm_avg[0:min_][m]) != 'nan'):
            points[k,0] = joint_right_norm_avg[0:min_][m]
            points[k,1] = joint_left_norm_avg[0:min_][m]
            k+=1       
    # compute convex hull --> NOT DOING THAT ANYMORE
    #hull = ConvexHull(points)
    #area = hull.volume # dimension = 2
    alpha = 0.05
    # Create alpha shape (concave hull)
    alpha_shape = alphashape.alphashape(points.astype(float), alpha)
    exterior_coords = alpha_shape.exterior.coords.xy
    # Create a Polygon object from the coordinates
    hull_polygon = Polygon(list(zip(exterior_coords[0], exterior_coords[1])))
    # Calculate the area of the concave hull
    area = hull_polygon.area

    # compute the covariance matrix
    cov = np.cov(points.astype(float), rowvar=False)
    # compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # find the index of the maximum eigenvalue
    max_index = np.argmax(eigenvalues)
    # get the corresponding eigenvector
    principal_axis = eigenvectors[:, max_index]
    # compute cyclogram orientation
    centroid = np.mean(points, axis=0)
    slope = principal_axis[1] / principal_axis[0]
    y_intercept = centroid[1] - slope * centroid[0]
    # compute angle
    angle = -np.rad2deg(math.atan ((0-slope)/(1+0*slope)))

    # translate points
    points_translated = points - np.mean(points, axis=0)
    # rotate points
    theta = np.deg2rad(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    points_rotated = np.matmul(points_translated,R)
    # get variation along the eigen vector
    var_X = np.var(points_rotated[:,0])
    # get variation about the eigen vector
    var_Y = np.var(points_rotated[:,1])
    # compute ratio (trend symmetry)
    ratio = var_Y/var_X*100

    if (plot):
        # print
        print(joint)
        print('Area: ' + str(np.round(area,2)))
        print('Angle: ' + str(np.round(angle,2)))
        print('Ratio: ' + str(np.round(ratio,2)) +'%')
        
        fig, ax = plt.subplots(1,1,figsize= (6,6))
        # plot angles left and right according to time
        #ax.plot(joint_angles_left[0:min_], 'black', label='left')
        #ax.plot(joint_angles_right[0:min_], 'grey', label='right')
        # plot angles left and right according to time
        #ax.plot(joint_right_norm[0:min_], 'black')
        #ax.plot(joint_left_norm[0:min_], 'grey')
        # plot bilateral cyclogram 
        ax.plot(joint_right_norm_avg[0:min_], joint_left_norm_avg[0:min_], color='black', alpha=0.2)
        # plot symmetry line
        ax.plot(np.arange(max_), np.arange(max_), linestyle='dashed', color='grey')
        # plot convex hull
        #for simplex in hull.simplices:
            #plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        plt.plot(*alpha_shape.exterior.xy, 'k-') 
        # plot x-abcissa
        ax.axhline(0, linestyle = 'dashed', color='grey')
        # plot cyclogram orientation
        ax.plot(np.arange(max_), slope*np.arange(max_)+y_intercept, color='red')
        # figure parameters
        ax.set_xlabel( joint+ ' angle right [deg]'); ax.set_ylabel(joint+ ' angle left [deg]')
        #ax.set_xlim([-5,65]); ax.set_ylim([-5,65]); 
        ax.set_title(joint+' angle bilateral cyclogram')
        ax.spines[['right', 'top']].set_visible(False); ax.spines[['right', 'top']].set_visible(False); ax.spines[['right', 'top']].set_visible(False)
        ax.legend()
        fig.tight_layout()
        fig.show()
    
    # triplet of values
    values = [area, angle, ratio]
    
    # compute asymmetry
    A = compute_asymmetry(values)
    
    return A


def compute_asymmetry(subject_values):
    """
    The triplet of geometric properties of cyclograms, namely the area within the cyclogram (S), the angle between the x-axis the principal axis of inertia of the cyclogram (α), and the trend symmetry (J), can be represented by a point in a 3-D space. The ideal point for perfect symmetric gait has coordinates (0, 45, 0). By using only those three characteristics together, significant asymmetry can be identified. The asymmetry A was defined as the distance from an individual’s point to the ideal point.
    Inputs: INPUTS NON NORMALIZED THEY ARE NORMALIZED IN THIS FUNCTION!
        * subject_values (list 1x3): area (float): area of the convex hull within the cyclogram, angle (float): orientation of the cyclogram, ratio (float): the trend symmetry in %
        
    Outputs:
        * A (float): asymmetry
    """
    # recover parameters of subject and ideal symmetry
    area = subject_values[0]; angle = subject_values[1]; ratio = subject_values[2]
    area_symm = 0; angle_symm = 45; ratio_symm = 0
    
    # normalize parameters between [0,1]
    s_ratio = 1/10; a_ratio = 10
    s_area = 1/350; a_area = 350
    
    area = 1/(1+np.exp(-s_area*(area-a_area)))
    angle = (angle+5)/(90+5) # min max in [-5,90]
    ratio = 1/(1+np.exp(-s_ratio*(ratio-a_ratio)))
    
    area_symm = 1/(1+np.exp(-s_area*(area_symm-a_area)))
    angle_symm = (angle_symm+5)/(90+5) # min max in [-5,90]
    ratio_symm = 1/(1+np.exp(-s_ratio*(ratio_symm-a_ratio)))
    
    # max possible values after normalization in [0,1] (the max values are 1)
    area_max = 1
    angle_max = 1 #or 0 but same result as both 0 and 1 are at equal distance of 0.5 which is the ideal normalized angle 
    ratio_max = 1
    
    # compute asymmetry score
    d = np.sqrt( (area-area_symm)**2 + (angle - angle_symm)**2 + (ratio - ratio_symm)**2)
    d_max = np.sqrt((area_max-area_symm)**2 + (angle_max - angle_symm)**2 + (ratio_max - ratio_symm)**2)
    # normalize and express as a percentage
    A = d/d_max*100
    
    return A



def compute_asymmetry_ss_speed_three_trials(trajectories, angles, ID, plot=False, new_ss = False, T0 = False, T1 = False, T2 = False):
    """
    Compute the asymmetry for all the trials at self-selected speed.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * A_knee
        * A_hip
        * A_ankle
    """
    # recover and sort the trials in ascending order
    list_ = []
    for i in range(len(trajectories)):
        list_.append(trajectories[i][0])
    list_sorted = natsort.natsorted(list_,reverse=False)

    # store asymmetry scores for each joint
    A_knee = []; A_hip = []; A_ankle = []
    
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
                if (plot):
                    print(trajectories[j][0])
                A_knee.append(bilateral_cyclograms_single_trial(angles[j][1], trajectories[j][1], ID, 'Knee', plot))
                A_hip.append(bilateral_cyclograms_single_trial(angles[j][1], trajectories[j][1], ID, 'Hip', plot))
                A_ankle.append(bilateral_cyclograms_single_trial(angles[j][1], trajectories[j][1], ID, 'Ankle', plot))
      
    # take the mean asymmetry over the trials at the same self-selected speed 
    A_knee = np.mean(A_knee)
    A_hip = np.mean(A_hip)
    A_ankle = np.mean(A_ankle)
    
    return A_hip, A_knee, A_ankle
    
def compute_asymmetry_ss_speed_AB(angles, AB_number, plot=False):
    """
    Compute the asymmetry for all the trials at self-selected speed.
    
    Inputs:
        * trajectories (list): contains the trajectories of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * angles (list): contains the angles of all the 9 trials for the selected subject as given by download_parameters(subject, assessment) of dataloader.py 
        * ID (string): ID of the subject, e.g. "BO2ST_101"
    Outputs:
        * A_knee
        * A_hip
        * A_ankle
    """

    A_knee = bilateral_cyclograms_AB(angles, AB_number, 'Knee', plot)
    A_hip = bilateral_cyclograms_AB(angles, AB_number, 'Hip', plot)
    A_ankle = bilateral_cyclograms_AB(angles, AB_number, 'Ankle', plot)
    
    return A_hip, A_knee, A_ankle
    

def bilateral_cyclograms_AB(angles, AB_number, joint, plot=False):

    for i in range(len(angles)):
        # ss speed
        if (angles[i][0] == 'WBDS'+AB_number+'walkT05ang'):
            # extract left and right knee angles
            joint_angles_left = angles[i][1]["L"+joint+'AngleZ']
            joint_angles_right = angles[i][1]["R"+joint+'AngleZ']
            
            # get minimum len of trajectory 
            min_ = min(len(joint_angles_right), len(joint_angles_left))
            # get maximum angle between left and right leg
            max_ = max(np.nanmax(joint_angles_left), np.nanmax(joint_angles_right))

            # get the set of points
            k = 0
            for m in range(len(joint_angles_right[0:min_])):
                if (str(joint_angles_right[0:min_][m]) != 'nan' and str(joint_angles_left[0:min_][m]) != 'nan'):
                    k+=1
            points = np.empty(shape=(k, 2), dtype='object')

            k = 0
            for m in range(len(joint_angles_right[0:min_])):
                if (str(joint_angles_right[0:min_][m]) != 'nan' and str(joint_angles_left[0:min_][m]) != 'nan'):
                    points[k,0] = joint_angles_right[0:min_][m]
                    points[k,1] = joint_angles_left[0:min_][m]
                    k+=1       
            # compute convex hull --> NOT DOING THAT ANYMORE
            #hull = ConvexHull(points)
            #area = hull.volume # dimension = 2
            alpha = 0.05
                
            # Create alpha shape (concave hull)
            alpha_shape = alphashape.alphashape(points.astype(float), alpha)
            exterior_coords = alpha_shape.exterior.coords.xy
            # Create a Polygon object from the coordinates
            hull_polygon = Polygon(list(zip(exterior_coords[0], exterior_coords[1])))
            # Calculate the area of the concave hull
            area = hull_polygon.area
            
            # compute the covariance matrix
            cov = np.cov(points.astype(float), rowvar=False)
            # compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            # find the index of the maximum eigenvalue
            max_index = np.argmax(eigenvalues)
            # get the corresponding eigenvector
            principal_axis = eigenvectors[:, max_index]
            # compute cyclogram orientation
            centroid = np.mean(points, axis=0)
            slope = principal_axis[1] / principal_axis[0]
            y_intercept = centroid[1] - slope * centroid[0]
            # compute angle
            angle = np.abs(np.rad2deg(math.atan ((0-slope)/(1+0*slope))))

            # translate points
            points_translated = points - np.mean(points, axis=0)
            # rotate points
            theta = np.deg2rad(angle)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            points_rotated = np.matmul(points_translated,R)
            # get variation along the eigen vector
            var_X = np.var(points_rotated[:,0])
            # get variation about the eigen vector
            var_Y = np.var(points_rotated[:,1])
            # compute ratio (trend symmetry)
            ratio = var_Y/var_X*100

            if (plot):
                
                # print
                print('Area: ' + str(np.round(area,2)))
                print('Angle: ' + str(np.round(angle,2)))
                print('Ratio: ' + str(np.round(ratio,2)) +'%')
                
                fig, ax = plt.subplots(1,1,figsize= (5,5))
                # plot bilateral cyclogram 
                ax.plot(joint_angles_right[0:min_], joint_angles_left[0:min_], color='black', alpha=0.2)
                # plot symmetry line
                ax.plot(np.arange(max_), np.arange(max_), linestyle='dashed', color='grey')
                # plot convex hull
                #for simplex in hull.simplices:
                    #plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
                plt.plot(*alpha_shape.exterior.xy, 'k-') 
                # plot x-abcissa
                ax.axhline(0, linestyle = 'dashed', color='grey')
                # plot cyclogram orientation
                ax.plot(np.arange(max_), slope*np.arange(max_)+y_intercept, color='red')
                # figure parameters
                ax.set_xlabel(joint+' angle right [deg]'); ax.set_ylabel(joint+' angle left [deg]')
                #ax.set_xlim([-5,65]); ax.set_ylim([-5,65]); 
                ax.set_title(joint+' angle bilateral cyclogram')
                ax.spines[['right', 'top']].set_visible(False)
                fig.tight_layout()
                fig.show()
            
            # triplet of parameters
            values = [area, angle, ratio]
            
            # compute asymmetry score
            A = compute_asymmetry(values)
            
    return A

####################################################################################################################################
#DON'T USE THE FUNCTIONS BELOW THEY ARE WRONG!!!!!!!!!!
####################################################################################################################################

def compute_asymmetry_ss_plot_AB(trajectories, angles, trajectories_AB, angles_AB, plot=False):
    
    """
    NORMALIZATION HERE IS WRONG!!!!!!
    """
    
    trajectories_BL = trajectories[0]; trajectories_T0 = trajectories[1]; trajectories_T1 = trajectories[2]; trajectories_T2 = trajectories[3]; trajectories_F1 = trajectories[4]; trajectories_F4 = trajectories[5]; trajectories_F8 = trajectories[6]
    angles_BL = angles[0]; angles_T0 = angles[1]; angles_T1 = angles[2]; angles_T2 = angles[3]; angles_F1 = angles[4]; angles_F4 = angles[5]; angles_F8 = angles[6]
    
    if (plot):
        fig, ax = plt.subplots(1,figsize= (7,5))
        
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    A_ss_BL = []; A_ss_T0 = []; A_ss_T1 = []; A_ss_T2 = []; A_ss_F1 = []; A_ss_F4 = []; A_ss_F8 = []

    # create an Empty DataFrame object
    bilateral_cyclo_df = pd.DataFrame()

    for i in range(len(IDs)):
        A_ss_BL.append(bilateral_cyclograms(angles_BL[i], trajectories_BL[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_T0.append(bilateral_cyclograms(angles_T0[i], trajectories_T0[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_T1.append(bilateral_cyclograms(angles_T1[i], trajectories_T1[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_T2.append(bilateral_cyclograms(angles_T2[i], trajectories_T2[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_F1.append(bilateral_cyclograms(angles_F1[i], trajectories_F1[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_F4.append(bilateral_cyclograms(angles_F4[i], trajectories_F4[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_F8.append(bilateral_cyclograms(angles_F8[i], trajectories_F8[i], IDs[i], plot=False)[0]) #[0] for ss

    # append columns to the empty DataFrame
    bilateral_cyclo_df['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']  
    bilateral_cyclo_df['Areas'] = [0, np.mean(A_ss_BL[0][0]), np.mean(A_ss_BL[1][0]), np.mean(A_ss_BL[2][0]), np.mean(A_ss_BL[3][0]), np.mean(A_ss_BL[4][0]), np.mean(A_ss_BL[5][0])]
    bilateral_cyclo_df['Angles'] = [45, np.mean(A_ss_BL[0][1]), np.mean(A_ss_BL[1][1]), np.mean(A_ss_BL[2][1]), np.mean(A_ss_BL[3][1]), np.mean(A_ss_BL[4][1]), np.mean(A_ss_BL[5][1])]
    bilateral_cyclo_df['Ratios'] = [0, np.mean(A_ss_BL[0][2]), np.mean(A_ss_BL[1][2]), np.mean(A_ss_BL[2][2]), np.mean(A_ss_BL[3][2]), np.mean(A_ss_BL[4][2]), np.mean(A_ss_BL[5][2])]
    bilateral_cyclo_df['Time'] = ['NA', 'BL', 'BL', 'BL', 'BL', 'BL', 'BL']

    dict_T0 = { 'ID':['BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106'],
                'Areas':[np.mean(A_ss_T0[0][0]), np.mean(A_ss_T0[1][0]), np.mean(A_ss_T0[2][0]), np.mean(A_ss_T0[3][0]), np.mean(A_ss_T0[4][0]), np.mean(A_ss_T0[5][0])],
                'Angles':[np.mean(A_ss_T0[0][1]), np.mean(A_ss_T0[1][1]), np.mean(A_ss_T0[2][1]), np.mean(A_ss_T0[3][1]), np.mean(A_ss_T0[4][1]), np.mean(A_ss_T0[5][1])],
                'Ratios':[np.mean(A_ss_T0[0][2]), np.mean(A_ss_T0[1][2]), np.mean(A_ss_T0[2][2]), np.mean(A_ss_T0[3][2]), np.mean(A_ss_T0[4][2]), np.mean(A_ss_T0[5][2])],
                'Time': ['T0', 'T0', 'T0', 'T0', 'T0', 'T0']
              }
    
    dict_T1 = { 'ID':['BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106'],
                'Areas':[np.mean(A_ss_T1[0][0]), np.mean(A_ss_T1[1][0]), np.mean(A_ss_T1[2][0]), np.mean(A_ss_T1[3][0]), np.mean(A_ss_T1[4][0]), np.mean(A_ss_T1[5][0])],
                'Angles':[np.mean(A_ss_T1[0][1]), np.mean(A_ss_T1[1][1]), np.mean(A_ss_T1[2][1]), np.mean(A_ss_T1[3][1]), np.mean(A_ss_T1[4][1]), np.mean(A_ss_T1[5][1])],
                'Ratios':[np.mean(A_ss_T1[0][2]), np.mean(A_ss_T1[1][2]), np.mean(A_ss_T1[2][2]), np.mean(A_ss_T1[3][2]), np.mean(A_ss_T1[4][2]), np.mean(A_ss_T1[5][2])],
                'Time': ['T1', 'T1', 'T1', 'T1', 'T1', 'T1']
              }
    
    dict_T2 = { 'ID':['BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106'],  
                'Areas':[np.mean(A_ss_T2[0][0]), np.mean(A_ss_T2[1][0]), np.mean(A_ss_T2[2][0]), np.mean(A_ss_T2[3][0]), np.mean(A_ss_T2[4][0]), np.mean(A_ss_T2[5][0])],
                'Angles':[np.mean(A_ss_T2[0][1]), np.mean(A_ss_T2[1][1]), np.mean(A_ss_T2[2][1]), np.mean(A_ss_T2[3][1]), np.mean(A_ss_T2[4][1]), np.mean(A_ss_T2[5][1])],
                'Ratios':[np.mean(A_ss_T2[0][2]), np.mean(A_ss_T2[1][2]), np.mean(A_ss_T2[2][2]), np.mean(A_ss_T2[3][2]), np.mean(A_ss_T2[4][2]), np.mean(A_ss_T2[5][2])],
                'Time': ['T2', 'T2', 'T2', 'T2', 'T2', 'T2']
              }
    
    dict_F1 = { 'ID':['BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106'],  
                'Areas':[np.mean(A_ss_F1[0][0]), np.mean(A_ss_F1[1][0]), np.mean(A_ss_F1[2][0]), np.mean(A_ss_F1[3][0]), np.mean(A_ss_F1[4][0]), np.mean(A_ss_F1[5][0])],
                'Angles':[np.mean(A_ss_F1[0][1]), np.mean(A_ss_F1[1][1]), np.mean(A_ss_F1[2][1]), np.mean(A_ss_F1[3][1]), np.mean(A_ss_F1[4][1]), np.mean(A_ss_F1[5][1])],
                'Ratios':[np.mean(A_ss_F1[0][2]), np.mean(A_ss_F1[1][2]), np.mean(A_ss_F1[2][2]), np.mean(A_ss_F1[3][2]), np.mean(A_ss_F1[4][2]), np.mean(A_ss_F1[5][2])],
                'Time': ['F1', 'F1', 'F1', 'F1', 'F1', 'F1']
              }
    
    dict_F4 = { 'ID':['BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106'],  
                'Areas':[np.mean(A_ss_F4[0][0]), np.mean(A_ss_F4[1][0]), np.mean(A_ss_F4[2][0]), np.mean(A_ss_F4[3][0]), np.mean(A_ss_F4[4][0]), np.mean(A_ss_F4[5][0])],
                'Angles':[np.mean(A_ss_F4[0][1]), np.mean(A_ss_F4[1][1]), np.mean(A_ss_F4[2][1]), np.mean(A_ss_F4[3][1]), np.mean(A_ss_F4[4][1]), np.mean(A_ss_F4[5][1])],
                'Ratios':[np.mean(A_ss_F4[0][2]), np.mean(A_ss_F4[1][2]), np.mean(A_ss_F4[2][2]), np.mean(A_ss_F4[3][2]), np.mean(A_ss_F4[4][2]), np.mean(A_ss_F4[5][2])],
                'Time': ['F4', 'F4', 'F4', 'F4', 'F4', 'F4']
              }
        
    dict_F8 = { 'ID':['BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106'],  
                'Areas':[np.mean(A_ss_F8[0][0]), np.mean(A_ss_F8[1][0]), np.mean(A_ss_F8[2][0]), np.mean(A_ss_F8[3][0]), np.mean(A_ss_F8[4][0]), np.mean(A_ss_F8[5][0])],
                'Angles':[np.mean(A_ss_F8[0][1]), np.mean(A_ss_F8[1][1]), np.mean(A_ss_F8[2][1]), np.mean(A_ss_F8[3][1]), np.mean(A_ss_F8[4][1]), np.mean(A_ss_F8[5][1])],
                'Ratios':[np.mean(A_ss_F8[0][2]), np.mean(A_ss_F8[1][2]), np.mean(A_ss_F8[2][2]), np.mean(A_ss_F8[3][2]), np.mean(A_ss_F8[4][2]), np.mean(A_ss_F8[5][2])],
                'Time': ['F8', 'F8', 'F8', 'F8', 'F8', 'F8']
              }    
    
    df_dict_T0 = pd.DataFrame(dict_T0)
    df_dict_T1 = pd.DataFrame(dict_T1)
    df_dict_T2 = pd.DataFrame(dict_T2)
    df_dict_F1 = pd.DataFrame(dict_F1)
    df_dict_F4 = pd.DataFrame(dict_F4)
    df_dict_F8 = pd.DataFrame(dict_F8)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_T0], ignore_index = True)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_T1], ignore_index = True)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_T2], ignore_index = True)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_F1], ignore_index = True)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_F4], ignore_index = True)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_F8], ignore_index = True)

    # compute and add AB data to the df
    AB_03_A = bilateral_cyclograms_AB(angles_AB[0], '03')
    AB_11_A = bilateral_cyclograms_AB(angles_AB[1], '11')
    AB_21_A = bilateral_cyclograms_AB(angles_AB[2], '21')
    AB_38_A = bilateral_cyclograms_AB(angles_AB[3], '38')
    AB_42_A = bilateral_cyclograms_AB(angles_AB[4], '42')
    dict_AB = {'ID':['AB03', 'AB11', 'AB21', 'AB38', 'AB42'],
            'Areas':[AB_03_A[0], AB_11_A[0], AB_21_A[0], AB_38_A[0], AB_42_A[0]],
            'Angles':[AB_03_A[1], AB_11_A[1], AB_21_A[1], AB_38_A[1], AB_42_A[1]],
            'Ratios':[AB_03_A[2], AB_11_A[2], AB_21_A[2], AB_38_A[2], AB_42_A[2]],
            'Time': ['AB', 'AB', 'AB', 'AB', 'AB']
           }
    df_dict_AB = pd.DataFrame(dict_AB)
    bilateral_cyclo_df = pd.concat([bilateral_cyclo_df, df_dict_AB], ignore_index = True)

    # normalize columns (except ID and Time)
    scaler = MinMaxScaler()
    scaler.fit(bilateral_cyclo_df.loc[:, ~bilateral_cyclo_df.columns.isin(['ID', 'Time'])])
    scaled = scaler.fit_transform(bilateral_cyclo_df.loc[:, ~bilateral_cyclo_df.columns.isin(['ID', 'Time'])])
    scaled_df = pd.DataFrame(scaled, columns=bilateral_cyclo_df.loc[:, ~bilateral_cyclo_df.columns.isin(['ID', 'Time'])].columns)
    scaled_df['ID'] = bilateral_cyclo_df['ID']
    scaled_df['Time'] = bilateral_cyclo_df['Time']

    # compute asymmetry
    scaled_df_BL = scaled_df.loc[scaled_df['Time'].isin(['BL', 'AB'])]
    scaled_df_T0 = scaled_df.loc[scaled_df['Time'].isin(['T0', 'AB'])]
    scaled_df_T1 = scaled_df.loc[scaled_df['Time'].isin(['T1', 'AB'])]
    scaled_df_T2 = scaled_df.loc[scaled_df['Time'].isin(['T2', 'AB'])]
    scaled_df_F1 = scaled_df.loc[scaled_df['Time'].isin(['F1', 'AB'])]
    scaled_df_F4 = scaled_df.loc[scaled_df['Time'].isin(['F4', 'AB'])]
    scaled_df_F8 = scaled_df.loc[scaled_df['Time'].isin(['F8', 'AB'])]
    scaled_df_ideal = scaled_df.loc[scaled_df['ID'] == 'Ideal Symmetry']
    
    # compute asymmetry
    Asymmetry_BL = []; Asymmetry_T0 = []; Asymmetry_T1 = []; Asymmetry_T2 = []; Asymmetry_F1 = []; Asymmetry_F4 = []; Asymmetry_F8 = []

    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106", "AB03", 'AB11', 'AB21', 'AB38', 'AB42']

    for i in range(len(IDs)):
        A_BL = compute_asymmetry(scaled_df_BL.loc[scaled_df_BL['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_T0 = compute_asymmetry(scaled_df_T0.loc[scaled_df_T0['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_T1 = compute_asymmetry(scaled_df_T1.loc[scaled_df_T1['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_T2 = compute_asymmetry(scaled_df_T2.loc[scaled_df_T2['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_F1 = compute_asymmetry(scaled_df_F1.loc[scaled_df_F1['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_F4 = compute_asymmetry(scaled_df_F4.loc[scaled_df_F4['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_F8 = compute_asymmetry(scaled_df_F8.loc[scaled_df_F8['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_ideal[['Areas', 'Angles', 'Ratios']].values.squeeze())
        Asymmetry_BL.append(A_BL)
        Asymmetry_T0.append(A_T0)
        Asymmetry_T1.append(A_T1)
        Asymmetry_T2.append(A_T2)
        Asymmetry_F1.append(A_F1)
        Asymmetry_F4.append(A_F4)
        Asymmetry_F8.append(A_F8)

        if (plot):
            data = [A_BL, A_T0, A_T1, A_T2, A_F1, A_F4, A_F8]
            ax.plot(["BL","T0", "T1", "T2", "F1", "F4", "F8"], data, label=IDs[i], color="C"+str(i))
            ax.plot(["BL","T0", "T1", "T2", "F1", "F4", "F8"], data, 'o',color="C"+str(i))
            ax.legend(title = "Participants")
            ax.spines[['right', 'top']].set_visible(False)

            # figure parameters
            ax.set_ylabel('Asymmetry')
            ax.set_xlabel('Time')
            fig.show()
            fig.tight_layout();

    return Asymmetry_BL, Asymmetry_T0, Asymmetry_T1, Asymmetry_T2, Asymmetry_F1, Asymmetry_F4, Asymmetry_F8


def compute_asymmetry_ss_plot(trajectories_BL, angles_BL, trajectories_T0, angles_T0, trajectories_T2, angles_T2, IDs, plot=False):
    """
    NORMALIZATION HERE IS WRONG!!!!!!
    """
    

    if (plot):
        fig, ax = plt.subplots(1,figsize= (7,5))
    A_ss_BL = []; A_ss_T0 = []; A_ss_T2 = []
    
    # create an Empty DataFrame object
    bilateral_cyclo_df_BL = pd.DataFrame()
    bilateral_cyclo_df_T0 = pd.DataFrame()
    bilateral_cyclo_df_T2 = pd.DataFrame()

    for i in range(len(IDs)):
        A_ss_BL.append(bilateral_cyclograms(angles_BL[i], trajectories_BL[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_T0.append(bilateral_cyclograms(angles_T0[i], trajectories_T0[i], IDs[i], plot=False)[0]) #[0] for ss
        A_ss_T2.append(bilateral_cyclograms(angles_T2[i], trajectories_T2[i], IDs[i], plot=False)[0]) #[0] for ss

    # append columns to the empty DataFrame
    bilateral_cyclo_df_BL['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']  
    bilateral_cyclo_df_BL['Areas'] = [0, np.mean(A_ss_BL[0][0]), np.mean(A_ss_BL[1][0]), np.mean(A_ss_BL[2][0]), np.mean(A_ss_BL[3][0]), np.mean(A_ss_BL[4][0]), np.mean(A_ss_BL[5][0])]
    bilateral_cyclo_df_BL['Angles'] = [45, np.mean(A_ss_BL[0][1]), np.mean(A_ss_BL[1][1]), np.mean(A_ss_BL[2][1]), np.mean(A_ss_BL[3][1]), np.mean(A_ss_BL[4][1]), np.mean(A_ss_BL[5][1])]
    bilateral_cyclo_df_BL['Ratios'] = [0, np.mean(A_ss_BL[0][2]), np.mean(A_ss_BL[1][2]), np.mean(A_ss_BL[2][2]), np.mean(A_ss_BL[3][2]), np.mean(A_ss_BL[4][2]), np.mean(A_ss_BL[5][2])]
    #print(bilateral_cyclo_df_BL)
    bilateral_cyclo_df_T0['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']  
    bilateral_cyclo_df_T0['Areas'] = [0, np.mean(A_ss_T0[0][0]), np.mean(A_ss_T0[1][0]), np.mean(A_ss_T0[2][0]), np.mean(A_ss_T0[3][0]), np.mean(A_ss_T0[4][0]), np.mean(A_ss_T0[5][0])]
    bilateral_cyclo_df_T0['Angles'] = [45, np.mean(A_ss_T0[0][1]), np.mean(A_ss_T0[1][1]), np.mean(A_ss_T0[2][1]), np.mean(A_ss_T0[3][1]), np.mean(A_ss_T0[4][1]), np.mean(A_ss_T0[5][1])]
    bilateral_cyclo_df_T0['Ratios'] = [0, np.mean(A_ss_T0[0][2]), np.mean(A_ss_T0[1][2]), np.mean(A_ss_T0[2][2]), np.mean(A_ss_T0[3][2]), np.mean(A_ss_T0[4][2]), np.mean(A_ss_T0[5][2])]
    # T2
    bilateral_cyclo_df_T2['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']  
    bilateral_cyclo_df_T2['Areas'] = [0, np.mean(A_ss_T2[0][0]), np.mean(A_ss_T2[1][0]), np.mean(A_ss_T2[2][0]), np.mean(A_ss_T2[3][0]), np.mean(A_ss_T2[4][0]), np.mean(A_ss_T2[5][0])]
    bilateral_cyclo_df_T2['Angles'] = [45, np.mean(A_ss_T2[0][1]), np.mean(A_ss_T2[1][1]), np.mean(A_ss_T2[2][1]), np.mean(A_ss_T2[3][1]), np.mean(A_ss_T2[4][1]), np.mean(A_ss_T2[5][1])]
    bilateral_cyclo_df_T2['Ratios'] = [0, np.mean(A_ss_T2[0][2]), np.mean(A_ss_T2[1][2]), np.mean(A_ss_T2[2][2]), np.mean(A_ss_T2[3][2]), np.mean(A_ss_T2[4][2]), np.mean(A_ss_T2[5][2])]


    # normalize columns (except ID)
    scaler = MinMaxScaler()
    scaler.fit(bilateral_cyclo_df_BL.loc[:, bilateral_cyclo_df_BL.columns != 'ID'])
    scaled_BL = scaler.fit_transform(bilateral_cyclo_df_BL.loc[:, bilateral_cyclo_df_BL.columns != 'ID'])
    scaled_df_BL = pd.DataFrame(scaled_BL, columns=bilateral_cyclo_df_BL.loc[:, bilateral_cyclo_df_BL.columns != 'ID'].columns)
    scaled_df_BL['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']
    #print(scaled_df_BL)
    scaler.fit(bilateral_cyclo_df_T0.loc[:, bilateral_cyclo_df_T0.columns != 'ID'])
    scaled_T0 = scaler.fit_transform(bilateral_cyclo_df_T0.loc[:, bilateral_cyclo_df_T0.columns != 'ID'])
    scaled_df_T0 = pd.DataFrame(scaled_T0, columns=bilateral_cyclo_df_T0.loc[:, bilateral_cyclo_df_T0.columns != 'ID'].columns)
    scaled_df_T0['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']
    #T2
    scaler.fit(bilateral_cyclo_df_T2.loc[:, bilateral_cyclo_df_T2.columns != 'ID'])
    scaled_T2 = scaler.fit_transform(bilateral_cyclo_df_T2.loc[:, bilateral_cyclo_df_T2.columns != 'ID'])
    scaled_df_T2 = pd.DataFrame(scaled_T2, columns=bilateral_cyclo_df_T2.loc[:, bilateral_cyclo_df_T2.columns != 'ID'].columns)
    scaled_df_T2['ID'] = ['Ideal Symmetry', 'BO2ST_101', 'BO2ST_102', 'BO2ST_103', 'BO2ST_104', 'BO2ST_105', 'BO2ST_106']

    # compute asymmetry
    Asymmetry_BL = []; Asymmetry_T0 = []; Asymmetry_T2 = []
    for i in range(len(IDs)):
        A_BL = compute_asymmetry(scaled_df_BL.loc[scaled_df_BL['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_BL.loc[scaled_df_BL['ID'] == 'Ideal Symmetry'][['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_T0 = compute_asymmetry(scaled_df_T0.loc[scaled_df_T0['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_T0.loc[scaled_df_T0['ID'] == 'Ideal Symmetry'][['Areas', 'Angles', 'Ratios']].values.squeeze())
        A_T2 = compute_asymmetry(scaled_df_T2.loc[scaled_df_T2['ID'] == IDs[i]][['Areas', 'Angles', 'Ratios']].values.squeeze(), scaled_df_T2.loc[scaled_df_T2['ID'] == 'Ideal Symmetry'][['Areas', 'Angles', 'Ratios']].values.squeeze())
        Asymmetry_BL.append(A_BL)
        Asymmetry_T0.append(A_T0)
        Asymmetry_T2.append(A_T2)
        
        if (plot):
            data = [A_BL, A_T0, A_T2]
            ax.plot(["BL","T0","T2"],data, label=IDs[i], color="C"+str(i))
            ax.plot(["BL","T0","T2"],data, 'o',color="C"+str(i))
            ax.legend(title = "Participants")
            ax.spines[['right', 'top']].set_visible(False)

            # figure parameters
            ax.set_ylabel('Asymmetry')
            ax.set_xlabel('Time')
            fig.show()
            fig.tight_layout();

    return Asymmetry_BL, Asymmetry_T0, Asymmetry_T2

