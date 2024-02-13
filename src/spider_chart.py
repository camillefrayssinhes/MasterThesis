import numpy as np
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import natsort
import math
import pathlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalize_functional_walking(ID, time):
    """
    Normalize the 10MWT speed, the 6MWT speed, the ss speed, the TUG time and the LEMS score of the participant between 0 and 100.
    Input:
        * ID(string): e.g. "BO2ST_101"
        * time (string): time point e.g. "BL" or "T0
    Output:
        * _10MWT_speed, _6MWT_speed, ss_speed, TUG, LEMS
    """
    # read file
    file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    overground_gait_assessments_xcl = pd.read_excel(file_name, header = [0], index_col = [0])
    file_name_ss_speeds = ("BO2STTrial/SS_speeds.xls")
    ss_speeds_xcl = pd.read_excel(file_name_ss_speeds, header = [0], index_col = [0])
    file_name_LEMS = ("BO2STTrial/LEMS.xlsx")
    LEMS_xcl = pd.read_excel(file_name_LEMS, header = [0], index_col = [0])
    
    # match control
    control_file_name = ("BO2STTrial/matched_subjects.xlsx")
    control_xcl = pd.read_excel(control_file_name, header = [0], index_col = [0])
    ID_AB = control_xcl.loc[ID].Matched_control
    control_file_name_info = ("BO2STTrial/WBDSascii/WBDSinfo.xlsx")
    control_info_xcl = pd.read_excel(control_file_name_info, header = [0], index_col = [0])
    
    # read physiological data
    weight = overground_gait_assessments_xcl.loc[ID]['Weight'] #kg
    height = overground_gait_assessments_xcl.loc[ID]['Height'] #cm
    age = overground_gait_assessments_xcl.loc[ID]['Age']
    sex = overground_gait_assessments_xcl.loc[ID]['Sex']
    
    # compute normative reference value (NRF) for this subject
    if (sex == 'F'):
        # 6MWT
        _6MWT_max = 2.11*height - 2.29*weight - 5.78*age + 667 # distance
        _6MWT_max = _6MWT_max/(6*60) # speed
        # 10MWT
        if (20<=age<30):
            _10MWT_max = 1.502*10**(-2)*height
        elif (30<=age<40):
            _10MWT_max = 1.428*10**(-2)*height
        elif (40<=age<50):
            _10MWT_max = 1.304*10**(-2)*height
        elif (50<=age<60):
            _10MWT_max = 1.243*10**(-2)*height
        elif (60<=age<70):
            _10MWT_max = 1.107*10**(-2)*height
        elif (70<=age<80):
            _10MWT_max = 1.110*10**(-2)*height
        
    elif (sex == 'M'):   
        # 6MWT
        _6MWT_max = 7.57*height - 5.02*age - 1.76*weight - 309 # distance
        _6MWT_max = _6MWT_max/(6*60) # speed
        # 10MWT
        if (20<=age<30):
            _10MWT_max = 1.431*10**(-2)*height
        elif (30<=age<40):
            _10MWT_max = 1.396*10**(-2)*height
        elif (40<=age<50):
            _10MWT_max = 1.395*10**(-2)*height
        elif (50<=age<60):
            _10MWT_max = 1.182*10**(-2)*height
        elif (60<=age<70):
            _10MWT_max = 1.104*10**(-2)*height
        elif (70<=age<80):
            _10MWT_max = 1.192*10**(-2)*height
    
    # TUG
    if (20<=age<30):
        TUG_max = 8.57
    elif (30<=age<40):
        TUG_max = 8.56
    elif (40<=age<50):
        TUG_max = 8.86
    elif (age >=50):
        TUG_max = 9.90
        
    TUG_max = 1/TUG_max # take the inverse 
    # ss speed of control
    ss_max = float(control_info_xcl[control_info_xcl.FileName=='WBDS'+ID_AB[-2:]+'walkT05.c3d']['GaitSpeed(m/s)'])

    # 10MWT speed
    _10MWT_speed = overground_gait_assessments_xcl.loc[ID]["10MWT_speed_"+time]
    _10MWT_speed_norm = _10MWT_speed/_10MWT_max*100
    _10MWT_speed_AB = _10MWT_max/_10MWT_max*100
    
    # 6MWT speed
    _6MWT_speed = overground_gait_assessments_xcl.loc[ID]["6MWT_speed_"+time]
    _6MWT_speed_norm = _6MWT_speed/_6MWT_max*100
    _6MWT_speed_AB = _6MWT_max/_6MWT_max*100
    
    # ss speed
    ss_speed = ss_speeds_xcl.loc[ID][time+"_SS_speed"]
    ss_speed_norm = ss_speed/ss_max*100
    ss_speed_AB = ss_max/ss_max*100
    
    # TUG time
    TUG = 1/overground_gait_assessments_xcl.loc[ID]["TUG_time_"+time]
    TUG_norm = TUG/TUG_max*100
    TUG_AB = TUG_max/TUG_max*100
    
    # LEMS
    LEMS_max = 25
    LEMS = LEMS_xcl.loc[ID]["LEMS_"+time]
    LEMS_norm = LEMS/LEMS_max*100
    LEMS_AB = LEMS_max/LEMS_max*100
    
    # construct pairs: value of the participant at time point and normative value of the matched AB control
    _10MWT_speed = (_10MWT_speed_norm, _10MWT_speed_AB)
    _6MWT_speed = (_6MWT_speed_norm, _6MWT_speed_AB)
    ss_speed = (ss_speed_norm, ss_speed_AB)
    TUG = (TUG_norm, TUG_AB)
    LEMS = (LEMS_norm, LEMS_AB)
    
    return _10MWT_speed, _6MWT_speed, ss_speed, TUG, LEMS


def normalize_kinetics(ID, time, new_ss=False):
    """
    Normalize the kinetic parameters (fore-aft GRF transition point, braking area, propulsion area) of the participant between 0 and 100.
    Input:
        * ID(string): e.g. "BO2ST_101"
        * time (string): time point e.g. "BL" or "T0"
    Output:
        * tp, area_braking, area_propul
    """
    
    # read files
    if (new_ss):
        file_name_grf = ("BO2STTrial/fore_aft_grf_new_ss.xlsx")
    else:
        file_name_grf = ("BO2STTrial/fore_aft_grf.xlsx")
    grf_xcl = pd.read_excel(file_name_grf, header = [0], index_col = [0])
    file_name_grf_AB = ("BO2STTrial/fore_aft_grf_AB.xlsx")
    grf_AB_xcl = pd.read_excel(file_name_grf_AB, header = [0], index_col = [0])
    
    # match control
    control_file_name = ("BO2STTrial/matched_subjects.xlsx")
    control_xcl = pd.read_excel(control_file_name, header = [0], index_col = [0])
    ID_AB = control_xcl.loc[ID].Matched_control
    
    # compute transition point
    tp_max = grf_AB_xcl.loc[ID_AB]["tp_"+time]
    tp = grf_xcl.loc[ID]["tp_"+time]
    tp_norm = tp/tp_max*100
    tp_AB = tp_max/tp_max*100
    
    # compute braking area
    area_braking_max = grf_AB_xcl.loc[ID_AB]["area_braking_"+time]
    area_braking = grf_xcl.loc[ID]["area_braking_"+time]
    area_braking_norm = area_braking/area_braking_max*100
    area_braking_AB = area_braking_max/area_braking_max*100
    
    # compute propulsion area
    area_propul_max = grf_AB_xcl.loc[ID_AB]["area_propul_"+time]
    area_propul = grf_xcl.loc[ID]["area_propul_"+time]
    area_propul_norm = area_propul/area_propul_max*100
    area_propul_AB = area_propul_max/area_propul_max*100
    
    # construct pairs: value of the participant at time point and normative value of the matched AB control
    tp = (tp_norm, tp_AB)
    area_braking = (area_braking_norm, area_braking_AB)
    area_propul = (area_propul_norm, area_propul_AB)
    
    return tp, area_braking, area_propul


def normalize_variability(ID, side, time, new_ss=False):
    """
    Normalize the variability parameters (EV, Cov step length, Cov step width) of the participant between 0 and 100.
    Input:
        * ID(string): e.g. "BO2ST_101"
        * side(string): MA side e.g. 'L' or 'R'
        * time (string): time point e.g. "BL" or "T0"
    Output:
        * EV, CoV_step_length, CoV_step_width
    """
    # read files
    # EV
    if (new_ss):
        file_name_EV = ("BO2STTrial/EV_new_ss.xlsx")
    else:
        file_name_EV = ("BO2STTrial/walking_mechanics_performances.xlsx")
    EV_xcl = pd.read_excel(file_name_EV, header = [0], index_col = [0])
    file_name_EV_AB = ("BO2STTrial/walking_mechanics_performances_control.xlsx")
    EV_AB_xcl = pd.read_excel(file_name_EV_AB, header = [0], index_col = [0])
    # Cov step length
    if (new_ss):
        file_name_CoV_step_length = ("BO2STTrial/coeff_var_new_ss.xlsx")
    else:
        file_name_CoV_step_length = ("BO2STTrial/coeff_var.xlsx")
    CoV_step_length_xcl = pd.read_excel(file_name_CoV_step_length, header = [0], index_col = [0])
    file_name_CoV_step_length_AB = ("BO2STTrial/coeff_var_AB.xlsx")
    CoV_step_length_AB_xcl = pd.read_excel(file_name_CoV_step_length_AB, header = [0], index_col = [0])
    # Cov step width (it was my definition of stride width)
    if (new_ss):
        file_name_CoV_step_width = ("BO2STTrial/coeff_var_stride_width_new_ss.xlsx")
    else:
        file_name_CoV_step_width = ("BO2STTrial/coeff_var_stride_width.xlsx")
    CoV_step_width_xcl = pd.read_excel(file_name_CoV_step_width, header = [0], index_col = [0])
    file_name_CoV_step_width_AB = ("BO2STTrial/coeff_var_stride_width_AB.xlsx")
    CoV_step_width_AB_xcl = pd.read_excel(file_name_CoV_step_width_AB, header = [0], index_col = [0])
    
    # match control
    control_file_name = ("BO2STTrial/matched_subjects.xlsx")
    control_xcl = pd.read_excel(control_file_name, header = [0], index_col = [0])
    ID_AB = control_xcl.loc[ID].Matched_control
    
    # normalize values
    EV_max = 1/EV_AB_xcl.loc[ID_AB]["EV_"+time]
    EV = 1/EV_xcl.loc[ID]["EV_"+time]
    EV_norm = EV/EV_max*100
    EV_AB = EV_max/EV_max*100
    
    CoV_step_length_max = 1/CoV_step_length_AB_xcl.loc[ID_AB]["coeff_var_"+side+'_'+time]
    CoV_step_length = 1/CoV_step_length_xcl.loc[ID]["coeff_var_"+side+'_'+time]
    CoV_step_length_norm = CoV_step_length/CoV_step_length_max*100
    CoV_step_length_AB = CoV_step_length_max/CoV_step_length_max*100
    
    CoV_step_width_max = 1/CoV_step_width_AB_xcl.loc[ID_AB]["coeff_var_"+time]
    CoV_step_width = 1/CoV_step_width_xcl.loc[ID]["coeff_var_"+time]
    CoV_step_width_norm = CoV_step_width/CoV_step_width_max*100
    CoV_step_width_AB = CoV_step_width_max/CoV_step_width_max*100
    
    # construct pairs: value of the participant at time point and normative value of the matched AB control
    EV = (EV_norm, EV_AB)
    CoV_step_length = (CoV_step_length_norm, CoV_step_length_AB)
    CoV_step_width = (CoV_step_width_norm, CoV_step_width_AB)
    
    return EV, CoV_step_length, CoV_step_width


def normalize_symmetry(ID, time, new_ss=False):
    """
    Normalize the symmetry parameters (A, SI) of the participant between 0 and 100.
    Input:
        * ID(string): e.g. "BO2ST_101"
        * time (string): time point e.g. "BL" or "T0"
    Output:
        * A_knee, A_hip, A_ankle
    """
    # read files
    if (new_ss):
        file_name_A = ("BO2STTrial/A_scores_new_ss.xlsx")
    else:
        file_name_A = ("BO2STTrial/A_scores.xlsx")
    A_xcl = pd.read_excel(file_name_A, header = [0], index_col = [0])
    file_name_A_AB = ("BO2STTrial/A_scores_AB.xlsx")
    A_AB_xcl = pd.read_excel(file_name_A_AB, header = [0], index_col = [0])
    
    # match control
    control_file_name = ("BO2STTrial/matched_subjects.xlsx")
    control_xcl = pd.read_excel(control_file_name, header = [0], index_col = [0])
    ID_AB = control_xcl.loc[ID].Matched_control
    
    # compute A scores
    A_max_knee = 1/A_AB_xcl.loc[ID_AB]["A_knee_"+time]; A_max_hip = 1/A_AB_xcl.loc[ID_AB]["A_hip_"+time]; A_max_ankle = 1/A_AB_xcl.loc[ID_AB]["A_ankle_"+time]
    A_knee = 1/A_xcl.loc[ID]["A_knee_"+time]
    A_knee_norm = A_knee/A_max_knee*100
    A_knee_AB = A_max_knee/A_max_knee*100
    A_hip = 1/A_xcl.loc[ID]["A_hip_"+time]
    A_hip_norm = A_hip/A_max_hip*100
    A_hip_AB = A_max_hip/A_max_hip*100
    A_ankle = 1/A_xcl.loc[ID]["A_ankle_"+time]
    A_ankle_norm = A_ankle/A_max_ankle*100
    A_ankle_AB = A_max_ankle/A_max_ankle*100

    
    # construct pairs: value of the participant at time point and normative value of the matched AB control
    A_knee = (A_knee_norm, A_knee_AB)
    A_hip = (A_hip_norm, A_hip_AB)
    A_ankle = (A_ankle_norm, A_ankle_AB)
    
    return A_knee, A_hip, A_ankle
    
def normalize_spatiotemporal(ID, side, time, new_ss=False):
    """
    Normalize the spatiotemporal parameters (stance phase, cadence, step length, step width) of the participant between 0 and 100.
    Input:
        * ID(string): e.g. "BO2ST_101"
        * time (string): time point e.g. "BL" or "T0"
    Output:
        * stance, cadence, step length, step width 
    """ 
    
    # read files
    if (new_ss):
        file_name_spatial = ("BO2STTrial/step_length_width_cadence_new_ss.xlsx")
    else:
        file_name_spatial = ("BO2STTrial/step_length_width_cadence.xlsx")
    spatial_xcl = pd.read_excel(file_name_spatial, header = [0], index_col = [0])
    file_name_spatial_AB = ("BO2STTrial/step_length_width_cadence_AB.xlsx")
    spatial_AB_xcl = pd.read_excel(file_name_spatial_AB, header = [0], index_col = [0])
    if (new_ss):
        file_name_step_width = ("BO2STTrial/stride_length_width_new_ss.xlsx")
    else:
        file_name_step_width = ("BO2STTrial/stride_length_width.xlsx")
    step_width_xcl = pd.read_excel(file_name_step_width, header = [0], index_col = [0])
    file_name_step_width_AB = ("BO2STTrial/stride_length_width_AB.xlsx")
    step_width_AB_xcl = pd.read_excel(file_name_step_width_AB, header = [0], index_col = [0])
    if (new_ss):
        file_name_cycle_phases = ("BO2STTrial/gait_cycle_phases_new_ss.xlsx")
    else:
        file_name_cycle_phases = ("BO2STTrial/gait_cycle_phases.xlsx")
    cycle_phases_xcl = pd.read_excel(file_name_cycle_phases, header = [0], index_col = [0])
    file_name_cycle_phases_AB = ("BO2STTrial/gait_cycle_phases_AB.xlsx")
    cycle_phases_AB_xcl = pd.read_excel(file_name_cycle_phases_AB, header = [0], index_col = [0])
    
    # match control
    control_file_name = ("BO2STTrial/matched_subjects.xlsx")
    control_xcl = pd.read_excel(control_file_name, header = [0], index_col = [0])
    ID_AB = control_xcl.loc[ID].Matched_control
    
    # compute stance 
    stance_max = 1/cycle_phases_AB_xcl.loc[ID_AB]["stance_"+time]
    stance = 1/cycle_phases_xcl.loc[ID]["stance_"+time]
    stance_norm = stance/stance_max*100
    stance_AB = stance_max/stance_max*100

    # compute cadence 
    cadence_max = spatial_AB_xcl.loc[ID_AB]["cadence_"+time]
    cadence = spatial_xcl.loc[ID]["cadence_"+time]
    cadence_norm = cadence/cadence_max*100
    cadence_AB = cadence_max/cadence_max*100
    
    # compute step length
    step_length_max = spatial_AB_xcl.loc[ID_AB]["step_length_"+side+'_'+time]
    step_length = spatial_xcl.loc[ID]["step_length_"+side+'_'+time]
    step_length_norm = step_length/step_length_max*100
    step_length_AB = step_length_max/step_length_max*100
    
    # compute step width 
    step_width_max = 1/step_width_AB_xcl.loc[ID_AB]["stride_width_"+time]
    step_width = 1/step_width_xcl.loc[ID]["stride_width_"+time]
    step_width_norm = step_width/step_width_max*100
    step_width_AB = step_width_max/step_width_max*100
    
    
    # construct pairs: value of the participant at time point and normative value of the matched AB control
    cadence = (cadence_norm, cadence_AB)
    stance = (stance_norm, stance_AB)
    step_width = (step_width_norm, step_width_AB)
    step_length = (step_length_norm, step_length_AB)
    
    return stance, cadence, step_length, step_width


def normalize_coordination(ID, side, time, new_ss=False):
    """
    Normalize the coordination parameters (ACC, relative phases) of the participant between 0 and 100.
    Input:
        * ID(string): e.g. "BO2ST_101"
        * time (string): time point e.g. "BL" or "T0"
    Output:
        * ACC
    """ 
    
    # read files
    if (new_ss):
        file_name_ACC = ("BO2STTrial/ACCs_new_ss.xlsx")
    else:    
        file_name_ACC = ("BO2STTrial/ACCs.xlsx")
    ACC_xcl = pd.read_excel(file_name_ACC, header = [0], index_col = [0])
    
    # match control
    control_file_name = ("BO2STTrial/matched_subjects.xlsx")
    control_xcl = pd.read_excel(control_file_name, header = [0], index_col = [0])
    ID_AB = control_xcl.loc[ID].Matched_control
    
    # compute ACC
    ACC_max = 0.97 # according to litterature
    ACC = ACC_xcl.loc[ID]["ACC_"+side+"_"+time]
    ACC_norm = ACC/ACC_max*100
    ACC_AB = ACC_max/ACC_max*100
    
    # construct pairs: value of the participant at time point and normative value of the matched AB control
    ACC = (ACC_norm, ACC_AB)
    
    return ACC


def get_spider_chart_2(ID, side):
    
    # functional performance parameters
    _10MWT_speed_BL, _6MWT_speed_BL, ss_speed_BL, TUG_BL, LEMS_BL = normalize_functional_walking(ID, "BL") # BL
    _10MWT_speed_T0, _6MWT_speed_T0, ss_speed_T0, TUG_T0, LEMS_T0 = normalize_functional_walking(ID, "T0") # T0
    
    # kinetic parameters
    tp_BL, area_braking_BL, area_propul_BL = normalize_kinetics(ID, "BL") # BL
    tp_T0, area_braking_T0, area_propul_T0 = normalize_kinetics(ID, "T0") # T0
    
    # variability parameters
    EV_BL, CoV_step_length_BL, CoV_step_width_BL = normalize_variability(ID, side, "BL") # BL
    EV_T0, CoV_step_length_T0, CoV_step_width_T0 = normalize_variability(ID, side, "T0") # T0
    
    # symmetry parameters
    A_knee_BL, A_hip_BL, A_ankle_BL = normalize_symmetry(ID, "BL") # BL
    A_knee_T0, A_hip_T0, A_ankle_T0 = normalize_symmetry(ID, "T0") # T0
   
    # spatiotemporal parameters 
    stance_BL, cadence_BL, step_length_BL, step_width_BL = normalize_spatiotemporal(ID, side, "BL") # BL
    stance_T0, cadence_T0, step_length_T0, step_width_T0 = normalize_spatiotemporal(ID, side, "T0") # T0
    
    # coordination parameters 
    ACC_BL = normalize_coordination(ID, side, "BL") # BL
    ACC_T0 = normalize_coordination(ID, side, "T0") # T0
    
    # initialize data of lists.
    data_participant = {# functional performance
                        '10MWT speed': [_10MWT_speed_BL[0], _10MWT_speed_T0[0], _10MWT_speed_BL[1]],
                        '6MWT speed': [_6MWT_speed_BL[0], _6MWT_speed_T0[0], _6MWT_speed_BL[1]],
                        'self-selected speed': [ss_speed_BL[0], ss_speed_T0[0], ss_speed_BL[1]],
                        'TUG': [TUG_BL[0], TUG_T0[0], TUG_BL[1]],
                        'LEMS': [LEMS_BL[0], LEMS_T0[0], LEMS_BL[1]],
                        # variability
                        'EV': [EV_BL[0], EV_T0[0], EV_BL[1]],
                        'CoV step length': [CoV_step_length_BL[0], CoV_step_length_T0[0], CoV_step_length_BL[1]],
                        'CoV step width': [CoV_step_width_BL[0], CoV_step_width_T0[0], CoV_step_width_BL[1]],
                        # asymmetry 
                        'A knee': [A_knee_BL[0], A_knee_T0[0], A_knee_BL[1]],
                        'A hip': [A_hip_BL[0], A_hip_T0[0], A_hip_BL[1]],
                        'A ankle': [A_ankle_BL[0], A_ankle_T0[0], A_ankle_BL[1]],
                        #'SI EV': [SI_EV_BL[0], SI_EV_T0[0], SI_EV_BL[1]],
                        # spatiotemporal
                        'stance phase': [stance_BL[0], stance_T0[0], stance_BL[1]],
                        'cadence': [cadence_BL[0], cadence_T0[0], cadence_BL[1]],
                        'step length': [step_length_BL[0], step_length_T0[0], step_length_BL[1]],
                        'step width': [step_width_BL[0], step_width_T0[0], step_width_BL[1]],
                        # kinetics
                        'GRF TP': [tp_BL[0], tp_T0[0], tp_BL[1]],
                        'GRF area braking': [area_braking_BL[0], area_braking_T0[0], area_braking_BL[1]],
                        'GRF area propulsion': [area_propul_BL[0], area_propul_T0[0], area_propul_BL[1]],
                        # coordination
                        'ACC': [ACC_BL[0], ACC_T0[0], ACC_BL[1]],
               }

    # Create DataFrame
    df_participant = pd.DataFrame(data_participant, index=['BL', 'T0', 'AB'])

    # create spider chart 
    fig = go.Figure()

    #BL
    fig.add_trace(go.Scatterpolar(
          r=df_participant.loc['BL'].values.tolist() + [df_participant.loc['BL'].values[0]],
          theta=df_participant.columns.tolist() + [df_participant.columns[0]],
          fill='toself',
          name='BL',
          mode='lines+markers'
    ))
    
    #T0
    fig.add_trace(go.Scatterpolar(
          r=df_participant.loc['T0'].values.tolist() + [df_participant.loc['T0'].values[0]],
          theta=df_participant.columns.tolist() + [df_participant.columns[0]],
          fill='toself',
          name='T0',
          mode='lines+markers'
    ))
    
    #AB
    fig.add_trace(go.Scatterpolar(
          r=df_participant.loc['AB'].values.tolist() + [df_participant.loc['AB'].values[0]],
          theta=df_participant.columns.tolist() + [df_participant.columns[0]],
          #fill='toself',
          name='AB',
          mode='lines+markers'
    ))
       
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=False,
          #range=[0,110]
        ),),
      showlegend=True,
    )
   
    fig.show()
    
    # export image
    fig.write_image("images/spiderpdf.pdf")
    
    return df_participant
    
        
########################################################################################################################
def get_spider_chart_new_ss(ID, side):
    
    # kinetic parameters
    tp_BL, area_braking_BL, area_propul_BL = normalize_kinetics(ID, "T0") # T0 initial ss
    tp_T0, area_braking_T0, area_propul_T0 = normalize_kinetics(ID, "T0", new_ss=True) # T0 new ss
    
    # variability parameters
    EV_BL, CoV_step_length_BL, CoV_step_width_BL = normalize_variability(ID, side, "T0") # T0 initial ss
    EV_T0, CoV_step_length_T0, CoV_step_width_T0 = normalize_variability(ID, side, "T0", new_ss=True) # T0 new ss
    
    # symmetry parameters
    A_knee_BL, A_hip_BL, A_ankle_BL = normalize_symmetry(ID, "T0") # T0 initial ss
    A_knee_T0, A_hip_T0, A_ankle_T0 = normalize_symmetry(ID, "T0", new_ss=True) # T0 new ss
   
    # spatiotemporal parameters 
    stance_BL, cadence_BL, step_length_BL, step_width_BL = normalize_spatiotemporal(ID, side, "T0") # T0 initial ss
    stance_T0, cadence_T0, step_length_T0, step_width_T0 = normalize_spatiotemporal(ID, side, "T0", new_ss=True) # T0 new ss
    
    # coordination parameters 
    ACC_BL = normalize_coordination(ID, side, "T0") # T0 initial ss
    ACC_T0 = normalize_coordination(ID, side, "T0", new_ss=True) # T0 new ss
    
    # initialize data of lists.
    data_participant = {# variability
                        'EV': [EV_BL[0], EV_T0[0], EV_BL[1]],
                        'CoV step length': [CoV_step_length_BL[0], CoV_step_length_T0[0], CoV_step_length_BL[1]],
                        'CoV step width': [CoV_step_width_BL[0], CoV_step_width_T0[0], CoV_step_width_BL[1]],
                        # asymmetry 
                        'A knee': [A_knee_BL[0], A_knee_T0[0], A_knee_BL[1]],
                        'A hip': [A_hip_BL[0], A_hip_T0[0], A_hip_BL[1]],
                        'A ankle': [A_ankle_BL[0], A_ankle_T0[0], A_ankle_BL[1]],
                        #'SI EV': [SI_EV_BL[0], SI_EV_T0[0], SI_EV_BL[1]],
                        # spatiotemporal
                        'stance phase': [stance_BL[0], stance_T0[0], stance_BL[1]],
                        'cadence': [cadence_BL[0], cadence_T0[0], cadence_BL[1]],
                        'step length': [step_length_BL[0], step_length_T0[0], step_length_BL[1]],
                        'step width': [step_width_BL[0], step_width_T0[0], step_width_BL[1]],
                        # kinetics
                        'GRF TP': [tp_BL[0], tp_T0[0], tp_BL[1]],
                        'GRF area braking': [area_braking_BL[0], area_braking_T0[0], area_braking_BL[1]],
                        'GRF area propulsion': [area_propul_BL[0], area_propul_T0[0], area_propul_BL[1]],
                        # coordination
                        'ACC': [ACC_BL[0], ACC_T0[0], ACC_BL[1]],
               }

    # Create DataFrame
    df_participant = pd.DataFrame(data_participant, index=['T0', 'new_ss', 'AB'])

    # create spider chart 
    fig = go.Figure()

    #new_ss
    fig.add_trace(go.Scatterpolar(
          r=df_participant.loc['new_ss'].values.tolist() + [df_participant.loc['new_ss'].values[0]],
          theta=df_participant.columns.tolist() + [df_participant.columns[0]],
          fill='toself',
          name='new self-selected speed',
          mode='lines+markers',
          marker_color="mediumpurple"
    ))
    
    #T0
    fig.add_trace(go.Scatterpolar(
          r=df_participant.loc['T0'].values.tolist() + [df_participant.loc['T0'].values[0]],
          theta=df_participant.columns.tolist() + [df_participant.columns[0]],
          fill='toself',
          name='initial self-selected speed',
          mode='lines+markers',
    ))
    
    #AB
    fig.add_trace(go.Scatterpolar(
          r=df_participant.loc['AB'].values.tolist() + [df_participant.loc['AB'].values[0]],
          theta=df_participant.columns.tolist() + [df_participant.columns[0]],
          #fill='toself',
          name='AB',
          mode='lines+markers'
    ))
       
    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=False,
          #range=[0,110]
        ),),
      showlegend=True,
    )
   
    fig.show()
    
    # export image
    fig.write_image("images/spiderpdf.pdf")
    
    return df_participant

########################################################################################################################
        
########################################################################################################################
def get_spider_chart_whole_trial(ID, side):
    
    # functional performance parameters
    _10MWT_speed_BL, _6MWT_speed_BL, ss_speed_BL, TUG_BL, LEMS_BL = normalize_functional_walking(ID, "BL") # BL
    _10MWT_speed_T0, _6MWT_speed_T0, ss_speed_T0, TUG_T0, LEMS_T0 = normalize_functional_walking(ID, "T0") # T0
    _10MWT_speed_T1, _6MWT_speed_T1, ss_speed_T1, TUG_T1, LEMS_T1 = normalize_functional_walking(ID, "T1") # T1
    _10MWT_speed_T2, _6MWT_speed_T2, ss_speed_T2, TUG_T2, LEMS_T2 = normalize_functional_walking(ID, "T2") # T2
    
    # kinetic parameters
    tp_BL, area_braking_BL, area_propul_BL = normalize_kinetics(ID, "BL")
    tp_T0, area_braking_T0, area_propul_T0 = normalize_kinetics(ID, "T0")
    tp_T1, area_braking_T1, area_propul_T1 = normalize_kinetics(ID, "T1")
    tp_T2, area_braking_T2, area_propul_T2 = normalize_kinetics(ID, "T2")
    
    # variability parameters
    EV_BL, CoV_step_length_BL, CoV_step_width_BL = normalize_variability(ID, side, "BL")
    EV_T0, CoV_step_length_T0, CoV_step_width_T0 = normalize_variability(ID, side, "T0")
    EV_T1, CoV_step_length_T1, CoV_step_width_T1 = normalize_variability(ID, side, "T1")
    EV_T2, CoV_step_length_T2, CoV_step_width_T2 = normalize_variability(ID, side, "T2")
    
    # symmetry parameters
    A_knee_BL, A_hip_BL, A_ankle_BL = normalize_symmetry(ID, "BL")
    A_knee_T0, A_hip_T0, A_ankle_T0 = normalize_symmetry(ID, "T0")
    A_knee_T1, A_hip_T1, A_ankle_T1 = normalize_symmetry(ID, "T1")
    A_knee_T2, A_hip_T2, A_ankle_T2 = normalize_symmetry(ID, "T2")
   
    # spatiotemporal parameters 
    stance_BL, cadence_BL, step_length_BL, step_width_BL = normalize_spatiotemporal(ID, side, "BL")
    stance_T0, cadence_T0, step_length_T0, step_width_T0 = normalize_spatiotemporal(ID, side, "T0")
    stance_T1, cadence_T1, step_length_T1, step_width_T1 = normalize_spatiotemporal(ID, side, "T1")
    stance_T2, cadence_T2, step_length_T2, step_width_T2 = normalize_spatiotemporal(ID, side, "T2")
    
    # coordination parameters 
    ACC_BL = normalize_coordination(ID, side, "BL")
    ACC_T0 = normalize_coordination(ID, side, "T0")
    ACC_T1 = normalize_coordination(ID, side, "T1")
    ACC_T2 = normalize_coordination(ID, side, "T2")
    
    # initialize data of lists.
    data_participant = {# functional performance
                        '10MWT speed': [_10MWT_speed_BL[0], _10MWT_speed_T0[0], _10MWT_speed_T1[0], _10MWT_speed_T2[0],  _10MWT_speed_BL[1]],
                        '6MWT speed': [_6MWT_speed_BL[0], _6MWT_speed_T0[0], _6MWT_speed_T1[0], _6MWT_speed_T2[0],  _6MWT_speed_BL[1]],
                        'self-selected speed': [ss_speed_BL[0], ss_speed_T0[0], ss_speed_T1[0], ss_speed_T2[0], ss_speed_BL[1]],
                        'TUG': [TUG_BL[0], TUG_T0[0], TUG_T1[0], TUG_T2[0], TUG_BL[1]],
                        'LEMS': [LEMS_BL[0], LEMS_T0[0], LEMS_T1[0], LEMS_T2[0], LEMS_BL[1]],
                        # variability
                        'EV': [EV_BL[0], EV_T0[0], EV_T1[0], EV_T2[0], EV_BL[1]],
                        'CoV step length': [CoV_step_length_BL[0], CoV_step_length_T0[0], CoV_step_length_T1[0], CoV_step_length_T2[0], CoV_step_length_BL[1]],
                        'CoV step width': [CoV_step_width_BL[0], CoV_step_width_T0[0], CoV_step_width_T1[0], CoV_step_width_T2[0], CoV_step_width_BL[1]],
                        # asymmetry 
                        'A knee': [A_knee_BL[0], A_knee_T0[0], A_knee_T1[0], A_knee_T2[0], A_knee_BL[1]],
                        'A hip': [A_hip_BL[0], A_hip_T0[0], A_hip_T1[0], A_hip_T2[0], A_hip_BL[1]],
                        'A ankle': [A_ankle_BL[0], A_ankle_T0[0], A_ankle_T1[0], A_ankle_T2[0], A_ankle_BL[1]],
                        #'SI EV': [SI_EV_BL[0], SI_EV_T0[0], SI_EV_BL[1]],
                        # spatiotemporal
                        'stance phase': [stance_BL[0], stance_T0[0], stance_T1[0], stance_T2[0], stance_BL[1]],
                        'cadence': [cadence_BL[0], cadence_T0[0], cadence_T1[0], cadence_T2[0], cadence_BL[1]],
                        'step length': [step_length_BL[0], step_length_T0[0], step_length_T1[0], step_length_T2[0], step_length_BL[1]],
                        'step width': [step_width_BL[0], step_width_T0[0], step_width_T1[0], step_width_T2[0], step_width_BL[1]],
                        # kinetics
                        'GRF TP': [tp_BL[0], tp_T0[0], tp_T1[0], tp_T2[0], tp_BL[1]],
                        'GRF area braking': [area_braking_BL[0], area_braking_T0[0], area_braking_T1[0],  area_braking_T2[0], area_braking_BL[1]],
                        'GRF area propulsion': [area_propul_BL[0], area_propul_T0[0], area_propul_T1[0], area_propul_T2[0], area_propul_BL[1]],
                        # coordination
                        'ACC': [ACC_BL[0], ACC_T0[0], ACC_T1[0], ACC_T2[0], ACC_BL[1]],
               }

    # Create DataFrame
    df_participant = pd.DataFrame(data_participant, index=['BL', 'T0', 'T1', 'T2', 'AB'])

    # Define a color scale for the gradient
    color_scale = np.linspace(0, 1, len(df_participant.index))

    # create spider chart 
    fig = go.Figure()

    # Iterate over each time point and add a trace with a different color
    for i, time_point in enumerate(df_participant.index):
        color_value = color_scale[i]
        color = f"rgba({int(255 * color_value)}, 0, {int(255 * (1 - color_value))}, 0.9)"  # Adjust the transparency as needed

        if (time_point!='AB'):
            fig.add_trace(go.Scatterpolar(
                r=df_participant.loc[time_point].values.tolist() + [df_participant.loc[time_point].values[0]],
                theta=df_participant.columns.tolist() + [df_participant.columns[0]],
                fill='toself',
                name=time_point,
                mode='lines+markers',
                line_color=color  # Use line_color instead of marker_color
            ))
            
        else:
            #AB
            fig.add_trace(go.Scatterpolar(
                  r=df_participant.loc['AB'].values.tolist() + [df_participant.loc['AB'].values[0]],
                  theta=df_participant.columns.tolist() + [df_participant.columns[0]],
                  #fill='toself',
                  name='AB',
                  mode='lines+markers'
            ))

    # Update layout if needed
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range = [0,350]
            )
        )
    )
    
   
   
    fig.show()
    
    # export image
    fig.write_image("images/spiderpdf.pdf")
    
    return df_participant

########################################################################################################################
def get_spider_chart_FU(ID, side):
    
    # functional performance parameters
    _10MWT_speed_BL, _6MWT_speed_BL, ss_speed_BL, TUG_BL, LEMS_BL = normalize_functional_walking(ID, "T2") # T2
    _10MWT_speed_T0, _6MWT_speed_T0, ss_speed_T0, TUG_T0, LEMS_T0 = normalize_functional_walking(ID, "F1") # F1
    _10MWT_speed_T1, _6MWT_speed_T1, ss_speed_T1, TUG_T1, LEMS_T1 = normalize_functional_walking(ID, "F4") # F4
    _10MWT_speed_T2, _6MWT_speed_T2, ss_speed_T2, TUG_T2, LEMS_T2 = normalize_functional_walking(ID, "F8") # F8
    
    # kinetic parameters
    tp_BL, area_braking_BL, area_propul_BL = normalize_kinetics(ID, "T2") # T2
    tp_T0, area_braking_T0, area_propul_T0 = normalize_kinetics(ID, "F1") # F1
    tp_T1, area_braking_T1, area_propul_T1 = normalize_kinetics(ID, "F4") # F4
    tp_T2, area_braking_T2, area_propul_T2 = normalize_kinetics(ID, "F8") # F8
    
    # variability parameters
    EV_BL, CoV_step_length_BL, CoV_step_width_BL = normalize_variability(ID, side, "T2")
    EV_T0, CoV_step_length_T0, CoV_step_width_T0 = normalize_variability(ID, side, "F1")
    EV_T1, CoV_step_length_T1, CoV_step_width_T1 = normalize_variability(ID, side, "F4")
    EV_T2, CoV_step_length_T2, CoV_step_width_T2 = normalize_variability(ID, side, "F8")
    
    # symmetry parameters
    A_knee_BL, A_hip_BL, A_ankle_BL = normalize_symmetry(ID, "T2") # T2
    A_knee_T0, A_hip_T0, A_ankle_T0 = normalize_symmetry(ID, "F1") # F1
    A_knee_T1, A_hip_T1, A_ankle_T1 = normalize_symmetry(ID, "F4") # F4
    A_knee_T2, A_hip_T2, A_ankle_T2 = normalize_symmetry(ID, "F8") # F8
   
    # spatiotemporal parameters 
    stance_BL, cadence_BL, step_length_BL, step_width_BL = normalize_spatiotemporal(ID, side, "T2")
    stance_T0, cadence_T0, step_length_T0, step_width_T0 = normalize_spatiotemporal(ID, side, "F1")
    stance_T1, cadence_T1, step_length_T1, step_width_T1 = normalize_spatiotemporal(ID, side, "F4")
    stance_T2, cadence_T2, step_length_T2, step_width_T2 = normalize_spatiotemporal(ID, side, "F8")
    
    # coordination parameters 
    ACC_BL = normalize_coordination(ID, side, "T2")
    ACC_T0 = normalize_coordination(ID, side, "F1")
    ACC_T1 = normalize_coordination(ID, side, "F4")
    ACC_T2 = normalize_coordination(ID, side, "F8")
    
    # initialize data of lists.
    data_participant = {# functional performance
                        '10MWT speed': [_10MWT_speed_BL[0], _10MWT_speed_T0[0], _10MWT_speed_T1[0], _10MWT_speed_T2[0],  _10MWT_speed_BL[1]],
                        '6MWT speed': [_6MWT_speed_BL[0], _6MWT_speed_T0[0], _6MWT_speed_T1[0], _6MWT_speed_T2[0],  _6MWT_speed_BL[1]],
                        'self-selected speed': [ss_speed_BL[0], ss_speed_T0[0], ss_speed_T1[0], ss_speed_T2[0], ss_speed_BL[1]],
                        'TUG': [TUG_BL[0], TUG_T0[0], TUG_T1[0], TUG_T2[0], TUG_BL[1]],
                        'LEMS': [LEMS_BL[0], LEMS_T0[0], LEMS_T1[0], LEMS_T2[0], LEMS_BL[1]],
                        # variability
                        'EV': [EV_BL[0], EV_T0[0], EV_T1[0], EV_T2[0], EV_BL[1]],
                        'CoV step length': [CoV_step_length_BL[0], CoV_step_length_T0[0], CoV_step_length_T1[0], CoV_step_length_T2[0], CoV_step_length_BL[1]],
                        'CoV step width': [CoV_step_width_BL[0], CoV_step_width_T0[0], CoV_step_width_T1[0], CoV_step_width_T2[0], CoV_step_width_BL[1]],
                        # asymmetry 
                        'A knee': [A_knee_BL[0], A_knee_T0[0], A_knee_T1[0], A_knee_T2[0], A_knee_BL[1]],
                        'A hip': [A_hip_BL[0], A_hip_T0[0], A_hip_T1[0], A_hip_T2[0], A_hip_BL[1]],
                        'A ankle': [A_ankle_BL[0], A_ankle_T0[0], A_ankle_T1[0], A_ankle_T2[0], A_ankle_BL[1]],
                        #'SI EV': [SI_EV_BL[0], SI_EV_T0[0], SI_EV_BL[1]],
                        # spatiotemporal
                        'stance phase': [stance_BL[0], stance_T0[0], stance_T1[0], stance_T2[0], stance_BL[1]],
                        'cadence': [cadence_BL[0], cadence_T0[0], cadence_T1[0], cadence_T2[0], cadence_BL[1]],
                        'step length': [step_length_BL[0], step_length_T0[0], step_length_T1[0], step_length_T2[0], step_length_BL[1]],
                        'step width': [step_width_BL[0], step_width_T0[0], step_width_T1[0], step_width_T2[0], step_width_BL[1]],
                        # kinetics
                        'GRF TP': [tp_BL[0], tp_T0[0], tp_T1[0], tp_T2[0], tp_BL[1]],
                        'GRF area braking': [area_braking_BL[0], area_braking_T0[0], area_braking_T1[0],  area_braking_T2[0], area_braking_BL[1]],
                        'GRF area propulsion': [area_propul_BL[0], area_propul_T0[0], area_propul_T1[0], area_propul_T2[0], area_propul_BL[1]],
                        # coordination
                        'ACC': [ACC_BL[0], ACC_T0[0], ACC_T1[0], ACC_T2[0], ACC_BL[1]],
               }

    # Create DataFrame
    df_participant = pd.DataFrame(data_participant, index=['T2', 'F1', 'F4', 'F8', 'AB'])

    # Define a color scale for the gradient
    color_scale = np.linspace(0, 1, len(df_participant.index))

    # create spider chart 
    fig = go.Figure()

    # Iterate over each time point and add a trace with a different color
    for i, time_point in enumerate(df_participant.index):
        color_value = color_scale[i]
        color = f"rgba(0,{int(255 * color_value)}, {int(255 * (1 - color_value))}, 0.9)"  # Adjust the transparency as needed

        if (time_point!='AB'):
            fig.add_trace(go.Scatterpolar(
                r=df_participant.loc[time_point].values.tolist() + [df_participant.loc[time_point].values[0]],
                theta=df_participant.columns.tolist() + [df_participant.columns[0]],
                fill='toself',
                name=time_point,
                mode='lines+markers',
                line_color=color  # Use line_color instead of marker_color
            ))
            
        else:
            #AB
            fig.add_trace(go.Scatterpolar(
                  r=df_participant.loc['AB'].values.tolist() + [df_participant.loc['AB'].values[0]],
                  theta=df_participant.columns.tolist() + [df_participant.columns[0]],
                  #fill='toself',
                  name='AB',
                  mode='lines+markers'
            ))

    # Update layout if needed
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=False,
                range = [0,350]
            )
        )
    )
   
    fig.show()
    
    # export image
    fig.write_image("images/spiderpdf.pdf")
    
    return df_participant


    
########################################################################################################################

def compute_corr(param1, param2, EV_BL, EV_T0, A_BL, A_T0, STD_BL, STD_T0):
    """
    Compute the Spearman correlation coefficient with associated p-value between param1 and param2.
    The Spearman rank-order correlation coefficient is a nonparametric measure of the monotonicity of the relationship between param1 and param2.
    Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
    Correlations of -1 or +1 imply an exact monotonic relationship.
    Positive correlations imply that as x increases, so does y.
    Negative correlations imply that as x increases, y decreases.
    
    Inputs:
        * param1 (string): '10MWT', '6MWT', 'TUG', 'A', 'EV', or 'STD PHI'
        * param2 (string): 'A', 'EV', or 'STD PHI'
        
    Outputs:
        * corr (float): Spearman correlation
        * p (float): p-value
    """
    
    # param1 can be 10MWT, 6MWT or TUG
    # read xcl file to obtain 6MWT, 10MWT, TUG data
    file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    overground_gait_assessments_xcl = pd.read_excel(file_name, header = [0], index_col = [0])
    #normalize_gait_assessments(overground_gait_assessments_xcl)
    overground_gait_assessments_xcl['delta_10MWT'] = overground_gait_assessments_xcl['10MWT_time_T0'] - overground_gait_assessments_xcl['10MWT_time_BL']
    overground_gait_assessments_xcl['delta_6MWT'] = overground_gait_assessments_xcl['6MWT_distance_T0'] - overground_gait_assessments_xcl['6MWT_distance_BL']
    overground_gait_assessments_xcl['delta_TUG'] = overground_gait_assessments_xcl['TUG_time_T0'] - overground_gait_assessments_xcl['TUG_time_BL']
     
    # param2 can be A, EV, or STD PHI
    # EV
    #max_EV = max(max(EV_BL), max(EV_T0))
    #EV_BL = [x/max_EV for x in EV_BL]
    #EV_T0 = [x/max_EV for x in EV_T0]
    # asymmetry
    #max_A = max(max(A_BL), max(A_T0))
    #A_BL = [x/max_A for x in A_BL]
    #A_T0 = [x/max_A for x in A_T0]
    # STD phi
    #max_STD = max(max(STD_BL), max(STD_T0))
    #STD_BL = [x/max_STD for x in STD_BL]
    #STD_T0 = [x/max_STD for x in STD_T0]
    
    # compute the delta for param1
    if (param1 == '10MWT'):
        delta_param1 = overground_gait_assessments_xcl['delta_10MWT'].values
    elif (param1 == '6MWT'):
        delta_param1 = overground_gait_assessments_xcl['delta_6MWT'].values
    elif (param1 == 'TUG'):
        delta_param1 = overground_gait_assessments_xcl['delta_TUG'].values
    elif (param1 == 'EV'):
        delta_param1 = [EV_T0[x] - EV_BL[x] for x in range(len(EV_T0))]
    elif (param1 == 'A'):
        delta_param1 = [A_T0[x] - A_BL[x] for x in range(len(A_T0))]
    elif (param1 == 'STD PHI'):
        delta_param1 = [STD_T0[x] - STD_BL[x] for x in range(len(STD_T0))]
    
    if (param2 == 'EV'):
        delta_param2 = [EV_T0[x] - EV_BL[x] for x in range(len(EV_T0))]
    elif (param2 == 'A'):
        delta_param2 = [A_T0[x] - A_BL[x] for x in range(len(A_T0))]
    elif (param2 == 'STD PHI'):
        delta_param2 = [STD_T0[x] - STD_BL[x] for x in range(len(STD_T0))]
        
    # corr
    corr, p = stats.spearmanr(delta_param1, delta_param2)
    
    return corr, p
    
        
def individual_progress_walking_mechanics(ID):
    """
    Plot the progress of the participant in terms of walking mechanics (EV, A, STD PHI) according to time.
    Inputs:
        * ID (string): ID of the participant, e.g. "BO2ST_101"
    """
    
    # read xcl files
    walking_mechanics_performances_file_name = ("BO2STTrial/walking_mechanics_performances.xlsx")
    walking_mechanics_performances_xcl = pd.read_excel(walking_mechanics_performances_file_name, header = [0], index_col = [0])
    
    # matched subject files
    matched_subjects = pd.read_excel("BO2STTrial/matched_subjects.xlsx", index_col=0)
    matched_control = matched_subjects.loc[ID]['Matched_control'] # find the matched healthy control to the SCI participant
    walking_mechanics_performances_control_file_name = ("BO2STTrial/walking_mechanics_performances_control.xlsx") # find the walking mechanics of the matched control
    walking_mechanics_performances_control_xcl = pd.read_excel(walking_mechanics_performances_control_file_name, header = [0], index_col = [0])
    
    x = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8']
    EV_y = walking_mechanics_performances_xcl.loc[ID][['EV_BL', 'EV_T0', 'EV_T1', 'EV_T2', 'EV_F1', 'EV_F4', 'EV_F8']] # relative to T0?
    A_y = walking_mechanics_performances_xcl.loc[ID][['A_BL', 'A_T0', 'A_T1', 'A_T2', 'A_F1', 'A_F4', 'A_F8']] # relative to T0?
    STDPHI_y = walking_mechanics_performances_xcl.loc[ID][['STD_BL', 'STD_T0', 'STD_T1', 'STD_T2', 'STD_F1', 'STD_F4', 'STD_F8']] # relative to T0?
    EV_AB = walking_mechanics_performances_control_xcl.loc[matched_control][['EV_BL', 'EV_T0', 'EV_T1', 'EV_T2', 'EV_F1', 'EV_F4', 'EV_F8']]
    A_AB = walking_mechanics_performances_control_xcl.loc[matched_control][['A_BL', 'A_T0', 'A_T1', 'A_T2', 'A_F1', 'A_F4', 'A_F8']]
   
    fig = make_subplots(rows=1, cols=3, x_title = 'Time', subplot_titles=("EV", "A", "STD PHI"), horizontal_spacing=0.115)

    # EV
    fig.add_trace(
        go.Scatter(x=x, y=EV_y, line=dict(color='black')),
        row=1, col=1)
    # EV control
    fig.add_trace(
        go.Scatter(x=x, y=EV_AB, line=dict(color='grey', dash='dot'), mode='lines'),
        row=1, col=1)

    # A
    fig.add_trace(
        go.Scatter(x=x, y=A_y, line=dict(color='black')),
        row=1, col=2)
    # A control
    fig.add_trace(
        go.Scatter(x=x, y=A_AB, line=dict(color='grey', dash='dot'), mode='lines'),
        row=1, col=2)
    
    # STD PHI
    fig.add_trace(
        go.Scatter(x=x, y=STDPHI_y, line=dict(color='black')),
        row=1, col=3)

    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=450, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    #fig.add_hline(y=0, line_color='black', line_dash="dash")
    fig['layout']['xaxis1'].update(
        tickmode = 'array',
        tickvals = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'],
        ticktext = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8']
    )
    fig.update_traces(connectgaps=True)
    # Update yaxis properties
    fig.update_yaxes(title_text="EV", range= [0,8], row=1, col=1)
    fig.update_yaxes(title_text="A", range= [0,1.5], row=1, col=2)
    fig.update_yaxes(title_text="$STD\Phi$", range= [0,60], row=1, col=3)
    fig.show()
    
    
    
def normalize_gait_assessments(overground_gait_assessments_xcl):
    """
    WRONG NORMALIZATION!!!!
    Normalize the 10MWT_time BL/T0, 6MWT_distance BL/T0 and TUG_time BL/T0 columns between 0 and 1 according to the maximum value in the cohort. 
    
    Inputs:
        * overground_gait_assessments_xcl (dataframe): data frame obtained when reading the excel file "Overground_gait_assessments.xls"
    """
    
    max_10MWT = max(overground_gait_assessments_xcl['10MWT_time_BL'].max(), overground_gait_assessments_xcl['10MWT_time_T0'].max())
    min_10MWT = min(overground_gait_assessments_xcl['10MWT_time_BL'].min(), overground_gait_assessments_xcl['10MWT_time_T0'].min())
    max_6MWT = max(overground_gait_assessments_xcl['6MWT_distance_BL'].max(), overground_gait_assessments_xcl['6MWT_distance_T0'].max())
    min_6MWT = min(overground_gait_assessments_xcl['6MWT_distance_BL'].min(), overground_gait_assessments_xcl['6MWT_distance_T0'].min())
    max_TUG = max(overground_gait_assessments_xcl['TUG_time_BL'].max(), overground_gait_assessments_xcl['TUG_time_T0'].max())
    min_TUG = min(overground_gait_assessments_xcl['TUG_time_BL'].min(), overground_gait_assessments_xcl['TUG_time_T0'].min())
    
    for column in ['10MWT_time_BL', '10MWT_time_T0']:
        overground_gait_assessments_xcl[column] = (overground_gait_assessments_xcl[column]/ (max_10MWT))
        
    for column in ['6MWT_distance_BL', '6MWT_distance_T0']:
        overground_gait_assessments_xcl[column] = (overground_gait_assessments_xcl[column]/ (max_6MWT ))
                                                   
    for column in ['TUG_time_BL', 'TUG_time_T0']:
        overground_gait_assessments_xcl[column] = (overground_gait_assessments_xcl[column]/ (max_TUG))      
            
        