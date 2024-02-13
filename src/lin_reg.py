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
import statsmodels.api as sm

def compute_percentage_change(early_value, late_value):
    """
    The improvement of a parameter is evaluated by comparing its change from an early (BL or T0) to a late (T0 or T2) time point
    expressed as absolute values and as a percentage change to illustrate the magnitude of improvement or decrement
    over time.
    Inputs:
        * early_value (float): value of the parameter at early time point
        * late_value (float): value of the parameter at late time point
    Output: 
        * percent (float): percentage change 
    """
    

    # compute percentage change 
    percent = (late_value - early_value)/early_value * 100
    
    return percent


def plot_percent_change(early, late):
    """
    The improvement of a parameter is evaluated by comparing its change from an early (BL or T0) to a late (T0 or T2) time point
    expressed as absolute values and as a percentage change to illustrate the magnitude of improvement or decrement
    over time.
    Plot the percentage change of each parameter in a bar plot.
    Print the p-values of the wilcoxon signed rank test for nonnormally distributed parameters.
    Inputs:
        * early (string): early time point e.g. "BL" or "T0"
        * late (string): late time point e.g. "T0" or "T2"

    """
    
    ############################################################################################################ 
    # COMPUTE ALL DELTAS
    ############################################################################################################ 
    
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    sides = ['L', 'R', 'L', 'R', 'L', 'L'] # MA side

    perc_changes = compute_perc_change(early, late)
    # ss speed
    ss_speed_change = np.squeeze(perc_changes[perc_changes.Param=='ss_speed'].Delta)
    # 10MWT speed
    _10MWT_change = np.squeeze(perc_changes[perc_changes.Param=='10MWT speed'].Delta)
    # 6MWT speed
    _6MWT_change = np.squeeze(perc_changes[perc_changes.Param=='6MWT speed'].Delta)
    # TUG time
    TUG_change = np.squeeze(perc_changes[perc_changes.Param=='TUG time'].Delta)
    # LEMS
    LEMS_change = np.squeeze(perc_changes[perc_changes.Param=='LEMS'].Delta)
    # EV
    EV_change = np.squeeze(perc_changes[perc_changes.Param=='EV'].Delta)
    # A knee
    A_knee_change = np.squeeze(perc_changes[perc_changes.Param=='A knee'].Delta)
    # A hip
    A_hip_change = np.squeeze(perc_changes[perc_changes.Param=='A hip'].Delta)
    # A ankle
    A_ankle_change = np.squeeze(perc_changes[perc_changes.Param=='A ankle'].Delta)
    # ACC
    ACC_change = np.squeeze(perc_changes[perc_changes.Param=='ACC'].Delta)
    # step length
    step_length_change = np.squeeze(perc_changes[perc_changes.Param=='step length'].Delta)
    # step width
    step_width_change = np.squeeze(perc_changes[perc_changes.Param=='step width'].Delta)
    # stance phase
    stance_change = np.squeeze(perc_changes[perc_changes.Param=='stance phase'].Delta)
    # hip ROM
    hip_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='hip ROM'].Delta)
    # knee ROM
    knee_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='knee ROM'].Delta)
    # ankle ROM
    ankle_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='ankle ROM'].Delta)     
    # cadence
    cadence_change = np.squeeze(perc_changes[perc_changes.Param=='cadence'].Delta)
    # CoV Step Length
    cov_step_length_change = np.squeeze(perc_changes[perc_changes.Param=='CoV Step Length'].Delta)
    # Cov step width
    cov_step_width_change = np.squeeze(perc_changes[perc_changes.Param=='CoV Step Width'].Delta)
    # grf tp
    grf_tp_change = np.squeeze(perc_changes[perc_changes.Param=='grf tp'].Delta)
    # grf area braking
    grf_area_braking_change = np.squeeze(perc_changes[perc_changes.Param=='grf area braking'].Delta)
    # grf area propul
    grf_area_propul_change = np.squeeze(perc_changes[perc_changes.Param=='grf area propul'].Delta)
    
    
    # ACC
    ACC_early = []; ACC_late = []
    ACC_xcl = pd.read_excel("BO2STTrial/ACCs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ACC_early.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early]); ACC_late.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late])
    # step_length
    step_length_early = []; step_length_late = []
    step_length_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_length_early.append(step_length_xcl.loc[ID]['step_length_'+sides[i]+'_'+early]); step_length_late.append(step_length_xcl.loc[ID]['step_length_'+sides[i]+'_'+late])
    # CoV step length
    cov_step_length_early = []; cov_step_length_late = []
    cov_step_length_xcl = pd.read_excel("BO2STTrial/coeff_var.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        cov_step_length_early.append(cov_step_length_xcl.loc[ID]['coeff_var_'+sides[i]+'_'+early]); cov_step_length_late.append(cov_step_length_xcl.loc[ID]['coeff_var_'+sides[i]+'_'+late])
             
        
   # create dataframe with mean change and std
    percent_changes = {
      "Param": ["ss_speed", "10MWT_speed", "6MWT_speed", "TUG_time", "ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A knee", "A hip", "A ankle", "EV", "cadence", "cov step length", "cov step width", "grf tp", "grf area braking", "grf area propul"],
      "Mean": [np.mean(ss_speed_change), np.mean(_10MWT_change), np.mean(_6MWT_change), np.mean(TUG_change), np.mean(ACC_change), np.mean(step_length_change), np.mean(step_width_change), np.mean(stance_change), np.mean(hip_ROM_change), np.mean(knee_ROM_change), np.mean(ankle_ROM_change), np.mean(A_knee_change), np.mean(A_hip_change), np.mean(A_ankle_change), np.mean(EV_change), np.mean(cadence_change), np.mean(cov_step_length_change), np.mean(cov_step_width_change), np.mean(grf_tp_change), np.mean(grf_area_braking_change), np.mean(grf_area_propul_change)],
      "Std": [np.std(ss_speed_change), np.std(_10MWT_change), np.std(_6MWT_change), np.std(TUG_change), np.std(ACC_change), np.std(step_length_change), np.std(step_width_change), np.std(stance_change), np.std(hip_ROM_change), np.std(knee_ROM_change), np.std(ankle_ROM_change), np.std(A_knee_change), np.std(A_hip_change), np.std(A_ankle_change), np.std(EV_change), np.std(cadence_change), np.std(cov_step_length_change), np.std(cov_step_width_change), np.std(grf_tp_change), np.std(grf_area_braking_change), np.std(grf_area_propul_change)]
    }
    #load data into a DataFrame object:
    df_percent_changes = pd.DataFrame(percent_changes)

    # sort dataframe by decrease in mean percentage change
    df_percent_changes.sort_values(by='Mean', ascending=False, inplace=True)
    display(df_percent_changes)    
    
    # plot percentage changes for each parameter
    #fig = px.bar(df_percent_changes, x='Param', y='Mean', error_y='Std')
    fig = px.bar(df_percent_changes, x='Param', y='Mean')
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=600, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.update_yaxes(title_text="Percentage change"); fig.update_xaxes(title_text="Gait Parameters")
    fig.update_traces(marker_color='lightgrey')
    fig.show()

    
    ############################################################################################################ 
    # STATISTICAL TESTS
    ############################################################################################################ 
    # read xcl files
    ss_speed_xcl = pd.read_excel("BO2STTrial/SS_speeds.xls", header = [0], index_col = [0])
    overground_assessments_xcl = pd.read_excel("BO2STTrial/Overground_gait_assessments.xls", header = [0], index_col = [0])
    mechanics_xcl = pd.read_excel("BO2STTrial/walking_mechanics_performances.xlsx", header = [0], index_col = [0])
    A_xcl = pd.read_excel("BO2STTrial/A_scores.xlsx", header = [0], index_col = [0])
    joint_ROM_xcl = pd.read_excel("BO2STTrial/joint_ROMS.xlsx", header = [0], index_col = [0])
    step_length_width_cadence_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    stance_xcl = pd.read_excel("BO2STTrial/gait_cycle_phases.xlsx", header = [0], index_col = [0])
    cov_step_width_xcl = pd.read_excel("BO2STTrial/coeff_var_stride_width.xlsx", header = [0], index_col = [0])
    step_width_xcl = pd.read_excel("BO2STTrial/stride_length_width.xlsx", header = [0], index_col = [0])
    grf_xcl = pd.read_excel("BO2STTrial/fore_aft_grf.xlsx", header = [0], index_col = [0])
    LEMS_xcl = pd.read_excel("BO2STTrial/LEMS.xlsx", header = [0], index_col = [0])
    
    # wilcoxon test for early/late change
    t_ss_speed_wilcoxon, p_ss_speed_wilcoxon = stats.wilcoxon(ss_speed_xcl[early+'_SS_speed'].values, ss_speed_xcl[late+'_SS_speed'].values)
    t_10MWT_wilcoxon, p_10MWT_wilcoxon = stats.wilcoxon(10/overground_assessments_xcl['10MWT_time_'+early].values, 10/overground_assessments_xcl['10MWT_time_'+late].values)
    t_6MWT_wilcoxon, p_6MWT_wilcoxon = stats.wilcoxon(overground_assessments_xcl['6MWT_distance_'+early].values/(6*60), overground_assessments_xcl['6MWT_distance_'+late].values/(6*60))
    t_TUG_wilcoxon, p_TUG_wilcoxon = stats.wilcoxon(overground_assessments_xcl['TUG_time_'+early].values, overground_assessments_xcl['TUG_time_'+late].values)
    t_EV_wilcoxon, p_EV_wilcoxon = stats.wilcoxon(mechanics_xcl['EV_'+early].values, mechanics_xcl['EV_'+late].values)
    t_A_knee_wilcoxon, p_A_knee_wilcoxon = stats.wilcoxon(A_xcl['A_knee_'+early].values, A_xcl['A_knee_'+late].values)
    t_A_hip_wilcoxon, p_A_hip_wilcoxon = stats.wilcoxon(A_xcl['A_hip_'+early].values, A_xcl['A_hip_'+late].values)
    t_A_ankle_wilcoxon, p_A_ankle_wilcoxon = stats.wilcoxon(A_xcl['A_ankle_'+early].values, A_xcl['A_ankle_'+late].values)
    t_ACC_wilcoxon, p_ACC_wilcoxon = stats.wilcoxon(ACC_early, ACC_late)
    t_step_length_wilcoxon, p_step_length_wilcoxon = stats.wilcoxon(step_length_early, step_length_late)
    t_step_width_wilcoxon, p_step_width_wilcoxon = stats.wilcoxon(step_width_xcl['stride_width_'+early].values, step_width_xcl['stride_width_'+late].values)
    #t_RP_wilcoxon, p_RP_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['TUG_time_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['TUG_time_'+early].values)
    #t_hip_wilcoxon, p_hip_wilcoxon = stats.wilcoxon(joint_ROM_xcl['hip_ROM_'+early].values, joint_ROM_xcl['hip_ROM_'+late].values)
    #t_knee_wilcoxon, p_knee_wilcoxon = stats.wilcoxon(joint_ROM_xcl['knee_ROM_'+early].values, joint_ROM_xcl['knee_ROM_'+late].values)
    #t_ankle_wilcoxon, p_ankle_wilcoxon = stats.wilcoxon(joint_ROM_xcl['ankle_ROM_'+early].values, joint_ROM_xcl['ankle_ROM_'+late].values)
    t_cadence_wilcoxon, p_cadence_wilcoxon = stats.wilcoxon(step_length_width_cadence_xcl['cadence_'+early].values, step_length_width_cadence_xcl['cadence_'+late].values)
    t_stance_wilcoxon, p_stance_wilcoxon = stats.wilcoxon(stance_xcl['stance_'+early].values, stance_xcl['stance_'+late].values)
    t_cov_step_length_wilcoxon, p_cov_step_length_wilcoxon = stats.wilcoxon(cov_step_length_early, cov_step_length_late)
    t_cov_step_width_wilcoxon, p_cov_step_width_wilcoxon = stats.wilcoxon(cov_step_width_xcl['coeff_var_'+early].values, cov_step_width_xcl['coeff_var_'+late].values)
    t_grf_tp_wilcoxon, p_grf_tp_wilcoxon = stats.wilcoxon(grf_xcl['tp_'+early].values, grf_xcl['tp_'+late].values)
    t_grf_area_braking_wilcoxon, p_grf_area_braking_wilcoxon = stats.wilcoxon(grf_xcl['area_braking_'+early].values, grf_xcl['area_braking_'+late].values)
    t_grf_area_propul_wilcoxon, p_grf_area_propul_wilcoxon = stats.wilcoxon(grf_xcl['area_propul_'+early].values, grf_xcl['area_propul_'+late].values)
    t_LEMS_wilcoxon, p_LEMS_wilcoxon = stats.wilcoxon(LEMS_xcl['LEMS_'+early].values, LEMS_xcl['LEMS_'+late].values)

    # print p-values 
    print('WILCOXON TEST ' + early+'/'+late)
    print(f'p-value wilcoxon test ss speed:   {p_ss_speed_wilcoxon:.2f}')
    print(f'p-value wilcoxon test 10MWT speed:   {p_10MWT_wilcoxon:.2f}')
    print(f'p-value wilcoxon test 6MWT speed:  {p_6MWT_wilcoxon:.2f}')
    print(f'p-value wilcoxon test TUG time: {p_TUG_wilcoxon:.2f}')
    print(f'p-value wilcoxon test EV: {p_EV_wilcoxon:.2f}')
    print(f'p-value wilcoxon test stance phase: {p_stance_wilcoxon:.2f}')
    print(f'p-value wilcoxon test A knee: {p_A_knee_wilcoxon:.2f}')
    print(f'p-value wilcoxon test A hip: {p_A_hip_wilcoxon:.2f}')
    print(f'p-value wilcoxon test A ankle: {p_A_ankle_wilcoxon:.2f}')
    print(f'p-value wilcoxon test ACC: {p_ACC_wilcoxon:.2f}')
    print(f'p-value wilcoxon test step length: {p_step_length_wilcoxon:.2f}')
    print(f'p-value wilcoxon test cov step length: {p_cov_step_length_wilcoxon:.2f}')
    print(f'p-value wilcoxon test step width: {p_step_width_wilcoxon:.2f}')
    print(f'p-value wilcoxon test cov step width: {p_cov_step_width_wilcoxon:.2f}')
    print(f'p-value wilcoxon test grf tp: {p_grf_tp_wilcoxon:.2f}')
    print(f'p-value wilcoxon test grf area braking: {p_grf_area_braking_wilcoxon:.2f}')
    print(f'p-value wilcoxon test grf area propul: {p_grf_area_propul_wilcoxon:.2f}')
    print(f'p-value wilcoxon test LEMS: {p_LEMS_wilcoxon:.2f}')
    # print(f'p-value wilcoxon test STD PHI: {p_RP_wilcoxon:.2f}')
    #print(f'p-value wilcoxon test hip ROM: {p_hip_wilcoxon:.2f}')
    #print(f'p-value wilcoxon test knee ROM: {p_knee_wilcoxon:.2f}')
    #print(f'p-value wilcoxon test ankle ROM: {p_ankle_wilcoxon:.2f}')
    print(f'p-value wilcoxon test cadence: {p_cadence_wilcoxon:.2f}')
    print('------------------------------------------------------------')

   
    
def normalize_param(param):
    """
    Normalize parameter to have a mean value of 0 and standard deviation of 1.
    Input: 
        * param (list): values of the parameter to normalize according to time
    Output:
        * normalized_param (list): values of the normalized parameter according to time
    """
    
    normalized_param = (param-np.mean(param))/np.std(param)
    
    return normalized_param


def linear_reg(X, y):
    """
    Fits a linear model with coefficients Beta to minimize the residual sum of squares between the output y in the dataset, and the output predicted by the linear approximation.
    Input: 
        * X: change of a gait-related parameter
        * y: change in the outcome (walking speed, walking endurance, balance)
    Outputs: 
        * coeff (float): estimated coefficient, inform on how strongly changes in the particular gait-related parameter are related to changes in the outcome
        * score (float): coefficient of determination of the prediction (R^2)
        * p_value (float): the p-value for a hypothesis test whose null hypothesis is that the slope is zero, using Wald Test with t-distribution of the test statistic. 
        * CI95 (float): 95% confidence interval on slope (beta ± CI95)
    """
    coeff, intercept, r, p_value, stderr = stats.linregress(X, y)
    score = r**2 # compute R^2
    ts = abs(stats.t.ppf(p_value/2, len(X)-2))
    CI95 = ts*stderr
    
    return coeff, score, p_value, CI95

def multi_linear_reg(y, x):
    """
    """
    
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    
    return results


def print_multiple_linear_reg(early, late):
    """
    To evaluate the respective influence of each gait parameter on the outcome of walking speed, endurance and balance, a multiple linear regression model will be used. The $\beta$-values inform on how strongly changes in particular gait parameters are related to changes in the outcome.
    Inputs: 
        * early (string): early time point e.g. "BL" or "T0"
        * late (string): late time point e.g. "T0" or "T2"
    """

    ############################################################################################################ 
    # COMPUTE ALL DELTAS
    ############################################################################################################ 
    
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    sides = ['L', 'R', 'L', 'R', 'L', 'L'] # MA side

    perc_changes = compute_perc_change_normalized(early, late)
    # ss speed
    ss_speed_change = np.squeeze(perc_changes[perc_changes.Param=='ss_speed'].Delta)
    # 10MWT speed
    _10MWT_change = np.squeeze(perc_changes[perc_changes.Param=='10MWT speed'].Delta)
    # 6MWT speed
    _6MWT_change = np.squeeze(perc_changes[perc_changes.Param=='6MWT speed'].Delta)
    # TUG time
    TUG_change = np.squeeze(perc_changes[perc_changes.Param=='TUG time'].Delta)
    # EV
    EV_change = np.squeeze(perc_changes[perc_changes.Param=='EV'].Delta)
    # A
    A_change = np.squeeze(perc_changes[perc_changes.Param=='A'].Delta)
    # ACC
    ACC_change = np.squeeze(perc_changes[perc_changes.Param=='ACC'].Delta)
    ACC_early = []; ACC_late = []
    ACC_xcl = pd.read_excel("BO2STTrial/ACCs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ACC_early.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early]); ACC_late.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late])
    # step length
    step_length_change = np.squeeze(perc_changes[perc_changes.Param=='step length'].Delta)
    step_length_early = []; step_length_late = []
    step_length_width_cadence_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_length_early.append(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+early]); step_length_late.append(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+late])
    # step width
    step_width_change = np.squeeze(perc_changes[perc_changes.Param=='step width'].Delta)
    step_width_early = []; step_width_late = []
    for i, ID in enumerate(IDs):
        step_width_early.append(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+early]); step_width_late.append(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+late])
    # stance phase
    stance_change = np.squeeze(perc_changes[perc_changes.Param=='stance phase'].Delta)
    # hip ROM
    hip_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='hip ROM'].Delta)
    # knee ROM
    knee_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='knee ROM'].Delta)
    # ankle ROM
    ankle_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='ankle ROM'].Delta)     
    # cadence
    cadence_change = np.squeeze(perc_changes[perc_changes.Param=='cadence'].Delta)
        
          
        
    ############################################################################################################ 
    # MULTIPLE LINEAR REGRESSIONS
    ############################################################################################################ 
    
    X = [
     ACC_change,
     step_length_change,
     step_width_change,
     stance_change,
     ]
    # link with 10MWT speed
    _10MWT_multiple_lin = multi_linear_reg(_10MWT_change, X)
    print('------------------------------------------------------------------------------------------------------------')
    print('MULTIPLE LINEAR REGRESSION Y=10MWT SPEED')
    print(_10MWT_multiple_lin.summary())
    
    # link with 6MWT speed
    _6MWT_multiple_lin = multi_linear_reg(_6MWT_change, X)
    print('------------------------------------------------------------------------------------------------------------')
    print('MULTIPLE LINEAR REGRESSION Y=6MWT SPEED')
    print(_6MWT_multiple_lin.summary())
    
    # link with TUG time
    TUG_multiple_lin = multi_linear_reg(TUG_change, X)
    print('------------------------------------------------------------------------------------------------------------')
    print('MULTIPLE LINEAR REGRESSION Y=TUG TIME')
    print(TUG_multiple_lin.summary())
    
    

def print_linear_reg(early, late):
    """
    To evaluate the respective influence of each gait parameter on the outcome of walking speed, endurance and balance, a linear regression model will be used. The $\beta$-values inform on how strongly changes in particular gait parameters are related to changes in the outcome.
    Inputs: 
        * early (string): early time point e.g. "BL" or "T0"
        * late (string): late time point e.g. "T0" or "T2"
    """

    ############################################################################################################ 
    # COMPUTE ALL DELTAS
    ############################################################################################################ 
    
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    sides = ['L', 'R', 'L', 'R', 'L', 'L'] # MA side

    perc_changes = compute_perc_change(early, late)
    # ss speed
    ss_speed_change = np.squeeze(perc_changes[perc_changes.Param=='ss_speed'].Delta)
    ss_speed_change = normalize_param(ss_speed_change)
    # 10MWT speed
    _10MWT_change = np.squeeze(perc_changes[perc_changes.Param=='10MWT speed'].Delta)
    _10MWT_change = normalize_param(_10MWT_change)
    # 6MWT speed
    _6MWT_change = np.squeeze(perc_changes[perc_changes.Param=='6MWT speed'].Delta)
    _6MWT_change = normalize_param(_6MWT_change)
    # TUG time
    TUG_change = np.squeeze(perc_changes[perc_changes.Param=='TUG time'].Delta)
    TUG_change = normalize_param(TUG_change)
    # EV
    EV_change = np.squeeze(perc_changes[perc_changes.Param=='EV'].Delta)
    EV_change = normalize_param(EV_change)
    # A
    A_change = np.squeeze(perc_changes[perc_changes.Param=='A'].Delta)
    A_change = normalize_param(A_change)
    # ACC
    ACC_change = np.squeeze(perc_changes[perc_changes.Param=='ACC'].Delta)
    ACC_change = normalize_param(ACC_change)
    ACC_early = []; ACC_late = []
    ACC_xcl = pd.read_excel("BO2STTrial/ACCs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ACC_early.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early]); ACC_late.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late])
    # step length
    step_length_change = np.squeeze(perc_changes[perc_changes.Param=='step length'].Delta)
    step_length_change = normalize_param(step_length_change)
    step_length_early = []; step_length_late = []
    step_length_width_cadence_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_length_early.append(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+early]); step_length_late.append(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+late])
    # step width
    step_width_change = np.squeeze(perc_changes[perc_changes.Param=='step width'].Delta)
    step_width_change = normalize_param(step_width_change)
    step_width_early = []; step_width_late = []
    for i, ID in enumerate(IDs):
        step_width_early.append(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+early]); step_width_late.append(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+late])
    # stance phase
    stance_change = np.squeeze(perc_changes[perc_changes.Param=='stance phase'].Delta)
    stance_change = normalize_param(stance_change)
    # hip ROM
    hip_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='hip ROM'].Delta)
    hip_ROM_change = normalize_param(hip_ROM_change)
    # knee ROM
    knee_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='knee ROM'].Delta)
    knee_ROM_change = normalize_param(knee_ROM_change)
    # ankle ROM
    ankle_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='ankle ROM'].Delta)  
    ankle_ROM_change = normalize_param(ankle_ROM_change)
    # cadence
    cadence_change = np.squeeze(perc_changes[perc_changes.Param=='cadence'].Delta)
    cadence_change = normalize_param(cadence_change)
        
          
        
    ############################################################################################################ 
    # LINEAR REGRESSIONS + SPEARMAN CORRELATIONS
    ############################################################################################################ 
    
    # link with 10MWT speed
    beta_EV, r_EV, p_EV, CI_EV = linear_reg(EV_change, _10MWT_change)
    corr_EV, p_corr_EV = stats.spearmanr(EV_change, _10MWT_change)
    beta_A, r_A, p_A, CI_A = linear_reg(A_change, _10MWT_change)
    corr_A, p_corr_A = stats.spearmanr(A_change, _10MWT_change)
    beta_ACC, r_ACC, p_ACC, CI_ACC = linear_reg(ACC_change, _10MWT_change)
    corr_ACC, p_corr_ACC = stats.spearmanr(ACC_change, _10MWT_change)
    beta_step_length, r_step_length, p_step_length, CI_step_length = linear_reg(step_length_change, _10MWT_change)
    corr_step_length, p_corr_step_length = stats.spearmanr(step_length_change, _10MWT_change)
    beta_step_width, r_step_width, p_step_width, CI_step_width = linear_reg(step_width_change, _10MWT_change)
    corr_step_width, p_corr_step_width = stats.spearmanr(step_width_change, _10MWT_change)
    beta_hip_ROM, r_hip_ROM, p_hip_ROM, CI_hip_ROM = linear_reg(hip_ROM_change, _10MWT_change)
    corr_hip_ROM, p_corr_hip_ROM = stats.spearmanr(hip_ROM_change, _10MWT_change)
    beta_knee_ROM, r_knee_ROM, p_knee_ROM, CI_knee_ROM = linear_reg(knee_ROM_change, _10MWT_change)
    corr_knee_ROM, p_corr_knee_ROM = stats.spearmanr(knee_ROM_change, _10MWT_change)
    beta_ankle_ROM, r_ankle_ROM, p_ankle_ROM, CI_ankle_ROM = linear_reg(ankle_ROM_change, _10MWT_change)
    corr_ankle_ROM, p_corr_ankle_ROM = stats.spearmanr(ankle_ROM_change, _10MWT_change)
    beta_stance, r_stance, p_stance, CI_stance = linear_reg(stance_change, _10MWT_change)
    corr_stance, p_corr_stance = stats.spearmanr(stance_change, _10MWT_change)
    beta_cadence, r_cadence, p_cadence, CI_cadence = linear_reg(cadence_change, _10MWT_change)
    corr_cadence, p_corr_cadence = stats.spearmanr(cadence_change, _10MWT_change)

    # create dataframe with beta and R^2 for linear regression
    coeffs = {
      "Param": ["ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A", "EV", "cadence"],
      "Beta": [float(beta_ACC), float(beta_step_length), float(beta_step_width), float(beta_stance), float(beta_hip_ROM), float(beta_knee_ROM), float(beta_ankle_ROM), float(beta_A), float(beta_EV), float(beta_cadence)],
      "R^2": [r_ACC, r_step_length, r_step_width, r_stance, r_hip_ROM, r_knee_ROM, r_ankle_ROM, r_A, r_EV, r_cadence],
      "p": [p_ACC, p_step_length, p_step_width, p_stance, p_hip_ROM, p_knee_ROM, p_ankle_ROM, p_A, p_EV, p_cadence],
      "CI": [CI_ACC, CI_step_length, CI_step_width, CI_stance, CI_hip_ROM, CI_knee_ROM, CI_ankle_ROM, CI_A, CI_EV, CI_cadence]
    }
    df_coeffs = pd.DataFrame(coeffs)
    # sort dataframe by decrease in mean percentage change
    df_coeffs.sort_values(by='Beta', ascending=False, inplace=True)
    print("Respective influence of each gait parameter on the outcome of walking speed")
    display(df_coeffs)  
    
    corr_coeffs = {
      "Param": ["ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A", "EV", "cadence"],
      "Corr": [corr_ACC, corr_step_length, corr_step_width, corr_stance, corr_hip_ROM, corr_knee_ROM, corr_ankle_ROM, corr_A, corr_EV, corr_cadence],
      "p": [p_corr_ACC, p_corr_step_length, p_corr_step_width, p_corr_stance, p_corr_hip_ROM, p_corr_knee_ROM, p_corr_ankle_ROM, p_corr_A, p_corr_EV, p_corr_cadence]
    }
    df_corr_coeffs = pd.DataFrame(corr_coeffs)
    # sort dataframe by decrease in mean percentage change
    df_corr_coeffs.sort_values(by='Corr', ascending=False, inplace=True)
    display(df_corr_coeffs)  
    
        
    # plot figure with all corr 
    fig = make_subplots(rows=3, cols=3, subplot_titles=("EV", "A", "ACC", "Step length", "Step width", "Hip ROM", "Knee ROM", "Cadence", "Stance phase"))
    # EV
    fig.add_trace(
        go.Scatter(name="EV", x=EV_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=EV_change, y=[beta_EV*temp for temp in EV_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=1)  
    # A
    fig.add_trace(
        go.Scatter(name="A", x=A_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=A_change, y=[beta_A*temp for temp in A_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=2)
    # ACC
    fig.add_trace(
        go.Scatter(name="ACC", x=ACC_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=ACC_change, y=[beta_ACC*temp for temp in ACC_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=3)
    # Step length
    fig.add_trace(
        go.Scatter(name="step_length", x=step_length_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=step_length_change, y=[beta_step_length* temp for temp in step_length_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=1)  
    # Step width
    fig.add_trace(
        go.Scatter(name="step_width", x=step_width_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=step_width_change, y=[beta_step_width*temp for temp in step_width_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=2)
    # Hip ROM
    fig.add_trace(
        go.Scatter(name="hip_ROM", x=hip_ROM_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=hip_ROM_change, y=[beta_hip_ROM*temp for temp in hip_ROM_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=3)
    # Knee ROM
    fig.add_trace(
        go.Scatter(name="knee_ROM", x=knee_ROM_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=knee_ROM_change, y=[beta_knee_ROM*temp for temp in knee_ROM_change], mode='lines', showlegend=False, line=dict(color='grey')),row=3, col=1)  
    # Cadence
    fig.add_trace(
       go.Scatter(name="cadence", x=cadence_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=cadence_change, y=[beta_cadence*temp for temp in cadence_change], mode='lines', showlegend=False, line=dict(color='black')),row=3, col=2)
    # Stance phase
    fig.add_trace(
        go.Scatter(name="stance", x=stance_change, y=_10MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=stance_change, y=[beta_stance*temp for temp in stance_change], mode='lines', showlegend=False, line=dict(color='grey')),row=3, col=3)
    
    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=1000, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.update_traces(connectgaps=True)
    fig.show()
        
        
    # link with 6MWT speed
    beta_EV, r_EV, p_EV, CI_EV = linear_reg(EV_change, _6MWT_change)
    corr_EV, p_corr_EV = stats.spearmanr(EV_change, _6MWT_change)
    beta_A, r_A, p_A, CI_A = linear_reg(A_change, _6MWT_change)
    corr_A, p_corr_A = stats.spearmanr(A_change, _6MWT_change)
    beta_ACC, r_ACC, p_ACC, CI_ACC = linear_reg(ACC_change, _6MWT_change)
    corr_ACC, p_corr_ACC = stats.spearmanr(ACC_change, _6MWT_change)
    beta_step_length, r_step_length, p_step_length, CI_step_length = linear_reg(step_length_change, _6MWT_change)
    corr_step_length, p_corr_step_length = stats.spearmanr(step_length_change, _6MWT_change)
    beta_step_width, r_step_width, p_step_width, CI_step_width = linear_reg(step_width_change, _6MWT_change)
    corr_step_width, p_corr_step_width = stats.spearmanr(step_width_change, _6MWT_change)
    beta_hip_ROM, r_hip_ROM, p_hip_ROM, CI_hip_ROM = linear_reg(hip_ROM_change, _6MWT_change)
    corr_hip_ROM, p_corr_hip_ROM = stats.spearmanr(hip_ROM_change, _6MWT_change)
    beta_knee_ROM, r_knee_ROM, p_knee_ROM, CI_knee_ROM = linear_reg(knee_ROM_change, _6MWT_change)
    corr_knee_ROM, p_corr_knee_ROM = stats.spearmanr(knee_ROM_change, _6MWT_change)
    beta_ankle_ROM, r_ankle_ROM, p_ankle_ROM, CI_ankle_ROM = linear_reg(ankle_ROM_change, _6MWT_change)
    corr_ankle_ROM, p_corr_ankle_ROM = stats.spearmanr(ankle_ROM_change, _6MWT_change)
    beta_stance, r_stance, p_stance, CI_stance = linear_reg(stance_change, _6MWT_change)
    corr_stance, p_corr_stance = stats.spearmanr(stance_change, _6MWT_change)
    beta_cadence, r_cadence, p_cadence, CI_cadence = linear_reg(cadence_change, _6MWT_change)
    corr_cadence, p_corr_cadence = stats.spearmanr(cadence_change, _6MWT_change)

    # create dataframe with beta and R^2 for linear regression
    coeffs = {
      "Param": ["ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A", "EV", "cadence"],
      "Beta": [float(beta_ACC), float(beta_step_length), float(beta_step_width), float(beta_stance), float(beta_hip_ROM), float(beta_knee_ROM), float(beta_ankle_ROM), float(beta_A), float(beta_EV), float(beta_cadence)],
      "R^2": [r_ACC, r_step_length, r_step_width, r_stance, r_hip_ROM, r_knee_ROM, r_ankle_ROM, r_A, r_EV, r_cadence],
      "p": [p_ACC, p_step_length, p_step_width, p_stance, p_hip_ROM, p_knee_ROM, p_ankle_ROM, p_A, p_EV, p_cadence],
      "CI": [CI_ACC, CI_step_length, CI_step_width, CI_stance, CI_hip_ROM, CI_knee_ROM, CI_ankle_ROM, CI_A, CI_EV, CI_cadence]
    }
    df_coeffs = pd.DataFrame(coeffs)
    # sort dataframe by decrease in mean percentage change
    df_coeffs.sort_values(by='Beta', ascending=False, inplace=True)
    print("Respective influence of each gait parameter on the outcome of walking endurance")
    display(df_coeffs) 
    
    corr_coeffs = {
      "Param": ["ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A", "EV", "cadence"],
      "Corr": [corr_ACC, corr_step_length, corr_step_width, corr_stance, corr_hip_ROM, corr_knee_ROM, corr_ankle_ROM, corr_A, corr_EV, corr_cadence],
      "p": [p_corr_ACC, p_corr_step_length, p_corr_step_width, p_corr_stance, p_corr_hip_ROM, p_corr_knee_ROM, p_corr_ankle_ROM, p_corr_A, p_corr_EV, p_corr_cadence]
    }
    df_corr_coeffs = pd.DataFrame(corr_coeffs)
    # sort dataframe by decrease in mean percentage change
    df_corr_coeffs.sort_values(by='Corr', ascending=False, inplace=True)
    display(df_corr_coeffs)  
    
    # plot figure with all corr 
    fig = make_subplots(rows=3, cols=3, subplot_titles=("EV", "A", "ACC", "Step length", "Step width", "Hip ROM", "Knee ROM", "Cadence", "Stance phase"))
    # EV
    fig.add_trace(
        go.Scatter(name="EV", x=EV_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=EV_change, y=[beta_EV*temp for temp in EV_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=1)  
    # A
    fig.add_trace(
        go.Scatter(name="A", x=A_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=A_change, y=[beta_A*temp for temp in A_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=2)
    # ACC
    fig.add_trace(
        go.Scatter(name="ACC", x=ACC_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=ACC_change, y=[beta_ACC*temp for temp in ACC_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=3)
    # Step length
    fig.add_trace(
        go.Scatter(name="step_length", x=step_length_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=step_length_change, y=[beta_step_length*temp for temp in step_length_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=1)  
    # Step width
    fig.add_trace(
        go.Scatter(name="step_width", x=step_width_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=step_width_change, y=[beta_step_width*temp for temp in step_width_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=2)
    # Hip ROM
    fig.add_trace(
        go.Scatter(name="hip_ROM", x=hip_ROM_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=hip_ROM_change, y=[beta_hip_ROM*temp for temp in hip_ROM_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=3)
    # Knee ROM
    fig.add_trace(
        go.Scatter(name="knee_ROM", x=knee_ROM_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=knee_ROM_change, y=[beta_knee_ROM*temp for temp in knee_ROM_change], mode='lines', showlegend=False, line=dict(color='grey')),row=3, col=1)  
    # Cadence
    fig.add_trace(
        go.Scatter(name="cadence", x=cadence_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=cadence_change, y=[beta_cadence*temp for temp in cadence_change], mode='lines', showlegend=False, line=dict(color='black')),row=3, col=2)
    # Stance phase
    fig.add_trace(
        go.Scatter(name="stance", x=stance_change, y=_6MWT_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=stance_change, y=[beta_stance*temp for temp in stance_change], mode='lines', showlegend=False, line=dict(color='grey')),row=3, col=3)
    
    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=1000, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.update_traces(connectgaps=True)
    fig.show()
        
    
    
    # link with TUG time
    beta_EV, r_EV, p_EV, CI_EV = linear_reg(EV_change,TUG_change)
    corr_EV, p_corr_EV = stats.spearmanr(EV_change, TUG_change)
    beta_A, r_A, p_A, CI_A = linear_reg(A_change, TUG_change)
    corr_A, p_corr_A = stats.spearmanr(A_change, TUG_change)
    beta_ACC, r_ACC, p_ACC, CI_ACC = linear_reg(ACC_change, TUG_change)
    corr_ACC, p_corr_ACC = stats.spearmanr(ACC_change, TUG_change)
    beta_step_length, r_step_length, p_step_length, CI_step_length = linear_reg(step_length_change, TUG_change)
    corr_step_length, p_corr_step_length = stats.spearmanr(step_length_change, TUG_change)
    beta_step_width, r_step_width, p_step_width, CI_step_width = linear_reg(step_width_change, TUG_change)
    corr_step_width, p_corr_step_width = stats.spearmanr(step_width_change, TUG_change)
    beta_hip_ROM, r_hip_ROM, p_hip_ROM, CI_hip_ROM = linear_reg(hip_ROM_change, TUG_change)
    corr_hip_ROM, p_corr_hip_ROM = stats.spearmanr(hip_ROM_change, TUG_change)
    beta_knee_ROM, r_knee_ROM, p_knee_ROM, CI_knee_ROM = linear_reg(knee_ROM_change, TUG_change)
    corr_knee_ROM, p_corr_knee_ROM = stats.spearmanr(knee_ROM_change, _6MWT_change)
    beta_ankle_ROM, r_ankle_ROM, p_ankle_ROM, CI_ankle_ROM = linear_reg(ankle_ROM_change, TUG_change)
    corr_ankle_ROM, p_corr_ankle_ROM = stats.spearmanr(ankle_ROM_change, TUG_change)
    beta_stance, r_stance, p_stance, CI_stance = linear_reg(stance_change, TUG_change)
    corr_stance, p_corr_stance = stats.spearmanr(stance_change, TUG_change)
    beta_cadence, r_cadence, p_cadence, CI_cadence = linear_reg(cadence_change, TUG_change)
    corr_cadence, p_corr_cadence = stats.spearmanr(cadence_change, TUG_change)

    # create dataframe with beta and R^2 for linear regression
    coeffs = {
      "Param": ["ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A", "EV", "cadence"],
      "Beta": [float(beta_ACC), float(beta_step_length), float(beta_step_width), float(beta_stance), float(beta_hip_ROM), float(beta_knee_ROM), float(beta_ankle_ROM), float(beta_A), float(beta_EV), float(beta_cadence)],
      "R^2": [r_ACC, r_step_length, r_step_width, r_stance, r_hip_ROM, r_knee_ROM, r_ankle_ROM, r_A, r_EV, r_cadence],
      "p": [p_ACC, p_step_length, p_step_width, p_stance, p_hip_ROM, p_knee_ROM, p_ankle_ROM, p_A, p_EV, p_cadence],
      "CI": [CI_ACC, CI_step_length, CI_step_width, CI_stance, CI_hip_ROM, CI_knee_ROM, CI_ankle_ROM, CI_A, CI_EV, CI_cadence]
    }
    df_coeffs = pd.DataFrame(coeffs)
    # sort dataframe by decrease in mean percentage change
    df_coeffs.sort_values(by='Beta', ascending=False, inplace=True)
    print("Respective influence of each gait parameter on the outcome of walking balance")
    display(df_coeffs) 
    
    corr_coeffs = {
      "Param": ["ACC", "step_length", "step_width", "stance", "hip_ROM", "knee_ROM", 'ankle_ROM', "A", "EV", "cadence"],
      "Corr": [corr_ACC, corr_step_length, corr_step_width, corr_stance, corr_hip_ROM, corr_knee_ROM, corr_ankle_ROM, corr_A, corr_EV, corr_cadence],
      "p": [p_corr_ACC, p_corr_step_length, p_corr_step_width, p_corr_stance, p_corr_hip_ROM, p_corr_knee_ROM, p_corr_ankle_ROM, p_corr_A, p_corr_EV, p_corr_cadence]
    }
    df_corr_coeffs = pd.DataFrame(corr_coeffs)
    # sort dataframe by decrease in mean percentage change
    df_corr_coeffs.sort_values(by='Corr', ascending=False, inplace=True)
    display(df_corr_coeffs) 
    
        # plot figure with all corr 
    fig = make_subplots(rows=3, cols=3, subplot_titles=("EV", "A", "ACC", "Step length", "Step width", "Hip ROM", "Knee ROM", "Cadence", "Stance phase"))
    # EV
    fig.add_trace(
        go.Scatter(name="EV", x=EV_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=EV_change, y=[beta_EV*temp for temp in EV_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=1)  
    # A
    fig.add_trace(
        go.Scatter(name="A", x=A_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=A_change, y=[beta_A*temp for temp in A_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=2)
    # ACC
    fig.add_trace(
        go.Scatter(name="ACC", x=ACC_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=1, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=ACC_change, y=[beta_ACC*temp for temp in ACC_change], mode='lines', showlegend=False, line=dict(color='grey')),row=1, col=3)
    # Step length
    fig.add_trace(
        go.Scatter(name="step_length", x=step_length_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=step_length_change, y=[beta_step_length*temp for temp in step_length_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=1)  
    # Step width
    fig.add_trace(
        go.Scatter(name="step_width", x=step_width_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=step_width_change, y=[beta_step_width*temp for temp in step_width_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=2)
    # Hip ROM
    fig.add_trace(
        go.Scatter(name="hip_ROM", x=hip_ROM_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=2, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=hip_ROM_change, y=[beta_hip_ROM*temp for temp in hip_ROM_change], mode='lines', showlegend=False, line=dict(color='grey')),row=2, col=3)
    # Knee ROM
    fig.add_trace(
        go.Scatter(name="knee_ROM", x=knee_ROM_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=1)
    fig.add_trace(
        go.Scatter(name="trendline", x=knee_ROM_change, y=[beta_knee_ROM*temp for temp in knee_ROM_change], mode='lines', showlegend=False, line=dict(color='grey')),row=3, col=1)  
    # Cadence
    fig.add_trace(
       go.Scatter(name="cadence", x=cadence_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=2)
    fig.add_trace(
        go.Scatter(name="trendline", x=cadence_change, y=[beta_cadence*temp for temp in cadence_change], mode='lines', showlegend=False, line=dict(color='black')),row=3, col=2)
    # Stance phase
    fig.add_trace(
        go.Scatter(name="stance", x=stance_change, y=TUG_change, mode='markers', line=dict(color='black'), showlegend=False),row=3, col=3)
    fig.add_trace(
        go.Scatter(name="trendline", x=stance_change, y=[beta_stance*temp for temp in stance_change], mode='lines', showlegend=False, line=dict(color='grey')),row=3, col=3)
    
    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=1000, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.update_traces(connectgaps=True)
    fig.show()       

    
    
def compute_perc_change(early, late):
    """
    """
    
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    sides = ['L', 'R', 'L', 'R', 'L', 'L'] # MA side
    perc_changes = pd.DataFrame(columns = ['Param', 'Delta'])

    # ss speed
    ss_speed_change = []
    ss_speed_xcl = pd.read_excel("BO2STTrial/SS_speeds.xls", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ss_speed_change.append(compute_percentage_change(ss_speed_xcl.loc[ID][early+'_SS_speed'], ss_speed_xcl.loc[ID][late+'_SS_speed']))
    perc_changes = perc_changes.append({'Param' : 'ss_speed', 'Delta' : ss_speed_change}, ignore_index = True)

    # 10MWT speed
    _10MWT_change = []
    overground_assessments_xcl = pd.read_excel("BO2STTrial/Overground_gait_assessments.xls", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        _10MWT_change.append(compute_percentage_change(overground_assessments_xcl.loc[ID]["10MWT_speed_"+early], overground_assessments_xcl.loc[ID]["10MWT_speed_"+late]))
    perc_changes = perc_changes.append({'Param' : '10MWT speed', 'Delta' : _10MWT_change}, ignore_index = True)

    # 6MWT speed
    _6MWT_change = []
    for i, ID in enumerate(IDs):
        _6MWT_change.append(compute_percentage_change(overground_assessments_xcl.loc[ID]["6MWT_speed_"+early], overground_assessments_xcl.loc[ID]["6MWT_speed_"+late]))
    perc_changes = perc_changes.append({'Param' : '6MWT speed', 'Delta' : _6MWT_change}, ignore_index = True)

    # TUG time
    TUG_change = []
    for i, ID in enumerate(IDs):
        TUG_change.append(compute_percentage_change(overground_assessments_xcl.loc[ID]['TUG_time_'+early], overground_assessments_xcl.loc[ID]['TUG_time_'+late]))
    perc_changes = perc_changes.append({'Param' : 'TUG time', 'Delta' : TUG_change}, ignore_index = True)
      
    # LEMS
    LEMS_change = []
    LEMS_xcl = pd.read_excel("BO2STTrial/LEMS.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        LEMS_change.append(compute_percentage_change(LEMS_xcl.loc[ID]['LEMS_'+early], LEMS_xcl.loc[ID]['LEMS_'+late]))
    perc_changes = perc_changes.append({'Param' : 'LEMS', 'Delta' : LEMS_change}, ignore_index = True)
    
    # EV
    EV_change = []
    mechanics_xcl = pd.read_excel("BO2STTrial/walking_mechanics_performances.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        EV_change.append(compute_percentage_change(mechanics_xcl.loc[ID]['EV_'+early], mechanics_xcl.loc[ID]['EV_'+late]))
    perc_changes = perc_changes.append({'Param' : 'EV', 'Delta' : EV_change}, ignore_index = True)
    
    # CoV Step Length
    CoV_step_length_xcl = pd.read_excel("BO2STTrial/coeff_var.xlsx", header = [0], index_col = [0])
    CoV_step_length_change = []
    for i, ID in enumerate(IDs):
        CoV_step_length_change.append(compute_percentage_change(CoV_step_length_xcl.loc[ID]["coeff_var_"+sides[i]+'_'+early], CoV_step_length_xcl.loc[ID]["coeff_var_"+sides[i]+'_'+late]))
    perc_changes = perc_changes.append({'Param' : 'CoV Step Length', 'Delta' : CoV_step_length_change}, ignore_index = True)
    
    # Cov step width (it was my definition of stride width)
    CoV_step_width_xcl = pd.read_excel("BO2STTrial/coeff_var_stride_width.xlsx", header = [0], index_col = [0])
    CoV_step_width_change = []
    for i, ID in enumerate(IDs):
        CoV_step_width_change.append(compute_percentage_change(CoV_step_width_xcl.loc[ID]["coeff_var_"+early], CoV_step_width_xcl.loc[ID]["coeff_var_"+late]))
    perc_changes = perc_changes.append({'Param' : 'CoV Step Width', 'Delta' : CoV_step_width_change}, ignore_index = True)
    
    # A scores
    A_xcl = pd.read_excel("BO2STTrial/A_scores.xlsx", header = [0], index_col = [0])
    A_knee_change = []; A_hip_change = []; A_ankle_change = []
    for i, ID in enumerate(IDs):
        A_knee_change.append(compute_percentage_change(A_xcl.loc[ID]['A_knee_'+early], A_xcl.loc[ID]['A_knee_'+late]))
        A_hip_change.append(compute_percentage_change(A_xcl.loc[ID]['A_hip_'+early], A_xcl.loc[ID]['A_hip_'+late]))
        A_ankle_change.append(compute_percentage_change(A_xcl.loc[ID]['A_ankle_'+early], A_xcl.loc[ID]['A_ankle_'+late]))
    perc_changes = perc_changes.append({'Param' : 'A knee', 'Delta' : A_knee_change}, ignore_index = True)
    perc_changes = perc_changes.append({'Param' : 'A hip', 'Delta' : A_hip_change}, ignore_index = True)
    perc_changes = perc_changes.append({'Param' : 'A ankle', 'Delta' : A_ankle_change}, ignore_index = True)
    
    # ACC
    ACC_change = []
    ACC_xcl = pd.read_excel("BO2STTrial/ACCs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ACC_change.append(compute_percentage_change(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early], ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late]))
    perc_changes = perc_changes.append({'Param' : 'ACC', 'Delta' : ACC_change}, ignore_index = True)
        
    # step length
    step_length_change = []
    step_length_width_cadence_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_length_change.append(compute_percentage_change(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+early], step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+late]))
    perc_changes = perc_changes.append({'Param' : 'step length', 'Delta' : step_length_change}, ignore_index = True)
        
    # step width
    step_width_change = []
    step_width_xcl = pd.read_excel("BO2STTrial/stride_length_width.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_width_change.append(compute_percentage_change(step_width_xcl.loc[ID]['stride_width_'+early], step_width_xcl.loc[ID]['stride_width_'+late]))
    perc_changes = perc_changes.append({'Param' : 'step width', 'Delta' : step_width_change}, ignore_index = True) 
    
    # stance phase
    stance_change = []
    stance_xcl = pd.read_excel("BO2STTrial/gait_cycle_phases.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        stance_change.append(compute_percentage_change(stance_xcl.loc[ID]['stance_'+early], stance_xcl.loc[ID]['stance_'+late]))
    perc_changes = perc_changes.append({'Param' : 'stance phase', 'Delta' : stance_change}, ignore_index = True)
    
    # grf tp 
    grf_tp_change = []
    grf_tp_xcl = pd.read_excel("BO2STTrial/fore_aft_grf.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        grf_tp_change.append(compute_percentage_change(grf_tp_xcl.loc[ID]['tp_'+early], grf_tp_xcl.loc[ID]['tp_'+late]))
    perc_changes = perc_changes.append({'Param' : 'grf tp', 'Delta' : grf_tp_change}, ignore_index = True)
    
    # grf area braking 
    grf_area_braking_change = []
    for i, ID in enumerate(IDs):
        grf_area_braking_change.append(compute_percentage_change(grf_tp_xcl.loc[ID]['area_braking_'+early], grf_tp_xcl.loc[ID]['area_braking_'+late]))
    perc_changes = perc_changes.append({'Param' : 'grf area braking', 'Delta' : grf_area_braking_change}, ignore_index = True)
    
    # grf area propul 
    grf_area_propul_change = []
    for i, ID in enumerate(IDs):
        grf_area_propul_change.append(compute_percentage_change(grf_tp_xcl.loc[ID]['area_propul_'+early], grf_tp_xcl.loc[ID]['area_propul_'+late]))
    perc_changes = perc_changes.append({'Param' : 'grf area propul', 'Delta' : grf_area_propul_change}, ignore_index = True)
    
    # hip ROM
    hip_ROM_change = []
    joint_ROM_xcl = pd.read_excel("BO2STTrial/joint_ROMs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        hip_ROM_change.append(compute_percentage_change(joint_ROM_xcl.loc[ID]['hip_ROM_'+early], joint_ROM_xcl.loc[ID]['hip_ROM_'+late]))
    perc_changes = perc_changes.append({'Param' : 'hip ROM', 'Delta' : hip_ROM_change}, ignore_index = True)
    
    # knee ROM
    knee_ROM_change = []
    for i, ID in enumerate(IDs):
        knee_ROM_change.append(compute_percentage_change(joint_ROM_xcl.loc[ID]['knee_ROM_'+early], joint_ROM_xcl.loc[ID]['knee_ROM_'+late]))
    perc_changes = perc_changes.append({'Param' : 'knee ROM', 'Delta' : knee_ROM_change}, ignore_index = True)
    
    # ankle ROM
    ankle_ROM_change = []
    for i, ID in enumerate(IDs):
        ankle_ROM_change.append(compute_percentage_change(joint_ROM_xcl.loc[ID]['ankle_ROM_'+early], joint_ROM_xcl.loc[ID]['ankle_ROM_'+late]))
    perc_changes = perc_changes.append({'Param' : 'ankle ROM', 'Delta' : ankle_ROM_change}, ignore_index = True)    
        
    # cadence
    cadence_change = []
    for i, ID in enumerate(IDs):
        cadence_change.append(compute_percentage_change(step_length_width_cadence_xcl.loc[ID]['cadence_'+early], step_length_width_cadence_xcl.loc[ID]['cadence_'+late]))
    perc_changes = perc_changes.append({'Param' : 'cadence', 'Delta' : cadence_change}, ignore_index = True)    
    
    return perc_changes


def compute_perc_change_normalized(early, late):
    """
    """
    
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    sides = ['L', 'R', 'L', 'R', 'L', 'L'] # MA side
    perc_changes = pd.DataFrame(columns = ['Param', 'Delta'])

    # ss speed
    ss_speed_change = []
    ss_speed_xcl = pd.read_excel("BO2STTrial/SS_speeds.xls", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ss_speed_change.append(compute_percentage_change(ss_speed_xcl.loc[ID][early+'_SS_speed'], ss_speed_xcl.loc[ID][late+'_SS_speed']))
    perc_changes = perc_changes.append({'Param' : 'ss_speed', 'Delta' : ss_speed_change}, ignore_index = True)

    # 10MWT speed
    _10MWT_change = []
    overground_assessments_xcl = pd.read_excel("BO2STTrial/Overground_gait_assessments.xls", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        _10MWT_change.append(compute_percentage_change(10/overground_assessments_xcl.loc[ID]['10MWT_time_'+early+'_norm'], 10/overground_assessments_xcl.loc[ID]['10MWT_time_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : '10MWT speed', 'Delta' : _10MWT_change}, ignore_index = True)

    # 6MWT speed
    _6MWT_change = []
    for i, ID in enumerate(IDs):
        _6MWT_change.append(compute_percentage_change(overground_assessments_xcl.loc[ID]['6MWT_distance_'+early+'_norm']/(6*60), overground_assessments_xcl.loc[ID]['6MWT_distance_'+late+'_norm']/(6*60)))
    perc_changes = perc_changes.append({'Param' : '6MWT speed', 'Delta' : _6MWT_change}, ignore_index = True)

    # TUG time
    TUG_change = []
    for i, ID in enumerate(IDs):
        TUG_change.append(compute_percentage_change(overground_assessments_xcl.loc[ID]['TUG_time_'+early+'_norm'], overground_assessments_xcl.loc[ID]['TUG_time_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'TUG time', 'Delta' : TUG_change}, ignore_index = True)
        
    # EV
    EV_change = []
    mechanics_xcl = pd.read_excel("BO2STTrial/walking_mechanics_performances.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        EV_change.append(compute_percentage_change(mechanics_xcl.loc[ID]['EV_'+early+'_norm'], mechanics_xcl.loc[ID]['EV_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'EV', 'Delta' : EV_change}, ignore_index = True)
        
    # A
    A_change = []
    for i, ID in enumerate(IDs):
        A_change.append(compute_percentage_change(mechanics_xcl.loc[ID]['A_'+early+'_norm'], mechanics_xcl.loc[ID]['A_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'A', 'Delta' : A_change}, ignore_index = True)
        
    # std phi
    #std_RP_change = []
    #for i, ID in enumerate(IDs):
        #std_RP_change.append(compute_percentage_change(mechanics_xcl.loc[ID]['STD_'+early], mechanics_xcl.loc[ID]['STD_']+late))

    # ACC
    ACC_change = []; ACC_early = []; ACC_late = []
    ACC_xcl = pd.read_excel("BO2STTrial/ACCs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ACC_change.append(compute_percentage_change(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early+'_norm'], ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late+'_norm']))
        ACC_early.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early+'_norm']); ACC_late.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late+'_norm'])
    perc_changes = perc_changes.append({'Param' : 'ACC', 'Delta' : ACC_change}, ignore_index = True)
        
    # step length
    step_length_change = []; step_length_early = []; step_length_late = []
    step_length_width_cadence_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_length_change.append(compute_percentage_change(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+early+'_norm'], step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+late+'_norm']))
        step_length_early.append(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+early+'_norm']); step_length_late.append(step_length_width_cadence_xcl.loc[ID]['step_length_'+sides[i]+'_'+late+'_norm'])
    perc_changes = perc_changes.append({'Param' : 'step length', 'Delta' : step_length_change}, ignore_index = True)
        
    # step width
    step_width_change = []; step_width_early = []; step_width_late = []
    for i, ID in enumerate(IDs):
        step_width_change.append(compute_percentage_change(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+early+'_norm'], step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+late+'_norm']))
        step_width_early.append(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+early+'_norm']); step_width_late.append(step_length_width_cadence_xcl.loc[ID]['step_width_'+sides[i]+'_'+late+'_norm'])
    perc_changes = perc_changes.append({'Param' : 'step width', 'Delta' : step_width_change}, ignore_index = True) 
    
    # stance phase
    stance_change = []
    stance_xcl = pd.read_excel("BO2STTrial/gait_cycle_phases.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        stance_change.append(compute_percentage_change(stance_xcl.loc[ID]['stance_'+early+'_norm'], stance_xcl.loc[ID]['stance_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'stance phase', 'Delta' : stance_change}, ignore_index = True)
    
    # hip ROM
    hip_ROM_change = []
    joint_ROM_xcl = pd.read_excel("BO2STTrial/joint_ROMs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        hip_ROM_change.append(compute_percentage_change(joint_ROM_xcl.loc[ID]['hip_ROM_'+early+'_norm'], joint_ROM_xcl.loc[ID]['hip_ROM_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'hip ROM', 'Delta' : hip_ROM_change}, ignore_index = True)
    
    # knee ROM
    knee_ROM_change = []
    for i, ID in enumerate(IDs):
        knee_ROM_change.append(compute_percentage_change(joint_ROM_xcl.loc[ID]['knee_ROM_'+early+'_norm'], joint_ROM_xcl.loc[ID]['knee_ROM_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'knee ROM', 'Delta' : knee_ROM_change}, ignore_index = True)
    
    # ankle ROM
    ankle_ROM_change = []
    for i, ID in enumerate(IDs):
        ankle_ROM_change.append(compute_percentage_change(joint_ROM_xcl.loc[ID]['ankle_ROM_'+early+'_norm'], joint_ROM_xcl.loc[ID]['ankle_ROM_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'ankle ROM', 'Delta' : ankle_ROM_change}, ignore_index = True)    
        
    # cadence
    cadence_change = []
    for i, ID in enumerate(IDs):
        cadence_change.append(compute_percentage_change(step_length_width_cadence_xcl.loc[ID]['cadence_'+early+'_norm'], step_length_width_cadence_xcl.loc[ID]['cadence_'+late+'_norm']))
    perc_changes = perc_changes.append({'Param' : 'cadence', 'Delta' : cadence_change}, ignore_index = True)    
    
    return perc_changes
    



    
def stem_plot(early, late):
    """
    Plot the stem plot of the percentage changes for each parameter.
    Inputs:
        * early (string): early time point e.g. "BL" or "T0"
        * late (string): late time point e.g. "T0" or "T2"
    
    """
    
    ############################################################################################################ 
    # COMPUTE ALL DELTAS
    ############################################################################################################ 
    
    IDs = ["BO2ST_101", "BO2ST_102", "BO2ST_103", "BO2ST_104", "BO2ST_105", "BO2ST_106"]
    sides = ['L', 'R', 'L', 'R', 'L', 'L'] # MA side

    perc_changes = compute_perc_change(early, late)
    # ss speed
    ss_speed_change = np.squeeze(perc_changes[perc_changes.Param=='ss_speed'].Delta)
    # 10MWT speed
    _10MWT_change = np.squeeze(perc_changes[perc_changes.Param=='10MWT speed'].Delta)
    # 6MWT speed
    _6MWT_change = np.squeeze(perc_changes[perc_changes.Param=='6MWT speed'].Delta)
    # TUG time
    TUG_change = np.squeeze(perc_changes[perc_changes.Param=='TUG time'].Delta)
    # LEMS
    LEMS_change = np.squeeze(perc_changes[perc_changes.Param=='LEMS'].Delta)
    # EV
    EV_change = np.squeeze(perc_changes[perc_changes.Param=='EV'].Delta)
    # A knee
    A_knee_change = np.squeeze(perc_changes[perc_changes.Param=='A knee'].Delta)
    # A hip
    A_hip_change = np.squeeze(perc_changes[perc_changes.Param=='A hip'].Delta)
    # A ankle
    A_ankle_change = np.squeeze(perc_changes[perc_changes.Param=='A ankle'].Delta)
    # ACC
    ACC_change = np.squeeze(perc_changes[perc_changes.Param=='ACC'].Delta)
    # step length
    step_length_change = np.squeeze(perc_changes[perc_changes.Param=='step length'].Delta)
    # step width
    step_width_change = np.squeeze(perc_changes[perc_changes.Param=='step width'].Delta)
    # stance phase
    stance_change = np.squeeze(perc_changes[perc_changes.Param=='stance phase'].Delta)
    # hip ROM
    hip_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='hip ROM'].Delta)
    # knee ROM
    knee_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='knee ROM'].Delta)
    # ankle ROM
    ankle_ROM_change = np.squeeze(perc_changes[perc_changes.Param=='ankle ROM'].Delta)     
    # cadence
    cadence_change = np.squeeze(perc_changes[perc_changes.Param=='cadence'].Delta)
    # CoV Step Length
    cov_step_length_change = np.squeeze(perc_changes[perc_changes.Param=='CoV Step Length'].Delta)
    # Cov step width
    cov_step_width_change = np.squeeze(perc_changes[perc_changes.Param=='CoV Step Width'].Delta)
    # grf tp
    grf_tp_change = np.squeeze(perc_changes[perc_changes.Param=='grf tp'].Delta)
    # grf area braking
    grf_area_braking_change = np.squeeze(perc_changes[perc_changes.Param=='grf area braking'].Delta)
    # grf area propul
    grf_area_propul_change = np.squeeze(perc_changes[perc_changes.Param=='grf area propul'].Delta)
    
    
    # ACC
    ACC_early = []; ACC_late = []
    ACC_xcl = pd.read_excel("BO2STTrial/ACCs.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        ACC_early.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+early]); ACC_late.append(ACC_xcl.loc[ID]['ACC_'+sides[i]+'_'+late])
    # step_length
    step_length_early = []; step_length_late = []
    step_length_xcl = pd.read_excel("BO2STTrial/step_length_width_cadence.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        step_length_early.append(step_length_xcl.loc[ID]['step_length_'+sides[i]+'_'+early]); step_length_late.append(step_length_xcl.loc[ID]['step_length_'+sides[i]+'_'+late])
    # CoV step length
    cov_step_length_early = []; cov_step_length_late = []
    cov_step_length_xcl = pd.read_excel("BO2STTrial/coeff_var.xlsx", header = [0], index_col = [0])
    for i, ID in enumerate(IDs):
        cov_step_length_early.append(cov_step_length_xcl.loc[ID]['coeff_var_'+sides[i]+'_'+early]); cov_step_length_late.append(cov_step_length_xcl.loc[ID]['coeff_var_'+sides[i]+'_'+late])
             
    
    
    # create dataframe with mean change and std
    percent_changes = {
      "Param": ["ACC", "step_length", "step_width", "stance", "A knee", "A hip", "A ankle", "EV", "cadence", "cov step length", "cov step width", "grf tp", "grf area braking", "grf area propul", "LEMS"],
      "Mean": [np.mean(ACC_change), np.mean(step_length_change), np.mean(step_width_change), np.mean(stance_change), np.mean(A_knee_change), np.mean(A_hip_change), np.mean(A_ankle_change), np.mean(EV_change), np.mean(cadence_change), np.mean(cov_step_length_change), np.mean(cov_step_width_change), np.mean(grf_tp_change), np.mean(grf_area_braking_change), np.mean(grf_area_propul_change), np.mean(LEMS_change)],
      "Values": [ACC_change, step_length_change, step_width_change, stance_change, A_knee_change, A_hip_change, A_ankle_change, EV_change, cadence_change, cov_step_length_change, cov_step_width_change, grf_tp_change, grf_area_braking_change, grf_area_propul_change, LEMS_change]
    }
    #load data into a DataFrame object:
    df_percent_changes = pd.DataFrame(percent_changes)
    # sort dataframe by decrease in mean percentage change
    df_percent_changes.sort_values(by='Mean', ascending=False, inplace=True) 
        
    ############################################################################################################ 
    # PLOT 
    ############################################################################################################ 
    
    n = np.arange(0, 15) # number of deltas    
    

    # participant values for each delta
    x0 = [df_percent_changes.iloc[0].Values[0], df_percent_changes.iloc[1].Values[0], df_percent_changes.iloc[2].Values[0], df_percent_changes.iloc[3].Values[0], df_percent_changes.iloc[4].Values[0], df_percent_changes.iloc[5].Values[0], df_percent_changes.iloc[6].Values[0], df_percent_changes.iloc[7].Values[0], df_percent_changes.iloc[8].Values[0], df_percent_changes.iloc[9].Values[0], df_percent_changes.iloc[10].Values[0], df_percent_changes.iloc[11].Values[0], df_percent_changes.iloc[12].Values[0], df_percent_changes.iloc[13].Values[0], df_percent_changes.iloc[14].Values[0]]
    x1 = [df_percent_changes.iloc[0].Values[1], df_percent_changes.iloc[1].Values[1], df_percent_changes.iloc[2].Values[1], df_percent_changes.iloc[3].Values[1], df_percent_changes.iloc[4].Values[1], df_percent_changes.iloc[5].Values[1], df_percent_changes.iloc[6].Values[1], df_percent_changes.iloc[7].Values[1], df_percent_changes.iloc[8].Values[1], df_percent_changes.iloc[9].Values[1], df_percent_changes.iloc[10].Values[1], df_percent_changes.iloc[11].Values[1], df_percent_changes.iloc[12].Values[1], df_percent_changes.iloc[13].Values[1], df_percent_changes.iloc[14].Values[1]]
    x2 = [df_percent_changes.iloc[0].Values[2], df_percent_changes.iloc[1].Values[2], df_percent_changes.iloc[2].Values[2], df_percent_changes.iloc[3].Values[2], df_percent_changes.iloc[4].Values[2], df_percent_changes.iloc[5].Values[2], df_percent_changes.iloc[6].Values[2], df_percent_changes.iloc[7].Values[2], df_percent_changes.iloc[8].Values[2], df_percent_changes.iloc[9].Values[2], df_percent_changes.iloc[10].Values[2], df_percent_changes.iloc[11].Values[2], df_percent_changes.iloc[12].Values[2], df_percent_changes.iloc[13].Values[2], df_percent_changes.iloc[14].Values[2]]
    x3 = [df_percent_changes.iloc[0].Values[3], df_percent_changes.iloc[1].Values[3], df_percent_changes.iloc[2].Values[3], df_percent_changes.iloc[3].Values[3], df_percent_changes.iloc[4].Values[3], df_percent_changes.iloc[5].Values[3], df_percent_changes.iloc[6].Values[3], df_percent_changes.iloc[7].Values[3], df_percent_changes.iloc[8].Values[3], df_percent_changes.iloc[9].Values[3], df_percent_changes.iloc[10].Values[3], df_percent_changes.iloc[11].Values[3], df_percent_changes.iloc[12].Values[3], df_percent_changes.iloc[13].Values[3], df_percent_changes.iloc[14].Values[3]]
    x4 = [df_percent_changes.iloc[0].Values[4], df_percent_changes.iloc[1].Values[4], df_percent_changes.iloc[2].Values[4], df_percent_changes.iloc[3].Values[4], df_percent_changes.iloc[4].Values[4], df_percent_changes.iloc[5].Values[4], df_percent_changes.iloc[6].Values[4], df_percent_changes.iloc[7].Values[4], df_percent_changes.iloc[8].Values[4], df_percent_changes.iloc[9].Values[4], df_percent_changes.iloc[10].Values[4], df_percent_changes.iloc[11].Values[4], df_percent_changes.iloc[12].Values[4], df_percent_changes.iloc[13].Values[4], df_percent_changes.iloc[14].Values[4]]
    x5 = [df_percent_changes.iloc[0].Values[5], df_percent_changes.iloc[1].Values[5], df_percent_changes.iloc[2].Values[5], df_percent_changes.iloc[3].Values[5], df_percent_changes.iloc[4].Values[5], df_percent_changes.iloc[5].Values[5], df_percent_changes.iloc[6].Values[5], df_percent_changes.iloc[7].Values[5], df_percent_changes.iloc[8].Values[5], df_percent_changes.iloc[9].Values[5], df_percent_changes.iloc[10].Values[5], df_percent_changes.iloc[11].Values[5], df_percent_changes.iloc[12].Values[5], df_percent_changes.iloc[13].Values[5], df_percent_changes.iloc[14].Values[5]]
    # mean values for each delta
    x_mean = [np.mean(df_percent_changes.iloc[0].Values), np.mean(df_percent_changes.iloc[1].Values), np.mean(df_percent_changes.iloc[2].Values), np.mean(df_percent_changes.iloc[3].Values), np.mean(df_percent_changes.iloc[4].Values), np.mean(df_percent_changes.iloc[5].Values), np.mean(df_percent_changes.iloc[6].Values), np.mean(df_percent_changes.iloc[7].Values), np.mean(df_percent_changes.iloc[8].Values), np.mean(df_percent_changes.iloc[9].Values), np.mean(df_percent_changes.iloc[10].Values), np.mean(df_percent_changes.iloc[11].Values), np.mean(df_percent_changes.iloc[12].Values), np.mean(df_percent_changes.iloc[13].Values),
np.mean(df_percent_changes.iloc[14].Values)]

    fig, ax = plt.subplots(figsize=(20,11))

    ax.stem(n, x0, linefmt = 'lightgrey')
    ax.plot(n, x0, 'o', markersize=16, color='lightgrey')
    ax.stem(n, x1, linefmt = 'lightgrey')
    ax.plot(n, x1, 'o', markersize=16, color='lightgrey')
    ax.stem(n, x2, linefmt = 'lightgrey')
    ax.plot(n, x2, 'o', markersize=16, color='lightgrey')
    ax.stem(n, x3, linefmt = 'lightgrey')
    ax.plot(n, x3, 'o', markersize=16, color='lightgrey')
    ax.stem(n, x4, linefmt = 'lightgrey')
    ax.plot(n, x4, 'o', markersize=16, color='lightgrey')
    ax.stem(n, x5, linefmt = 'lightgrey')
    ax.plot(n, x5, 'o', markersize=16, color='lightgrey')
    ax.stem(n, x_mean, 'lightgrey')
    ax.plot(n, x_mean, 'o', markersize=16, color='black')

    ax.spines[['right', 'top']].set_visible(False)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [df_percent_changes.iloc[0].Param, df_percent_changes.iloc[1].Param, df_percent_changes.iloc[2].Param, df_percent_changes.iloc[3].Param, df_percent_changes.iloc[4].Param, df_percent_changes.iloc[5].Param, df_percent_changes.iloc[6].Param, df_percent_changes.iloc[7].Param, df_percent_changes.iloc[8].Param, df_percent_changes.iloc[9].Param, df_percent_changes.iloc[10].Param, df_percent_changes.iloc[11].Param, df_percent_changes.iloc[12].Param, df_percent_changes.iloc[13].Param, df_percent_changes.iloc[14].Param], rotation=-45)  # Set text labels and properties.
    plt.show()
    