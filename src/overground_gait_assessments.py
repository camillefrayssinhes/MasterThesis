import numpy as np
import plotly.io as pio
import pandas as pd
import matplotlib.pyplot as plt
import natsort
import math
from scipy import stats
import pathlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def overground_gait_assessments():
    """
    Compute the mean and std of the differences in time and distance for the 10MWT, 6MWT and TUG overground gait assessments between T0 and BL assessments (separating participants who use or do not use AD) and plot them in a bar plot as well as individual data. 
    """
    
    # read xcl file
    file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    overground_gait_assessments_xcl = pd.read_excel(file_name, header = [0], index_col = [0])

    # compute relative diff between BL and T0 and add columns to df
    overground_gait_assessments_xcl['10MWT_time_diff'] = (overground_gait_assessments_xcl['10MWT_time_T0'] - overground_gait_assessments_xcl['10MWT_time_BL'])
    overground_gait_assessments_xcl['6MWT_distance_diff'] = (overground_gait_assessments_xcl['6MWT_distance_T0'] - overground_gait_assessments_xcl['6MWT_distance_BL'])
    overground_gait_assessments_xcl['TUG_time_diff'] = (overground_gait_assessments_xcl['TUG_time_T0'] - overground_gait_assessments_xcl['TUG_time_BL'])

   # compute mean and std percentage diff bipedal participants
    _10MWT_time_diff_mean_biped = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['10MWT_time_diff'].mean()
    _10MWT_time_diff_std_biped = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['10MWT_time_diff'].std()
    _6MWT_distance_diff_mean_biped = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['6MWT_distance_diff'].mean()
    _6MWT_distance_diff_std_biped = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['6MWT_distance_diff'].std()
    _TUG_time_diff_mean_biped = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['TUG_time_diff'].mean()
    _TUG_time_diff_std_biped = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['TUG_time_diff'].std()
    # compute mean and std percentage diff quadrup participants
    _10MWT_time_diff_mean_quadrup = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['10MWT_time_diff'].mean()
    _10MWT_time_diff_std_quadrup = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['10MWT_time_diff'].std()
    _6MWT_distance_diff_mean_quadrup = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['6MWT_distance_diff'].mean()
    _6MWT_distance_diff_std_quadrup = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['6MWT_distance_diff'].std()
    _TUG_time_diff_mean_quadrup = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['TUG_time_diff'].mean()
    _TUG_time_diff_std_quadrup = overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['TUG_time_diff'].std()
    
    # figures
    fig, ax = plt.subplots(3,2,figsize= (15,18))
    barWidth = 0.1

    # 10MWT
    ax[0,0].bar(np.arange(len([_10MWT_time_diff_mean_biped],)), _10MWT_time_diff_mean_biped, color = 'black', width = barWidth, yerr = _10MWT_time_diff_std_biped, capsize = 5, zorder=2, label="woAD")
    ax[0,0].bar([x + barWidth for x in np.arange(len([_10MWT_time_diff_mean_biped],))], _10MWT_time_diff_mean_quadrup, color = 'grey', width = barWidth, yerr = _10MWT_time_diff_std_quadrup, capsize = 5, zorder=2, label="wAD")
    ax[0,0].axhline(0, color = 'black', linestyle = "--")
    ax[0,0].set_ylabel("Speed 10MWT Time [sec] relative to BL")
    ax[0,0].legend()
    ax[0,0].set_xticks([])
    ax[0,0].set_xlim([-0.2,0.3])
    ax[0,0].set_ylim([-2,3])
    ax[0,0].spines[['right', 'top', 'bottom']].set_visible(False)

    for i in range(len(overground_gait_assessments_xcl['10MWT_time_diff'].values)):
        data = [0, overground_gait_assessments_xcl['10MWT_time_diff'].values[i]]
        ax[0,1].plot(["BL","T0"],data, label=overground_gait_assessments_xcl.index[i], color="C"+str(i))
        ax[0,1].plot(["BL","T0"],data, 'o',color="C"+str(i))
    ax[0,1].legend(title = "Participants")
    ax[0,1].axhline(0, color = 'black', linestyle = "--")
    ax[0,1].set_xticks([])
    ax[0,1].set_ylim([-2,3])
    ax[0,1].set_yticks([])
    ax[0,1].spines[['left','right', 'top', 'bottom']].set_visible(False)

    # 6MWT
    ax[1,0].bar(np.arange(len([_6MWT_distance_diff_mean_biped],)), _6MWT_distance_diff_mean_biped, color = 'black', width = barWidth, yerr = _6MWT_distance_diff_std_biped, capsize = 5, zorder=2, label="woAD")
    ax[1,0].bar([x + barWidth for x in np.arange(len([_6MWT_distance_diff_mean_biped],))], _6MWT_distance_diff_mean_quadrup, color = 'grey', width = barWidth, yerr = _6MWT_distance_diff_std_quadrup, capsize = 5, zorder=2, label="wAD")
    ax[1,0].axhline(0, color = 'black', linestyle = "--")
    ax[1,0].set_ylabel("Speed 6MWT Distance [metre] relative to BL")
    ax[1,0].set_xticks([])
    ax[1,0].set_xlim([-0.2,0.3])
    ax[1,0].set_ylim([0,65])
    ax[1,0].spines[['right', 'top', 'bottom']].set_visible(False)

    for i in range(len(overground_gait_assessments_xcl['6MWT_distance_diff'].values)):
        data = [0, overground_gait_assessments_xcl['6MWT_distance_diff'].values[i]]
        ax[1,1].plot(["BL","T0"],data, label=overground_gait_assessments_xcl.index[i], color="C"+str(i))
        ax[1,1].plot(["BL","T0"],data, 'o',color="C"+str(i))
    ax[1,1].axhline(0, color = 'black', linestyle = "--")
    ax[1,1].set_xticks([])
    ax[1,1].set_ylim([0,65])
    ax[1,1].set_yticks([])
    ax[1,1].spines[['left','right', 'top', 'bottom']].set_visible(False)

    # TUG
    ax[2,0].bar(np.arange(len([_TUG_time_diff_mean_biped],)), _TUG_time_diff_mean_biped, color = 'black', width = barWidth, yerr = _TUG_time_diff_std_biped, capsize = 5, zorder=2, label="woAD")
    ax[2,0].bar([x + barWidth for x in np.arange(len([_TUG_time_diff_mean_biped],))], _TUG_time_diff_mean_quadrup, color = 'grey', width = barWidth, yerr = _TUG_time_diff_std_quadrup, capsize = 5, zorder=2, label="wAD")
    ax[2,0].axhline(0, color = 'black', linestyle = "--")
    ax[2,0].set_ylabel("Speed TUG Time [sec] relative to BL")
    ax[2,0].set_xticks([])
    ax[2,0].set_xlim([-0.2,0.3])
    ax[2,0].set_ylim([-3.5,0.5])
    ax[2,0].spines[['right', 'top', 'bottom']].set_visible(False)

    for i in range(len(overground_gait_assessments_xcl['TUG_time_diff'].values)):
        data = [0, overground_gait_assessments_xcl['TUG_time_diff'].values[i]]
        ax[2,1].plot(["BL","T0"],data, label=overground_gait_assessments_xcl.index[i], color="C"+str(i))
        ax[2,1].plot(["BL","T0"],data, 'o',color="C"+str(i))
    ax[2,1].axhline(0, color = 'black', linestyle = "--")
    ax[2,1].set_ylim([-3.5,0.5])
    ax[2,1].set_yticks([])
    ax[2,1].set_xlabel("Time (Days)")
    ax[2,1].spines[['left','right', 'top', 'bottom']].set_visible(False)

    # figure parameters
    fig.show()
    fig.tight_layout();

    # wilcoxon test for T0/BL
    t_10MWT_biped_wilcoxon, p_10MWT_biped_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['10MWT_time_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['10MWT_time_T0'].values)
    t_6MWT_biped_wilcoxon, p_6MWT_biped_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['6MWT_distance_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['6MWT_distance_T0'].values)
    t_TUG_biped_wilcoxon, p_TUG_biped_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['TUG_time_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['TUG_time_T0'].values)
    t_10MWT_quadrup_wilcoxon, p_10MWT_quadrup_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['10MWT_time_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['10MWT_time_T0'].values)
    t_6MWT_quadrup_wilcoxon, p_6MWT_quadrup_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['6MWT_distance_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['6MWT_distance_T0'].values)
    t_TUG_quadrup_wilcoxon, p_TUG_quadrup_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['TUG_time_BL'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['TUG_time_T0'].values)

    print('WILCOXON TEST BL/T0')
    print(f'p-value wilcoxon test 10MWT time woAD:  {p_10MWT_biped_wilcoxon:.2f}')
    print(f'p-value wilcoxon test 6MWT distance woAD: {p_6MWT_biped_wilcoxon:.2f}')
    print(f'p-value wilcoxon test TUG time woAD: {p_TUG_biped_wilcoxon:.2f}')
    print('------------------------------------------------------------')
    print(f'p-value wilcoxon test 10MWT time wAD:  {p_10MWT_quadrup_wilcoxon:.2f}')
    print(f'p-value wilcoxon test 6MWT distance wAD: {p_6MWT_quadrup_wilcoxon:.2f}')
    print(f'p-value wilcoxon test TUG time wAD: {p_TUG_quadrup_wilcoxon:.2f}')
    print('------------------------------------------------------------')
    print(' ')
    
    
    # Kruskal-Wallis H-test for woAD/wAD
    stats_10MWT_kruskal, p_10MWT_kruskal = stats.kruskal(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['10MWT_time_diff'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['10MWT_time_diff'].values)
    stats_6MWT_kruskal, p_6MWT_kruskal = stats.kruskal(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['6MWT_distance_diff'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['6MWT_distance_diff'].values)
    stats_TUG_kruskal, p_TUG_kruskal = stats.kruskal(overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==False]['TUG_time_diff'].values, overground_gait_assessments_xcl.loc[overground_gait_assessments_xcl['Quadrupedal']==True]['TUG_time_diff'].values)

    print('KRUSKAL-WALLIS H-TEST woAD/wAD')
    print(f'p-value Kruskal-Wallis H-test 10MWT: {p_10MWT_kruskal:.2f}')
    print(f'p-value Kruskal-Wallis H-test 6MWT: {p_6MWT_kruskal:.2f}')
    print(f'p-value Kruskal-Wallis H-test TUG: {p_TUG_kruskal:.2f}')


def violin_plots(time_point):
    """
    Inputs:
        * time_point (string): time point of the assessment you want to compare to baseline, e.g. "T0" or "T2"
    Compute the mean and std of the differences in time and distance for the 10MWT, 6MWT and TUG overground gait assessments between the time point and BL assessments and plot them in a violin plot along box plot as well as individual data. 
    """
    
    # read xcl file
    file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    overground_gait_assessments_xcl = pd.read_excel(file_name, header = [0], index_col = [0])
    file_name_ss_speed = ("BO2STTrial/SS_speeds.xls")
    ss_speed_xcl = pd.read_excel(file_name_ss_speed, header = [0], index_col = [0])

    # compute relative diff between BL and T0 and add columns to df
    overground_gait_assessments_xcl['10MWT_speed_diff'] = (overground_gait_assessments_xcl['10MWT_speed_'+time_point] - overground_gait_assessments_xcl['10MWT_speed_BL'])
    overground_gait_assessments_xcl['6MWT_speed_diff'] = (overground_gait_assessments_xcl['6MWT_speed_'+time_point] - overground_gait_assessments_xcl['6MWT_speed_BL'])
    overground_gait_assessments_xcl['TUG_time_diff'] = (overground_gait_assessments_xcl['TUG_time_'+time_point] - overground_gait_assessments_xcl['TUG_time_BL'])
    ss_speed_xcl['ss_speed_diff'] = (ss_speed_xcl[time_point+'_SS_speed'] - ss_speed_xcl['BL_SS_speed'])
    
    fig = make_subplots(rows=1, cols=4)
    
    # 10MWT test speed
    fig.add_trace(
        go.Violin(y=overground_gait_assessments_xcl['10MWT_speed_diff'], box_visible=True, line_color='black', meanline_visible=True, fillcolor='grey', opacity=1, points='all', x0='10MWT speed difference'),
        row=1, col=1)
    
    # 6MWT test speed
    fig.add_trace(
        go.Violin(y=overground_gait_assessments_xcl['6MWT_speed_diff'], box_visible=True, line_color='black', meanline_visible=True, fillcolor='grey', opacity=1, points='all', x0='6MWT speed difference'),
        row=1, col=2)
    
    # TUG test
    fig.add_trace(
        go.Violin(y=overground_gait_assessments_xcl['TUG_time_diff'], box_visible=True, line_color='black', meanline_visible=True, fillcolor='grey', opacity=1, points='all', x0='TUG time difference'),
        row=1, col=3)
    
    # ss-speed
    fig.add_trace(
        go.Violin(y=ss_speed_xcl['ss_speed_diff'], box_visible=True, line_color='black', meanline_visible=True, fillcolor='grey', opacity=1, points='all', x0='self-selected speed difference'),
        row=1, col=4)

    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=450, width=1200, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.add_hline(y=0, line_color='black', line_dash="dash")
    fig.show()
    
    # WILCOXON TEST BL/T0
    t_10MWT_wilcoxon, p_10MWT_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl['10MWT_speed_BL'].values, overground_gait_assessments_xcl['10MWT_speed_'+time_point].values, alternative='less')
    t_6MWT_wilcoxon, p_6MWT_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl['6MWT_speed_BL'].values, overground_gait_assessments_xcl['6MWT_speed_'+time_point].values, alternative='less')
    t_TUG_wilcoxon, p_TUG_wilcoxon = stats.wilcoxon(overground_gait_assessments_xcl['TUG_time_BL'].values, overground_gait_assessments_xcl['TUG_time_'+time_point].values, alternative='greater')
    t_ss_wilcoxon, p_ss_wilcoxon = stats.wilcoxon(ss_speed_xcl['BL_SS_speed'].values, ss_speed_xcl[time_point+'_SS_speed'].values, alternative='less')

    print('WILCOXON TEST BL/T0')
    print(f'p-value wilcoxon test 10MWT speed:  {p_10MWT_wilcoxon:.2f}')
    print(f'p-value wilcoxon test 6MWT speed: {p_6MWT_wilcoxon:.2f}')
    print(f'p-value wilcoxon test TUG time: {p_TUG_wilcoxon:.2f}')
    print(f'p-value wilcoxon test ss-speed: {p_ss_wilcoxon:.2f}')
    print('------------------------------------------------------------')
    print(' ')
    
    
def individual_progress(ID):
    """
    Plot the progress of the participant in terms of functional walking (10MWT, 6MWT, TUG) relative to T0 according to time.
    Inputs:
        * ID (string): ID of the participant, e.g. "BO2ST_101"
    """
    
    # read xcl files
    overground_gait_assessments_file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    overground_gait_assessments_xcl = pd.read_excel(overground_gait_assessments_file_name, header = [0], index_col = [0])
    _10MWT_file_name = ("BO2STTrial/10MWT.xlsx")
    _10MWT_xcl = pd.read_excel(_10MWT_file_name, header = [0], index_col = [0])
    _10MWT_xcl = 10/_10MWT_xcl # convert to speed 
    
    # read physiological data
    weight = overground_gait_assessments_xcl.loc[ID]['Weight'] #kg
    height = overground_gait_assessments_xcl.loc[ID]['Height'] #cm
    age = overground_gait_assessments_xcl.loc[ID]['Age']
    sex = overground_gait_assessments_xcl.loc[ID]['Sex']
    
    # compute normative reference value (NRF) for this subject
    if (sex == 'F'):
        # 6MWT
        _6MWT_NRF = 2.11*height - 2.29*weight - 5.78*age + 667
        _6MWT_NRF_LLN = _6MWT_NRF - 139
        # 10MWT
        if (20<=age<30):
            _10MWT_NRF = 1.502*10**(-2)*height
            _10MWT_NRF_std = 0.142*10**(-2)*height
        elif (30<=age<40):
            _10MWT_NRF = 1.428*10**(-2)*height
            _10MWT_NRF_std = 0.206*10**(-2)*height
        elif (40<=age<50):
            _10MWT_NRF = 1.304*10**(-2)*height
            _10MWT_NRF_std = 0.160*10**(-2)*height
        elif (50<=age<60):
            _10MWT_NRF = 1.243*10**(-2)*height
            _10MWT_NRF_std = 0.158*10**(-2)*height
        elif (60<=age<70):
            _10MWT_NRF = 1.107*10**(-2)*height
            _10MWT_NRF_std = 0.157*10**(-2)*height
        elif (70<=age<80):
            _10MWT_NRF = 1.110*10**(-2)*height
            _10MWT_NRF_std = 0.176*10**(-2)*height
        
    elif (sex == 'M'):   
        # 6MWT
        _6MWT_NRF = 7.57*height - 5.02*age - 1.76*weight - 309
        _6MWT_NRF_LLN = _6MWT_NRF - 153
        # 10MWT
        if (20<=age<30):
            _10MWT_NRF = 1.431*10**(-2)*height
            _10MWT_NRF_std = 0.162*10**(-2)*height
        elif (30<=age<40):
            _10MWT_NRF = 1.396*10**(-2)*height
            _10MWT_NRF_std = 0.177*10**(-2)*height
        elif (40<=age<50):
            _10MWT_NRF = 1.395*10**(-2)*height
            _10MWT_NRF_std = 0.197*10**(-2)*height
        elif (50<=age<60):
            _10MWT_NRF = 1.182*10**(-2)*height
            _10MWT_NRF_std = 0.259*10**(-2)*height
        elif (60<=age<70):
            _10MWT_NRF = 1.104*10**(-2)*height
            _10MWT_NRF_std = 0.198*10**(-2)*height
        elif (70<=age<80):
            _10MWT_NRF = 1.192*10**(-2)*height
            _10MWT_NRF_std = 0.201*10**(-2)*height
    
    # TUG
    if (20<=age<30):
        TUG_NRF = 8.57
        TUG_NRF_std = 1.40
    elif (30<=age<40):
        TUG_NRF = 8.56
        TUG_NRF_std = 1.23
    elif (40<=age<50):
        TUG_NRF = 8.86
        TUG_NRF_std = 1.88
    elif (age >=50):
        TUG_NRF = 9.90
        TUG_NRF_std = 2.29
       
    x = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'] # time points
    _10MWT_speed = 10/overground_gait_assessments_xcl.loc[ID][['10MWT_time_BL', '10MWT_time_T0', '10MWT_time_T1', '10MWT_time_T2', '10MWT_time_F1', '10MWT_time_F4', '10MWT_time_F8']]
    _10MWT_y = _10MWT_speed - _10MWT_speed[1] # relative to T0
    _10MWT_NRF = _10MWT_NRF - _10MWT_speed[1] # relative to T0
    _6MWT_y = overground_gait_assessments_xcl.loc[ID][['6MWT_distance_BL', '6MWT_distance_T0', '6MWT_distance_T1', '6MWT_distance_T2', '6MWT_distance_F1', '6MWT_distance_F4', '6MWT_distance_F8']] - overground_gait_assessments_xcl.loc[ID]['6MWT_distance_T0'] # relative to T0
    _6MWT_NRF = _6MWT_NRF - overground_gait_assessments_xcl.loc[ID]['6MWT_distance_T0'] # relative to T0
    TUG_y = overground_gait_assessments_xcl.loc[ID][['TUG_time_BL', 'TUG_time_T0', 'TUG_time_T1', 'TUG_time_T2', 'TUG_time_F1', 'TUG_time_F4', 'TUG_time_F8']] - overground_gait_assessments_xcl.loc[ID]['TUG_time_T0'] # relative to T0
    TUG_NRF = TUG_NRF - overground_gait_assessments_xcl.loc[ID]['TUG_time_T0'] # relative to T0

    # plot 
    fig = make_subplots(rows=1, cols=3, x_title = 'Time', subplot_titles=("10MWT", "6MWT", "TUG"), horizontal_spacing=0.115)

    # 10MWT test
    y_assessments_10MWT = [_10MWT_xcl.loc[ID]['BL'], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['T0'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['T1'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['T2'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['F1'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['F4'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['F8']] - _10MWT_xcl.loc[ID]['T0']
    
    y_daily_10MWT = _10MWT_xcl.loc[ID].values - _10MWT_xcl.loc[ID]['T0']
    
    trace_10MWT_assessments = go.Scatter(
    x=_10MWT_xcl.loc[ID].index,
    y=np.array(y_assessments_10MWT),
    name='Assessments 10MWT',
    mode = 'lines+markers',
    marker=dict(color='black')
    )
    
    trace_10MWT_daily = go.Scatter(
        x=_10MWT_xcl.loc[ID].index,
        y=y_daily_10MWT,
        name='Daily 10MWT',
        mode = 'markers',
        marker = dict(color='black')
    )
    
    #fig.add_trace(
        #go.Scatter(name="Upper bound", x=_10MWT_xcl.loc[ID].index, y=np.repeat(_10MWT_NRF + _10MWT_NRF_std, len(_10MWT_xcl.loc[ID].index)), mode='lines',marker=dict(color="grey"), fillcolor='lightgrey',
        #fill=None, showlegend=False, line=dict(width=0)),row=1, col=1)   
    #fig.add_trace(
        #go.Scatter(name="Lower Bound", x=_10MWT_xcl.loc[ID].index, y=np.repeat(_10MWT_NRF - _10MWT_NRF_std, len(_10MWT_xcl.loc[ID].index)),mode='lines',marker=dict(color="grey"), fillcolor='lightgrey',
        #fill='tonexty', showlegend=False, line=dict(width=0, color='lightgrey')),row=1, col=1) 
    #fig.add_trace(
        #go.Scatter(name = "10MWT AB matched", x=_10MWT_xcl.loc[ID].index, y=np.repeat(_10MWT_NRF, len(_10MWT_xcl.loc[ID].index)), mode = "lines", showlegend=False, line=dict(color='darkgrey')),
        #row=1, col=1)
    fig.add_trace(trace_10MWT_assessments, row=1, col=1)
    fig.add_trace(trace_10MWT_daily, row=1, col=1)

    # 6MWT test
    #fig.add_trace(
        #go.Scatter(name = "6MWT AB matched", x=x, y=np.repeat(_6MWT_NRF, len(_6MWT_y))/(6*60), mode = "lines", showlegend=False, line=dict(color='darkgrey')),
        #row=1, col=2)
    #fig.add_trace(
        #go.Scatter(name="Lower limit of Normal Range", x=x, y=np.repeat(_6MWT_NRF - _6MWT_NRF_LLN, len(_6MWT_y))/(6*60), mode='lines',marker=dict(color="light grey"), fillcolor='lightgrey',
        #fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=2) 
    fig.add_trace(
        go.Scatter(name = "6MWT SCI", x=x, y=_6MWT_y/(6*60), showlegend=False, line=dict(color='black')),
        row=1, col=2)

    # TUG test
    #fig.add_trace(
        #go.Scatter(name="Upper bound", x=x, y=np.repeat(TUG_NRF + TUG_NRF_std, len(TUG_y)), mode='lines',marker=dict(color="lightgrey"), fillcolor='light grey',
        #fill=None, showlegend=False, line=dict(width=0)),row=1, col=3)   
    #fig.add_trace(
        #go.Scatter(name="Lower Bound", x=x, y=np.repeat(TUG_NRF - TUG_NRF_std, len(TUG_y)),mode='lines',marker=dict(color="lightgrey"), fillcolor='lightgrey',
        #fill='tonexty', showlegend=False, line=dict(width=0, color='lightgrey')),row=1, col=3)
    #fig.add_trace(
        #go.Scatter(name = "TUG AB matched", x=x, y=np.repeat(TUG_NRF, len(TUG_y)), mode = "lines", showlegend=False, line=dict(color='darkgrey')),
        #row=1, col=3)
    fig.add_trace(
        go.Scatter(name = "TUG SCI", x=x, y=TUG_y, showlegend=False, line=dict(color='black')),
        row=1, col=3)
    

    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=450, width=1000, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')
    fig.add_hline(y=0, line_color='black', line_dash="dash")
    fig['layout']['xaxis1'].update(
        tickmode = 'array',
        tickvals = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'],
        ticktext = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8']
    )
    fig.update_traces(connectgaps=True)
    # Update yaxis properties
    fig.update_yaxes(title_text="10MWT speed [m/s]", row=1, col=1)
    fig.update_yaxes(title_text="6MWT speed [m/s]", row=1, col=2)
    fig.update_yaxes(title_text="TUG time [s]", row=1, col=3)
    fig.show()
    
    
    
    
def individual_progress_raw_data(ID):
    """
    Plot the progress of the participant in terms of functional walking (10MWT, 6MWT, TUG) relative to T0 according to time.
    Inputs:
        * ID (string): ID of the participant, e.g. "BO2ST_101"
    """
    
    # read xcl files
    overground_gait_assessments_file_name = ("BO2STTrial/Overground_gait_assessments.xls")
    overground_gait_assessments_xcl = pd.read_excel(overground_gait_assessments_file_name, header = [0], index_col = [0])
    _10MWT_file_name = ("BO2STTrial/10MWT.xlsx")
    _10MWT_xcl = pd.read_excel(_10MWT_file_name, header = [0], index_col = [0])
    _10MWT_xcl = 10/_10MWT_xcl # convert to speed 
    
    # read physiological data
    weight = overground_gait_assessments_xcl.loc[ID]['Weight'] #kg
    height = overground_gait_assessments_xcl.loc[ID]['Height'] #cm
    age = overground_gait_assessments_xcl.loc[ID]['Age']
    sex = overground_gait_assessments_xcl.loc[ID]['Sex']
    
    # compute normative reference value (NRF) for this subject
    if (sex == 'F'):
        # 6MWT
        _6MWT_NRF = 2.11*height - 2.29*weight - 5.78*age + 667
        _6MWT_NRF_LLN = _6MWT_NRF - 139
        # 10MWT
        if (20<=age<30):
            _10MWT_NRF = 1.502*10**(-2)*height
            _10MWT_NRF_std = 0.142*10**(-2)*height
        elif (30<=age<40):
            _10MWT_NRF = 1.428*10**(-2)*height
            _10MWT_NRF_std = 0.206*10**(-2)*height
        elif (40<=age<50):
            _10MWT_NRF = 1.304*10**(-2)*height
            _10MWT_NRF_std = 0.160*10**(-2)*height
        elif (50<=age<60):
            _10MWT_NRF = 1.243*10**(-2)*height
            _10MWT_NRF_std = 0.158*10**(-2)*height
        elif (60<=age<70):
            _10MWT_NRF = 1.107*10**(-2)*height
            _10MWT_NRF_std = 0.157*10**(-2)*height
        elif (70<=age<80):
            _10MWT_NRF = 1.110*10**(-2)*height
            _10MWT_NRF_std = 0.176*10**(-2)*height
        
    elif (sex == 'M'):   
        # 6MWT
        _6MWT_NRF = 7.57*height - 5.02*age - 1.76*weight - 309
        _6MWT_NRF_LLN = _6MWT_NRF - 153
        # 10MWT
        if (20<=age<30):
            _10MWT_NRF = 1.431*10**(-2)*height
            _10MWT_NRF_std = 0.162*10**(-2)*height
        elif (30<=age<40):
            _10MWT_NRF = 1.396*10**(-2)*height
            _10MWT_NRF_std = 0.177*10**(-2)*height
        elif (40<=age<50):
            _10MWT_NRF = 1.395*10**(-2)*height
            _10MWT_NRF_std = 0.197*10**(-2)*height
        elif (50<=age<60):
            _10MWT_NRF = 1.182*10**(-2)*height
            _10MWT_NRF_std = 0.259*10**(-2)*height
        elif (60<=age<70):
            _10MWT_NRF = 1.104*10**(-2)*height
            _10MWT_NRF_std = 0.198*10**(-2)*height
        elif (70<=age<80):
            _10MWT_NRF = 1.192*10**(-2)*height
            _10MWT_NRF_std = 0.201*10**(-2)*height
    
    # TUG
    if (20<=age<30):
        TUG_NRF = 8.57
        TUG_NRF_std = 1.40
    elif (30<=age<40):
        TUG_NRF = 8.56
        TUG_NRF_std = 1.23
    elif (40<=age<50):
        TUG_NRF = 8.86
        TUG_NRF_std = 1.88
    elif (age >=50):
        TUG_NRF = 9.90
        TUG_NRF_std = 2.29
       
    x = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'] # time points
    _10MWT_speed = 10/overground_gait_assessments_xcl.loc[ID][['10MWT_time_BL', '10MWT_time_T0', '10MWT_time_T1', '10MWT_time_T2', '10MWT_time_F1', '10MWT_time_F4', '10MWT_time_F8']]
    _10MWT_y = _10MWT_speed 
    #_10MWT_NRF = _10MWT_NRF 
    _6MWT_y = overground_gait_assessments_xcl.loc[ID][['6MWT_distance_BL', '6MWT_distance_T0', '6MWT_distance_T1', '6MWT_distance_T2', '6MWT_distance_F1', '6MWT_distance_F4', '6MWT_distance_F8']]
    #_6MWT_NRF = _6MWT_NRF
    TUG_y = overground_gait_assessments_xcl.loc[ID][['TUG_time_BL', 'TUG_time_T0', 'TUG_time_T1', 'TUG_time_T2', 'TUG_time_F1', 'TUG_time_F4', 'TUG_time_F8']]
    #TUG_NRF = TUG_NRF

    # plot 
    fig = make_subplots(rows=1, cols=3, x_title = 'Time', subplot_titles=("10MWT", "6MWT", "TUG"), horizontal_spacing=0.115)

    # 10MWT test
    y_assessments_10MWT = [_10MWT_xcl.loc[ID]['BL'], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['T0'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['T1'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['T2'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['F1'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['F4'], np.nan, np.nan, np.nan, np.nan, _10MWT_xcl.loc[ID]['F8']]
    
    y_daily_10MWT = _10MWT_xcl.loc[ID].values
    
    trace_10MWT_assessments = go.Scatter(
    x=_10MWT_xcl.loc[ID].index,
    y=np.array(y_assessments_10MWT),
    name='Assessments 10MWT',
    mode = 'lines+markers',
    marker=dict(color='black')
    )
    
    trace_10MWT_daily = go.Scatter(
        x=_10MWT_xcl.loc[ID].index,
        y=y_daily_10MWT,
        name='Daily 10MWT',
        mode = 'markers',
        marker = dict(color='black')
    )
    
    fig.add_trace(
        go.Scatter(name="Upper bound", x=_10MWT_xcl.loc[ID].index, y=np.repeat(_10MWT_NRF + _10MWT_NRF_std, len(_10MWT_xcl.loc[ID].index)), mode='lines',marker=dict(color="grey"), fillcolor='lightgrey',
        fill=None, showlegend=False, line=dict(width=0)),row=1, col=1)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=_10MWT_xcl.loc[ID].index, y=np.repeat(_10MWT_NRF - _10MWT_NRF_std, len(_10MWT_xcl.loc[ID].index)),mode='lines',marker=dict(color="grey"), fillcolor='lightgrey',
        fill='tonexty', showlegend=False, line=dict(width=0, color='lightgrey')),row=1, col=1) 
    fig.add_trace(
        go.Scatter(name = "10MWT AB matched", x=_10MWT_xcl.loc[ID].index, y=np.repeat(_10MWT_NRF, len(_10MWT_xcl.loc[ID].index)), mode = "lines", showlegend=False, line=dict(color='darkgrey')),
        row=1, col=1)
    fig.add_trace(trace_10MWT_assessments, row=1, col=1)
    fig.add_trace(trace_10MWT_daily, row=1, col=1)

    # 6MWT test
    fig.add_trace(
        go.Scatter(name = "6MWT AB matched", x=x, y=np.repeat(_6MWT_NRF, len(_6MWT_y))/(6*60), mode = "lines", showlegend=False, line=dict(color='darkgrey')),
        row=1, col=2)
    fig.add_trace(
        go.Scatter(name="Lower limit of Normal Range", x=x, y=np.repeat(_6MWT_NRF - _6MWT_NRF_LLN, len(_6MWT_y))/(6*60), mode='lines',marker=dict(color="light grey"), fillcolor='lightgrey',
        fill='tonexty', showlegend=False, line=dict(width=0)),row=1, col=2) 
    fig.add_trace(
        go.Scatter(name = "6MWT SCI", x=x, y=_6MWT_y/(6*60), showlegend=False, line=dict(color='black')),
        row=1, col=2)

    # TUG test
    fig.add_trace(
        go.Scatter(name="Upper bound", x=x, y=np.repeat(TUG_NRF + TUG_NRF_std, len(TUG_y)), mode='lines',marker=dict(color="lightgrey"), fillcolor='light grey',
        fill=None, showlegend=False, line=dict(width=0)),row=1, col=3)   
    fig.add_trace(
        go.Scatter(name="Lower Bound", x=x, y=np.repeat(TUG_NRF - TUG_NRF_std, len(TUG_y)),mode='lines',marker=dict(color="lightgrey"), fillcolor='lightgrey',
        fill='tonexty', showlegend=False, line=dict(width=0, color='lightgrey')),row=1, col=3)
    fig.add_trace(
        go.Scatter(name = "TUG AB matched", x=x, y=np.repeat(TUG_NRF, len(TUG_y)), mode = "lines", showlegend=False, line=dict(color='darkgrey')),
        row=1, col=3)
    fig.add_trace(
        go.Scatter(name = "TUG SCI", x=x, y=TUG_y, showlegend=False, line=dict(color='black')),
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
    fig.update_yaxes(title_text="10MWT speed [m/s]", row=1, col=1)
    fig.update_yaxes(title_text="6MWT speed [m/s]", row=1, col=2)
    fig.update_yaxes(title_text="TUG time [s]", row=1, col=3)
    fig.show()
    
def individual_progress_ss_speed(ID):
    """
    Plot the progress of the participant in terms of functional walking (10MWT, 6MWT, TUG) relative to T0 according to time.
    Inputs:
        * ID (string): ID of the participant, e.g. "BO2ST_101"
    """    
    
    # read xcl file
    ss_speeds_file_name = ("BO2STTrial/SS_Speeds.xls")
    ss_speeds_xcl = pd.read_excel(ss_speeds_file_name, header = [0], index_col = [0])   
    
    # plot ss speed
    x = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'] # time points
    fig = make_subplots(rows=1, cols=1, x_title = 'Time', horizontal_spacing=0.115)
    fig.add_trace(
            go.Scatter(name = "ss speed", x=x, y=ss_speeds_xcl.loc[ID], mode = "lines", showlegend=False, line=dict(color='black')),
            row=1, col=1)
    # fig params
    fig.update_layout(yaxis_zeroline=False, plot_bgcolor="rgba(0,0,0,0)", height=450, width=600, showlegend=False)
    fig.update_xaxes(linecolor='black'); fig.update_yaxes(linecolor='black', ticks='outside')

    fig['layout']['xaxis1'].update(
        tickmode = 'array',
        tickvals = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8'],
        ticktext = ['BL', 'T0', 'T1', 'T2', 'F1', 'F4', 'F8']
    )
    fig.update_traces(connectgaps=True)
    # Update yaxis properties
    fig.update_yaxes(title_text="Self-selected Speed [m/s]", row=1, col=1)
    fig.show() 