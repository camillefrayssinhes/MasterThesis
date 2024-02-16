# MasterThesis
This repository hosts the code I have developed and used for the analysis of the data for my EPFL master thesis conducted at the Inspire Lab at Harvard Medical School. 

# Characterisation of walking function in individuals with incomplete spinal cord injury

## Overview

Incomplete spinal cord injury (iSCI) refers to partial damage to the spinal cord, disrupting sensory, motor, and autonomic functions. The resulting partial loss of walking ability is a significant concern, impacting independence and overall quality of life. Restoring walking after iSCI is crucial not only for reducing secondary complications, but also for promoting mental health and social reintegration.
Current clinical assessments of walking function after iSCI primarily focus on speed measures, considered as the most responsive to improvements in walking capacity. However, understanding the underlying mechanics of observed speed deficits, and monitoring the impact of rehabilitation interventions on walking mechanics remain a challenge. Existing studies explore specific aspects of walking mechanics, but lack a comprehensive understanding of the multifaceted impairments faced by individuals with SCI in their walking recovery journey. This absence of a holistic overview creates a significant gap in identifying and addressing individual gait deficits post-injury.
The project introduces a comprehensive methodology to evaluate walking recovery after iSCI, aiming to establish a robust framework for personalised assessments encompassing functional walking, kinematics, and kinetics. We apply this methodology to 7 individuals with iSCI to characterise the impacts of rehabilitation, neuromodulatory interventions, and changes in walking speed on gait quality. By offering insights into individual gait deficits, encompassing force production, gait coordination, consistency, symmetry, and spatiotemporal aspects of gait, this project represents a step toward more tailored therapeutic approaches.
Additionally, with the ultimate goal of providing an accessible clinical tool, the project assesses OpenCap’s markerless motion capture system accuracy against the gold-standard marker-based system. While promising, OpenCap requires further investigation for accurate kinematics and kinetics estimation specific to iSCI individuals.

## Table of Contents

- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Analysis](#analysis)
- [Results](#results)


## Usage

Data were analysed using Python 3.8.12.

Execute the two provided Jupyter notebooks ([Data_Analysis](/Data_Analysis.ipynb) and [OpenCap](/OpenCap.ipynb)) to initiate the generation of data and plots. These notebooks serve as the primary interface for interacting with the project. The code required for the Jupyter notebooks is organized in the [src](/src) and [src_opencap](/src_opencap) directories.

## Project Structure

- [src](/src): Main source code directory
  - [ACC.py](/src/ACC.py): Quantify hip-knee coordination by computing the angular component of the coefficient of correspondence (ACC).
  - [bilateral_cyclogram.py](/src/bilateral_cyclogram.py): Quantify asymmetry using bilateral cyclograms for lower extremity joint angles (hip, knee, ankle).
  - [compute_gait_cycle_phases.py](/src/compute_gait_cycle_phases.py): Compute stance and swing phases.
  - [compute_step_width_length.py](/src/compute_step_width_length.py): Compute step/stride width and step/stride length.
  - [CRP.py](/src/CRP.py): Compute the continuous relative phase between two joint angles time series (e.g. shoulder and hip).
  - [dataloader.py](/src/dataloader.py): Load the kinematic and kinetic data of the walking trials and calibration trials.
  - [endpoint_variability.py](/src/endpoint_variability.py): Quantify the consistency of the step-to-step footpath trajectory by computing the endpoint variability.
  - [gait_cycle.py](/src/gait_cycle.py): Compute and normalize gait cycle.
  - [GRF.py](/src/GRF.py): Compute ground reaction forces and extract specific features (e.g. braking area, propulsive area, transition point).
  - [kinematics.py](/src/kinematics.py): Extract the range of motion of the hip, knee and ankle joint angles.
  - [lin_reg.py](/src/lin_reg.py): Assess the responsiveness of gait-related parameters to parameters of functional walking (speed, endurance, balance).
  - [overground_gait_assessments.py](/src/overground_gait_assessments.py): Assess overground walking performances (speed, endurance, balance).
  - [RP.py](/src/RP.py): Compute the relative phase between two joint angles time series (e.g. shoulder and hip).
  - [spider_chart.py](/src/spider_chart.py): Construct spider charts (aka polar plots) to depicts a comprehensive overview of strengths and weaknesses for each participant.
- [src_opencap](/src_opencap): Source code directory for comparing the accuracy of OpenCap estimated kinetics to gold standard marker based Vicon MoCap system.
  - [dataloader_opencap.py](/src_opencap/dataloader_opencap.py): Load the kinematic data of both Vicon and OpenCap for the walking trials and calibration trials.
  - [angles.py](/src_opencap/angles.py): Resample, synchronise, and compute the MAE of lower extremities joint angle time series between Vicon and OpenCap.
- [Data_Analysis](/Data_Analysis.ipynb): Jupyter notebook dedicated to generating data and plots for the master thesis. This notebook specifically examines the walking mechanics of individuals with iSCI and AB controls, and produce the finalised polar plots used to assess the evolution of gait quality throughout rehabilitation, interventions, and changes in walking speed. 
- [OpenCap](/OpenCap.ipynb): Jupyter notebook responsible for generating the data and plots utilised in the master thesis. This notebook focuses on comparing the accuracy of OpenCap-estimated joint angles with those obtained from Vicon.


## Data

The raw data employed in this project were gathered within the framework of a study at the Inspire Lab, Spaulding Rehabilitation Hospital Cambridge.
Eight persons with chronic motor iSCI participated in the study. However, data from only 7 participants were analysed due to numerous occlusions encountered during the data capture process for one participant. 
To establish normative values for walking mechanics parameters, a publicly available dataset containing kinematic and kinetic data from healthy volunteers walking both overground and on a treadmill at various gait speeds was employed to identify matched AB controls for each participant.
iSCI and AB data can be downloaded here: https://partnershealthcare-my.sharepoint.com/:f:/r/personal/cfrayssinhes_mgh_harvard_edu/Documents/BO2STTrial?csf=1&web=1&e=zrTvCd

## Results

This work introduces a methodology for a comprehensive assessment of the walking patterns of individuals with iSCI. The employed gait-related parameters complemented conventional measures of walking by offering insights into force production, coordination, consistency, symmetry, and spatiotemporal aspects of gait. The combination of these parameters has the potential to shed light on compensatory mechanisms, fatigue, and rhythm issues. Consequently, the diverse gait impairments within this heterogeneous population can be characterised in a more objective and comprehensive manner than with current clinical practices. OpenCap software holds the promise for translating this methodology into a user-friendly clinical tool for gait assessments; however, further investigations are required to assess the accuracy of estimated kinematics specific to the challenges faced by the SCI population during walking. Additionally, estimating kinetics remains computationally expensive, currently limiting its applicability in clinical settings. This research is a step toward improved monitoring of the impacts of rehabilitation or treatment on walking recovery, laying the foundation for more personalised therapeutic approaches.

## Acknowledgments

This Master's Thesis marks the culmination of my studies in Life Sciences Engineering at the EPFL. My academic journey at the EPFL has been a remarkable experience, and I am extremely grateful to have concluded this chapter at Harvard Medical School within the Inspire Lab team.

I would like to thank Professor Randy Trumbower for giving me the opportunity to work under his guidance, fostering my passion and understanding in the field of neuro-engineering, and for challenging me throughout this thesis. Thank you Randy for granting me the freedom to explore the field of walking recovery after a spinal cord injury.

Also, many thanks to Christopher Tuthill for his unconditional support, numerous proof-readings of the thesis, and constant availability. Merci beaucoup Chris for your invaluable input on my work, and for the engaging discussions.

Many thanks to the vibrant lab community, who even while chasing deadlines, consistently maintains a positive atmosphere.  Your support and the shared  moments such as shared meals, birthday celebrations, Christmas cookie competitions, and escape rooms have added a delightful dimension to this thesis.

My gratitude extends to my EPFL supervisor, Professor Grégoire Courtine. It has been a true honour to have you as a mentor throughout this fulfilling journey.

Last but not least, I would like to express my deepest appreciation to the study participants for their courage, trust, and unwavering enthusiasm.

