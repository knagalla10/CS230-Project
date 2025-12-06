# Code submission for CS230 Final Project:

### GenerateData
This contains the workflow for generating training datasets by running the submission1.sh script on the FarmShare cluster. 
The inputs.txt file contains the ranges of each thermal parameter to generate combinations from.The thermal_model directory 
contains the scripts for the heat diffusion model used for fitting actual TDTR signals (ThermalResponse.m + MatrixQuadrupole.m).
These are used to simulate response signals for each generated parameter combination (InitializeData.m) and apply a noise signal 
to them (pinknoise.m). The submission1.sh scipt runs the GenerateData.m script which uses these codes to generate an array of 
examples and organize them into training, development, and test sets (data directory).

### define_model.py
This script contains the code for the proposed approach to test residual connections to enable deeper networks. The submission2.sh 
script runs this script on the FarmShare cluster, which unfortunately limited what architechtures we could test.
