# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from utils import summarise, getRisk
from parameters import baseline

st.title("T2D Survival")
col1, col2 = st.columns(2)
col1.subheader("Patient 1")
col2.subheader("Patient 2")


Sex = col1.selectbox("Patient Sex:", ["Male", "Female"])
if Sex == "Male":
    male = 1
elif Sex == "Female":
    male = 0

aged = col1.number_input("Age at T2D diagnosis:", value=65., step=0.1)
imd = col1.number_input("IMD (deciles of index of multiple deprivation):", value=6, step=1)
indexdate_n = col1.number_input("indexdate_n (T2DM diagnosis date, in days from 1-1-2000):", value=2, step=1, min_value=1, max_value=10*365)

Ethn = col1.selectbox("Patient Ethnicity:",
	["White", "Black", "South_East_Asian", "Other_Asian", "Mixed", "Chinese", "Other"])
if Ethn == "White":
    White = 1
    Black  = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0    
elif Ethn == "Black":
    White = 0
    Black = 1
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0
elif Ethn == "South_East_Asian":
    White = 0
    Black = 0
    South_East_Asian = 1
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0
elif Ethn == "Other_Asian":
    White = 0
    Black = 0
    South_East_Asian = 0
    Other_Asian = 1
    Mixed = 0
    Chinese = 0
    Other = 0
elif Ethn == "Mixed":
    White = 0
    Black = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 1
    Chinese = 0
    Other = 0
elif Ethn == "Chinese":
    White = 0
    Black = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 1
    Chinese = 0
    Other = 0
elif Ethn == "Other":
    White = 1
    Black  = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0
    
XColumns = ['residuals_mean', 'coeff', 'indexdate_n', 'aged', 'imd', 'hba1c_value',
       'White', 'Black', 'South_East_Asian', 'Other_Asian', 'Mixed', 'Chinese',
       'Other', 'male', 'tyears', 'prediction']

newPatientCharact = {
        "aged": aged,
        'indexdate_n': indexdate_n,
        'imd': imd,
        'White': White,
        'Black': Black,
        'South_East_Asian': South_East_Asian,
        'Other_Asian': Other_Asian,
        'Mixed': Mixed,
        'Chinese': Chinese,
        'Other': Other,
        'male': male, #,        'tyears': 1.1 #time elapsed from T2D diagnosis will be taken by newPatientHBA1c
    }
    

# Initialize session state
if "num_inputs" not in col1.session_state:
    col1.session_state.num_inputs = 3
num = col1.slider("How many HbA1c measurements?", 1, 10, col1.session_state.num_inputs)
col1.session_state.num_inputs 
newPatientHBA1c = {}

value = col1.number_input(f"HbA1c value at diagnosis:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
newPatientHBA1c[0] = value

for i in range(1, num):
    key = col1.number_input(f"Time of observation {i+1} (years from diagnosis):", step=0.1, value=float(i))
    value = col1.number_input(f"HbA1c value {i+1}:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
    if key:  # Only add if key is not empty
        newPatientHBA1c[key] = value

blup = summarise(newPatientCharact, newPatientHBA1c) 
Xnew = {key: blup[key] for key in XColumns}

newPatientSummary = np.array(list(Xnew.values()), dtype=float).reshape(1, -1)
risk1 = getRisk(newPatientSummary)
col1.write("Risk 1: ", risk1)








Sex = col2.selectbox("Patient Sex:", ["Male", "Female"])
if Sex == "Male":
    male = 1
elif Sex == "Female":
    male = 0

aged = col2.number_input("Age at T2D diagnosis:", value=65., step=0.1)
imd = col2.number_input("IMD (deciles of index of multiple deprivation):", value=6, step=1)
indexdate_n = col2.number_input("indexdate_n (T2DM diagnosis date, in days from 1-1-2000):", value=2, step=1, min_value=1, max_value=10*365)

Ethn = col2.selectbox("Patient Ethnicity:",
	["White", "Black", "South_East_Asian", "Other_Asian", "Mixed", "Chinese", "Other"])
if Ethn == "White":
    White = 1
    Black  = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0    
elif Ethn == "Black":
    White = 0
    Black = 1
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0
elif Ethn == "South_East_Asian":
    White = 0
    Black = 0
    South_East_Asian = 1
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0
elif Ethn == "Other_Asian":
    White = 0
    Black = 0
    South_East_Asian = 0
    Other_Asian = 1
    Mixed = 0
    Chinese = 0
    Other = 0
elif Ethn == "Mixed":
    White = 0
    Black = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 1
    Chinese = 0
    Other = 0
elif Ethn == "Chinese":
    White = 0
    Black = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 1
    Chinese = 0
    Other = 0
elif Ethn == "Other":
    White = 1
    Black  = 0
    South_East_Asian = 0
    Other_Asian = 0
    Mixed = 0
    Chinese = 0
    Other = 0
    
XColumns = ['residuals_mean', 'coeff', 'indexdate_n', 'aged', 'imd', 'hba1c_value',
       'White', 'Black', 'South_East_Asian', 'Other_Asian', 'Mixed', 'Chinese',
       'Other', 'male', 'tyears', 'prediction']

newPatientCharact = {
        "aged": aged,
        'indexdate_n': indexdate_n,
        'imd': imd,
        'White': White,
        'Black': Black,
        'South_East_Asian': South_East_Asian,
        'Other_Asian': Other_Asian,
        'Mixed': Mixed,
        'Chinese': Chinese,
        'Other': Other,
        'male': male, #,        'tyears': 1.1 #time elapsed from T2D diagnosis will be taken by newPatientHBA1c
    }
    

# Initialize session state
if "num_inputs" not in col2.session_state:
    col2.session_state.num_inputs = 3
num = col2.slider("How many HbA1c measurements?", 1, 10, col2.session_state.num_inputs)
col2.session_state.num_inputs 
newPatientHBA1c = {}

value = col2.number_input(f"HbA1c value at diagnosis:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
newPatientHBA1c[0] = value

for i in range(1, num):
    key = col2.number_input(f"Time of observation {i+1} (years from diagnosis):", step=0.1, value=float(i))
    value = col2.number_input(f"HbA1c value {i+1}:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
    if key:  # Only add if key is not empty
        newPatientHBA1c[key] = value

blup = summarise(newPatientCharact, newPatientHBA1c) 
Xnew = {key: blup[key] for key in XColumns}

newPatientSummary = np.array(list(Xnew.values()), dtype=float).reshape(1, -1)
risk2 = getRisk(newPatientSummary)
col2.write("Risk 2: ", risk2)



rate1 = baseline['hazard'] * risk1
prob1 = 100 * (1 - np.exp(-np.cumsum(rate1)))
    
rate2 = baseline['hazard'] * risk2
prob2 = 100 * (1 - np.exp(-np.cumsum(rate2)))



fig, ax = plt.subplots()
ax.plot(baseline['time'], prob1, lw=2, label='Patient 1')
ax.plot(baseline['time'], prob2, lw=2, label='Patient 2')
ax.set_ylabel('Patient probability of death [\%]')
ax.set_xlabel('Years from last HBa1c measurement')
ax.set_ylim([0,100])
ax.legend()
ax.grid(True)
st.pyplot(fig)
   
