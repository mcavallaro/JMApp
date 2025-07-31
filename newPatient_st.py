#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import streamlit as st
import utils as ut
import paramters as pr

st.title("T2D Survival")

Sex = st.selectbox("Patient Sex:", ["Male", "Female"])
if Sex == "Male":
    male = 1
elif Sex == "Female":
    male = 0

aged = st.number_input("Age:", value=65., step=0.1)
imd = st.number_input("IMD:", value=6, step=1)
indexdate_n = st.number_input("indexdate_n:", value=2, step=1)

Ethn = st.selectbox("Patient Ethnicity:",
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
if "num_inputs" not in st.session_state:
    st.session_state.num_inputs = 3
num = st.slider("How HbA1c measurements?", 1, 10, st.session_state.num_inputs)
st.session_state.num_inputs 
newPatientHBA1c = {}
for i in range(num):
    key = st.number_input(f"Time of observation (years from Diagnosis) {i+1}:", value=float(i + 1), step=0.1)
    value = st.number_input(f"HbA1c value {i+1}:", value=float(4 + i), step=0.1)
    if key:  # Only add if key is not empty
        newPatientHBA1c[key] = value

blup = ut.summarise(newPatientCharact, newPatientHBA1c) 
Xnew = {key: blup[key] for key in XColumns}

newPatientSummary = np.array(list(Xnew.values()), dtype=float).reshape(1, -1)
risk = ut.getRisk(newPatientSummary)
st.write("Risk: ", risk)

rate = pr.baseline.hazard * risk    
prob = 100 * (1 - np.exp(-np.cumsum(rate)))
    
fig, ax = plt.subplots()

ax.plot(pr.baseline.time, prob, lw=2)
ax.set_ylabel('Patient probability of death [\%]')
ax.set_xlabel('Years from last HBa1c measurement')
    
ax.grid(True)

st.pyplot(fig)
   
