# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from utils import summarise, getRisk
from parameters import baseline

st.title("T2D Survival")
col1, col2 = st.columns(2)
col1.subheader("Patient A")
col2.subheader("Patient B")

with col1:
    Sex = st.selectbox("A) Patient Sex:", ["Male", "Female"])
    if Sex == "Male":
        male = 1
    elif Sex == "Female":
        male = 0

    aged = st.number_input("A) Age at T2D diagnosis:", value=65., step=0.1)
    imd = st.number_input("A) IMD (deciles of index of multiple deprivation):", value=6, step=1, min_value=1, max_value=10)
    indexdate_n = st.number_input("A) indexdate_n (T2DM diagnosis date, in days from 1-1-2000):", value=2, step=1, min_value=1, max_value=10*365)

    Ethn = st.selectbox("A) Ethnicity:",
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
    num = st.slider("A) How many HbA1c measurements?", 1, 10, st.session_state.num_inputs)
    st.session_state.num_inputs 
    newPatientHBA1c = {}

    value = st.number_input(f"A) HbA1c value at diagnosis:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
    newPatientHBA1c[0] = value

    for i in range(1, num):
        key = st.number_input(f"A) Time of observation {i+1} (years from diagnosis):", step=0.1, value=float(i))
        value = st.number_input(f"A) HbA1c value {i+1}:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
        if key:  # Only add if key is not empty
            newPatientHBA1c[key] = value

    blup = summarise(newPatientCharact, newPatientHBA1c) 
    Xnew = {key: float(blup[key]) for key in XColumns}

    newPatientSummary = np.array(list(Xnew.values()), dtype=float).reshape(1, -1)
    x = pd.DataFrame(newPatientSummary, columns=Xnew.keys())
    risk1 = getRisk(x)
    st.write("A) Risk: ", risk1)


with col2:
    Sex2 = st.selectbox("B) Patient Sex:", ["Male", "Female"])
    if Sex2 == "Male":
        male2 = 1
    elif Sex2 == "Female":
        male2 = 0

    aged2 = st.number_input("B) Age at T2D diagnosis:", value=65., step=0.1)
    imd2 = st.number_input("B) IMD (deciles of index of multiple deprivation):", value=6, step=1, min_value=1, max_value=10)
    indexdate_n2 = st.number_input("B) indexdate_n (T2DM diagnosis date, in days from 1-1-2000):", value=2, step=1, min_value=1, max_value=10*365)

    Ethn2 = st.selectbox("B) Patient Ethnicity:",
	    ["White", "Black", "South_East_Asian", "Other_Asian", "Mixed", "Chinese", "Other"])
    if Ethn2 == "White":
        White2 = 1
        Black2  = 0
        South_East_Asian2 = 0
        Other_Asian2 = 0
        Mixed2 = 0
        Chinese2 = 0
        Other2 = 0    
    elif Ethn2 == "Black":
        White2 = 0
        Black2 = 1
        South_East_Asian2 = 0
        Other_Asian2 = 0
        Mixed2 = 0
        Chinese2 = 0
        Other2 = 0
    elif Ethn2 == "South_East_Asian":
        White2 = 0
        Black2 = 0
        South_East_Asian2 = 1
        Other_Asian2 = 0
        Mixed2 = 0
        Chinese2 = 0
        Other2 = 0
    elif Ethn2 == "Other_Asian":
        White2 = 0
        Black2 = 0
        South_East_Asian2 = 0
        Other_Asian2 = 1
        Mixed2 = 0
        Chinese2 = 0
        Other2 = 0
    elif Ethn2 == "Mixed":
        White2 = 0
        Black2 = 0
        South_East_Asian2 = 0
        Other_Asian2 = 0
        Mixed2 = 1
        Chinese2 = 0
        Other2 = 0
    elif Ethn2 == "Chinese":
        White2 = 0
        Black2 = 0
        South_East_Asian2 = 0
        Other_Asian2 = 0
        Mixed2 = 1
        Chinese2 = 0
        Other2 = 0
    elif Ethn2 == "Other":
        White2 = 1
        Black2  = 0
        South_East_Asian2 = 0
        Other_Asian2 = 0
        Mixed2 = 0
        Chinese2 = 0
        Other2 = 0
        
    newPatientCharact2 = {
            "aged": aged2,
            'indexdate_n': indexdate_n2,
            'imd': imd2,
            'White': White2,
            'Black': Black2,
            'South_East_Asian': South_East_Asian2,
            'Other_Asian': Other_Asian2,
            'Mixed': Mixed2,
            'Chinese': Chinese2,
            'Other': Other2,
            'male': male2, #,        'tyears': 1.1 #time elapsed from T2D diagnosis will be taken by newPatientHBA1c
        }
        

    # Initialize session state
    if "num_inputs" not in st.session_state:
        st.session_state.num_inputs = 3
    num2 = st.slider("B) How many HbA1c measurements?", 1, 10, st.session_state.num_inputs)
    st.session_state.num_inputs 
    newPatientHBA1c2 = {}

    value2 = st.number_input(f"B) HbA1c value at diagnosis:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
    newPatientHBA1c2[0] = value2

    for i in range(1, num2):
        key2 = st.number_input(f"B) Time of observation {i+1} (years from diagnosis):", step=0.1, value=float(i))
        value2 = st.number_input(f"B) HbA1c value {i+1}:", step=0.1, value=float(4), min_value=float(4), max_value=float(20))
        if key2:  # Only add if key is not empty
            newPatientHBA1c2[key2] = value2

    blup2 = summarise(newPatientCharact2, newPatientHBA1c2) 
    Xnew2 = {key: blup2[key] for key in XColumns}

    newPatientSummary2 = np.array(list(Xnew2.values()), dtype=float).reshape(1, -1)
    x2 = pd.DataFrame(newPatientSummary2, columns=Xnew2.keys())
    risk2 = getRisk(x2)
    st.write("B) Risk: ", risk2)


rate1 = baseline['hazard'] * risk1
prob1 = 100 * (1 - np.exp(-np.cumsum(rate1)))
   
rate2 = baseline['hazard'] * risk2
prob2 = 100 * (1 - np.exp(-np.cumsum(rate2)))

fig, ax = plt.subplots()
ax_right = ax.twinx()
time_from_Dx1 = list(newPatientHBA1c.keys())
value1 = list(newPatientHBA1c.values())
time_from_Dx2 = list(newPatientHBA1c2.keys())
value2 = list(newPatientHBA1c2.values())

#
ax.plot(time_from_Dx1, value1, linestyle='--', marker="o", color='tab:blue')
ax.plot(time_from_Dx2, value2, linestyle='--', marker="o", color='tab:orange')
#
ax.axvline(time_from_Dx1[-1], '--', alpha=0.5, color='tab:blue')
ax.axvline(time_from_Dx1[-1], '--', alpha=0.5, color='tab:orange')
#
ax_right.plot(baseline['time'] + time_from_Dx1[-1], prob1, lw=2, color='tab:blue', label='Patient A')
ax_right.plot(baseline['time'] + time_from_Dx2[-1], prob2, lw=2, color='tab:orange', label='Patient B')
ax_right.set_ylabel('Patient probability of death [\%]')
ax.set_xlabel('Years from T2D diagnosis')
ax.set_ylabel('HbA1c value')
ax_right.set_ylim([0,100])
ax.set_ylim(min(value1+value2), max(value1+value2))
ax_right.legend()
ax.grid(True)
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(baseline['time'], prob1, lw=2, color='tab:blue', label='Patient A')
ax.plot(baseline['time'], prob2, lw=2, color='tab:orange', label='Patient B')
ax.set_ylabel('Patient probability of death [\%]')
ax.set_xlabel('Years from last HbA1c measurement')
ax.set_ylim([0,100])
ax.legend()
ax.grid(True)
st.pyplot(fig)   
   
   
