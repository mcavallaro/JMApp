# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from utils import summarise, getRisk
from parameters import baseline
from PIL import Image
from datetime import date
from io import BytesIO

col1, col2 = st.columns([2, 1])

with col1:
    st.title("T2D Survival")
with col2:
    im = Image.open('RedBlockUoLlogo.png')
    st.image(im, caption=None, output_format="PNG")



st.text("This app uses demographic data and repeated HbA1c measurements to predict T2D-associated mortality over time. It visualises HbA1c trajectories alongside the predicted cumulative probability of death, enabling comparison between two patient scenarios.")

col1, col2 = st.columns(2, border=True)
col1.subheader("Scenario A")
col2.subheader("Scenario B")

with col1:
    Sex = st.selectbox("A) Patient Sex:", ["Male", "Female"])
    if Sex == "Male":
        male = 1
    elif Sex == "Female":
        male = 0

    aged = st.number_input("A) Age at T2D diagnosis:", value=65., step=0.1)
    imd = st.number_input("A) IMD (deciles of index of multiple deprivation):", value=6, step=1, min_value=1, max_value=10)
    # indexdate_n = st.number_input("A) indexdate_n (T2D diagnosis date, in days from 1-1-2000):", value=2, step=1, min_value=1, max_value=10*365)
    indexdate_n_d = st.date_input("A) indexdate_n (T2D diagnosis date from 1-1-2000)", value="2000-01-01", min_value="2000-01-01", max_value="2010-01-01", format="YYYY/MM/DD")
    indexdate_n = (indexdate_n_d - date(2000,1,1)).days

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
        st.session_state.num_inputs = 5
    num = st.slider("A) How many HbA1c measurements?", 1, 20, st.session_state.num_inputs)
    #st.session_state.num_inputs 
    newPatientHBA1c = {}

    value = st.number_input(f"A) Percentage of HbA1c at diagnosis:", step=0.1, value=float(6), min_value=float(4), max_value=float(20))
    newPatientHBA1c[0] = value

    col_time, col_value = st.columns(2)

    for i in range(1, num):
        key = col_time.number_input(   f"A) Time {i+1} (years from diagnosis):", step=0.1, value=float(i))
        value = col_value.number_input(f"A) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; HbA1c value {i+1} (%):", step=0.1, value=float(6), min_value=float(4), max_value=float(20))
        if key:  # Only add if key is not empty
            newPatientHBA1c[key] = value

    summary = summarise(newPatientCharact, newPatientHBA1c) 
    Xnew = {key: float(summary[key]) for key in XColumns}
    
    newPatientSummary = np.array(list(Xnew.values()), dtype=float).reshape(1, -1)
    x = pd.DataFrame(newPatientSummary, columns=Xnew.keys())
    risk1 = getRisk(x)
    #st.write("A) Risk: ", risk1)


with col2:
    Sex2 = st.selectbox("B) Patient Sex:", ["Male", "Female"])
    if Sex2 == "Male":
        male2 = 1
    elif Sex2 == "Female":
        male2 = 0

    aged2 = st.number_input("B) Age at T2D diagnosis:", value=65., step=0.1)
    imd2 = st.number_input("B) IMD (deciles of index of multiple deprivation):", value=6, step=1, min_value=1, max_value=10)
    #indexdate_n2 = st.number_input("B) indexdate_n (T2DM diagnosis date, in days from 1-1-2000):", value=2, step=1, min_value=1, max_value=10*365)
    indexdate_n2_d = st.date_input("B) indexdate_n (T2D diagnosis date from 1-1-2000)", value="2000-01-01", min_value="2000-01-01", max_value="2010-01-01", format="YYYY/MM/DD")
    indexdate_n2 = (indexdate_n2_d - date(2000,1,1)).days
    
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
        st.session_state.num_inputs = 5
    num2 = st.slider("B) How many HbA1c measurements?", 1, 20, st.session_state.num_inputs)
    #st.session_state.num_inputs 
    newPatientHBA1c2 = {}

    value2 = st.number_input(f"B) Percentage of HbA1c at diagnosis:", step=0.1, value=float(6), min_value=float(4), max_value=float(20))
    newPatientHBA1c2[0] = value2
    
    col_time, col_value = st.columns(2)
    
    for i in range(1, num2):
        key2 = col_time.number_input(f"B) Time {i+1} (years from diagnosis):", step=0.1, value=float(i))
        value2 = col_value.number_input(f"B) &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; HbA1c value {i+1} (%):", step=0.1, value=float(6), min_value=float(4), max_value=float(20))
        if key2:  # Only add if key is not empty
            newPatientHBA1c2[key2] = value2

    summary2 = summarise(newPatientCharact2, newPatientHBA1c2) 
    Xnew2 = {key: summary2[key] for key in XColumns}

    newPatientSummary2 = np.array(list(Xnew2.values()), dtype=float).reshape(1, -1)
    x2 = pd.DataFrame(newPatientSummary2, columns=Xnew2.keys())
    risk2 = getRisk(x2)
    #st.write("B) Risk: ", risk2)


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


#for y in [20, 40, 60, 80]:
#    ax_right.hlines(y=y,
#                    xmin=time_from_Dx1[-1] + newPatientCharact['aged'],
#                    xmax=time_from_Dx1[-1] + newPatientCharact['aged'] + baseline['time'][-1], color="gray", linestyle="--")
#for y in [4, 6, 8, 10, 12, 14, 16, 18]:
#    ax.hlines(y=y,
#              xmin=newPatientCharact['aged'],
#              xmax=time_from_Dx1[-1] + newPatientCharact['aged'], color="gray", linestyle="--")

colour_A = "#1F77B4"
colour_B = "#E15759"

ax.plot(np.array(time_from_Dx1) + newPatientCharact['aged'], value1, linestyle='--', marker="o", lw=1, ms=3, color=colour_A, label="Scenario A\'s HbA1c 2trajectory")
ax.plot(np.array(time_from_Dx2) + newPatientCharact2['aged'], value2, linestyle='--', marker="o", lw=1, ms=3, color=colour_B, label="Scenario B\'s HbA1c trajectory")

ax.plot(summary['predictions'][0], summary['predictions'][1], linestyle='-', lw=2, color=colour_A, alpha=0.7)
ax.plot(summary2['predictions'][0], summary2['predictions'][1], linestyle='-', lw=2, color=colour_B, alpha=0.7)

#
ax.axvline(time_from_Dx1[-1] + newPatientCharact['aged'], linestyle='--', alpha=0.5, color=colour_A)
ax.axvline(time_from_Dx2[-1] + newPatientCharact2['aged'], linestyle='--', alpha=0.5, color=colour_B)
#2
ax_right.plot(baseline['time'] + time_from_Dx1[-1] + newPatientCharact['aged'], prob1, lw=2, color=colour_A, label="Scenario A\'s prob. of death")
ax_right.plot(baseline['time'] + time_from_Dx2[-1] + newPatientCharact2['aged'], prob2, lw=2, color=colour_B, label="Scenario B\'s prob. of death")
ax_right.set_ylabel('Patient probability of death [%]')
ax.set_xlabel('Age')
ax.set_ylabel('HbA1c [%]')
ax_right.set_ylim([0,100])
ax.set_ylim([3.8, 20])
ax.legend(loc='upper left')
ax_right.legend(loc='lower right')

ax.grid(True) #, axis='x')
ax_right.grid(True, axis='y')



st.pyplot(fig)



# Convert figure to PNG in memory
buf = BytesIO()
fig.savefig(
    buf,
    format="png",
    dpi=300,
    bbox_inches="tight"
)
buf.seek(0)

# Download button
st.download_button(
    label="Download figure",
    data=buf,
    width="stretch",
    file_name="figure.png",
    mime="image/png"
)




#fig, ax = plt.subplots()
#ax.plot(baseline['time'], prob1, lw=2, color=colour_A, label='Patient A')
#ax.plot(baseline['time'], prob2, lw=2, color=colour_B, label='Patient B')
#ax.set_ylabel('Patient probability of death [%]')
#ax.set_xlabel('Years from last HbA1c measurement')
#ax.set_ylim([0,100])
#ax.legend(loc='lower right')
#ax.grid(True)
#st.pyplot(fig)   
   
   
