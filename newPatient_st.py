#!/usr/bin/env python
# coding: utf-8
# conda activate streamlit

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import xgboost as xgb
import pickle
import streamlit as st

st.title("T2D Survival")

fixedEffects = {
    'White': 0.1289826971209142,
    'Black': 0.22738555698885063,
    'South_East_Asian': 0.4155943029110619,
    'Other_Asian': 0.42343972831991056,
    'Mixed': 0.12842796602203269,
    'Chinese': 0.08770811171262975,
    'Other': 0.13468703051653572,
    'male': 0.316921375640643,
    "intercept": 6.014093238480706,
    "tyears": 0.035924574056116564
    }

#    - G: 2D array, random effect covariance matrix
#    - sigma2: float, residual variance
#vc = VarCorr(lmeFit)
#print(vc)
#Variance components:
#            Column   Variance Std.Dev.   Corr.
#patid2   (Intercept)  1.390534 1.179209
#         tyears       0.014563 0.120678 -0.31
#Residual              0.838233 0.915551

# cov = vc.σρ[:patid2].ρ[1] * vc.σρ[:patid2].σ[1] * vc.σρ[:patid2].σ[2]
# -0.043312501402308475
G = np.array([[1.390534, -0.043312501402308475],[-0.043312501402308475, 0.014563]])

#println("Residual variance:")
#println(lmeFit.σ^2)
# 0.8382328677496468
sigma2 = 0.8382328677496468

def estimate_random_effects(patientCharacteristics, HbA1c):
    """    
    Args:
    patientCharacteristic (dict): Dictionary where keys are outcome values (float or int), and values are lists or arrays of features (predictors).
    HBA1c (dict)
    
    Returns:
        tuple: (predictions, average_residual)
    """    
    # Extract Y and X
    HbA1c_ = {float(k): v for k, v in HbA1c.items()}
    HbA1c_ = dict(sorted(HbA1c_.items()))
    y_new = np.array(list(HbA1c_.values()), dtype=float)
    Z_new = np.array(list(HbA1c_.keys()), dtype=float)
    
    patientCharacteristics_ =  patientCharacteristics.copy()    
    patientCharacteristics_['tyears'] = Z_new[len(Z_new) - 1]
    
    # Add intercept term to X    
    Z_new = sm.add_constant(Z_new)

    sortedFixedEffects = {key: fixedEffects[key] for key in patientCharacteristics_.keys() if key in fixedEffects.keys()}
    
    # beta_hat: array-like, fixed effect coefficients
    beta_hat = np.array(list(sortedFixedEffects.values()))

    # X_new: 2D array, fixed-effects design matrix
    # X_new = np.array(list(patientCharacteristics_.values()))
    X_new = np.array([values for key, values in patientCharacteristics_.items() if key in fixedEffects.keys()])
    
    # Compute residual from fixed effects
    resid = y_new - (X_new @ beta_hat + fixedEffects["intercept"])

    # Compute marginal variance of y_new
    V = Z_new @ G @ Z_new.T + sigma2 * np.eye(len(y_new))

    V_inv = np.linalg.inv(V)
    # b_hat: estimated random effects (BLUP)
    b_hat = G @ Z_new.T @ V_inv @ resid

    predictions = (X_new @ beta_hat + fixedEffects["intercept"]) + Z_new @ b_hat
    #print("pred", predictions)
    # Compute residuals
    #print("resid", resid)

    #print("b_hat", b_hat)
    residuals = resid - Z_new @ b_hat
    avg_residual = np.mean(np.abs(residuals))
    
    #  from the julia lm code:
    # coeff = fixed_effects[2] .+ random_effects[2, :];
    coeff = fixedEffects['tyears'] + b_hat[1]
    Dict = {"prediction": predictions[len(predictions) - 1], 
                "residuals_mean": avg_residual,
                "coeff": coeff,
                "hba1c_value": y_new[len(y_new) - 1]}
    Dict = Dict | patientCharacteristics_ #{**Dict, **patientCharacteristics_}
    return Dict


def getRisk(X):
    with open('/var/autofs/home/home/mc811/CPRD/trained_booster_.pkl', 'rb') as f:
        bst = pickle.load(f)
    d = xgb.DMatrix(X)
    predicted_risk = bst.predict(d)
    return predicted_risk[0]




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

# Display the result
#st.write("You entered the following HbA1c measurements:")
#st.json(newPatientHBA1c)
    
#newPatientHBA1c = {
#    "2": 6,
#    "3": 1,
#    "4": 8    
#}

blup = estimate_random_effects(newPatientCharact, newPatientHBA1c) 
Xnew = {key: blup[key] for key in XColumns}
#st.write(Xnew)
#import os
#st.write(os.getcwd())
newPatientSummary = np.array(list(Xnew.values()), dtype=float).reshape(1, -1)
risk = getRisk(newPatientSummary)
st.write("Risk: ", risk)
    
    
   
