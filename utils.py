#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import parameters as pr

def summarise(patientCharacteristics, HbA1c):
    """    
    Args:
    patientCharacteristic (dict): Dictionary where keys are outcome values (float or int), and values are lists or arrays of features (predictors).
    HBA1c (dict)
    
    Returns:
        Dict (dict): Contains all patient infomation, including summary of their trajectory
    """    
    # Extract Y and X
    HbA1c_ = {float(k): v for k, v in HbA1c.items()}
    HbA1c_ = dict(sorted(HbA1c_.items()))
    y_new = np.array(list(HbA1c_.values()), dtype=float)
    Z_new = np.array(list(HbA1c_.keys()), dtype=float)
    
    patientCharacteristics_ =  patientCharacteristics.copy()    
    patientCharacteristics_['tyears'] = Z_new[len(Z_new) - 1]
    
    # Add intercept term to X    
    I = np.ones(len(Z_new))
    Z_new = np.vstack([I, Z_new]).transpose()

    sortedFixedEffects = {key: pr.fixedEffects[key] for key in patientCharacteristics_.keys() if key in pr.fixedEffects.keys()}
    
    # beta_hat: array-like, fixed effect coefficients
    beta_hat = np.array(list(sortedFixedEffects.values()))

    # X_new: 2D array, fixed-effects design matrix
    # X_new = np.array(list(patientCharacteristics_.values()))
    X_new = np.array([values for key, values in patientCharacteristics_.items() if key in pr.fixedEffects.keys()])
    
    # Compute residual from fixed effects
    resid = y_new - (X_new @ beta_hat + pr.fixedEffects["intercept"])

    # Compute marginal variance of y_new and random effects
    V = Z_new @ pr.G @ Z_new.T + pr.sigma2 * np.eye(len(y_new))
    V_inv = np.linalg.inv(V)
    b_hat = pr.G @ Z_new.T @ V_inv @ resid

    predictions = (X_new @ beta_hat + pr.fixedEffects["intercept"]) + Z_new @ b_hat

    residuals = resid - Z_new @ b_hat
    avg_residual = np.mean(np.abs(residuals))
    
    #  from the julia lm code:
    # coeff = fixed_effects[2] .+ random_effects[2, :];
    coeff = pr.fixedEffects['tyears'] + b_hat[1]
    Dict = {"prediction": predictions[len(predictions) - 1], 
                "residuals_mean": avg_residual,
                "coeff": coeff,
                "hba1c_value": y_new[len(y_new) - 1]}
    Dict = Dict | patientCharacteristics_ 
    return Dict


def getRisk(X):
    with open('/mount/src/jmapp/trained_booster_.pkl', 'rb') as f:
        bst = pickle.load(f)
    d = xgb.DMatrix(X)
    predicted_risk = bst.predict(d)
    return predicted_risk[0]

