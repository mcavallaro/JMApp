# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
from parameters import fixedEffects, G, sigma2

def summarise(patientCharacteristics, HbA1c):
    """    
    Args:
    patientCharacteristic (dict): Dictionary where keys are outcome values (float or int), and values are lists or arrays of features (predictors).
    HBA1c (dict)
    
    Returns:
        Dict (dict): Contains all patient infomation, including a summary of their trajectory
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

    sortedFixedEffects = {key: fixedEffects[key] for key in fixedEffects.keys() if key in patientCharacteristics_.keys()}
    #fixed_effect_characteristics = ["aged", "White", 'Black', 'South_East_Asian', 'Other_Asian', 'Mixed', 'Chinese', 'Other', "male", "tyears"]
    #sortedFixedEffects = {key: fixedEffects[key] for key in fixed_effect_characteristics if key in fixedEffects.keys()}


    # hba1c_value ~ tyears + male + aged + White + Black + South_East_Asian + Other_Asian + Mixed + Chinese + Other + (tyears|patid2)),
   
    # beta_hat: array-like, fixed effect coefficients
    beta_hat = np.array(list(sortedFixedEffects.values()))

    # X_new: 2D array, fixed-effects design matrix
    # X_new = np.array(list(patientCharacteristics_.values()))
    X_new = np.array([values for key, values in patientCharacteristics_.items() if key in fixedEffects.keys()])
    
    # Compute residual from fixed effects
    resid = y_new - (X_new @ beta_hat + fixedEffects["intercept"])

    # Compute marginal variance of y_new and random effects
    V = Z_new @ G @ Z_new.T + sigma2 * np.eye(len(y_new))
    V_inv = np.linalg.inv(V)
    b_hat = G @ Z_new.T @ V_inv @ resid   

    predictions = (X_new @ beta_hat + fixedEffects["intercept"]) + Z_new @ b_hat

    residuals = resid - Z_new @ b_hat # y_new - predictions
    avg_residual = np.mean(np.abs(residuals))
    
    #  from the julia lm code:
    # coeff = fixed_effects[2] .+ random_effects[2, :];
    coeff = fixedEffects['tyears'] + b_hat[1]
    
    
    #alternative:
    #V = Z_new @ G @ Z_new.T + sigma2 * np.eye(len(y_new))
        #Vinvy = np.linalg.solve(V, y_new)
        #X =np.array(np.repeat(X_new, ))
        #VinvX = np.linalg.solve(V, X)
        #cov_beta = np.linalg.inv(X_new.T @ VinvX)          # Var(beta_hat)
        #beta_hat = cov_beta @ (X_new.T @ Vinvy)                # BLUE
        #b_hat_2 = G @ Z_new.T @ np.linalg.solve(V, y_new - X_new @ beta_hat) # BLUP
          
    
    Dict = {"prediction": predictions[len(predictions) - 1], 
                "residuals_mean": avg_residual,
                "coeff": coeff,
                "b_hat": b_hat,
                "hba1c_value": y_new[len(y_new) - 1],
                "predictions": (np.array(list(HbA1c_.keys()), dtype=float) + patientCharacteristics_['aged'], predictions)
                }
    Dict = Dict | patientCharacteristics_ 
    return Dict


 

def getRisk(X):
    with open('trained_booster_3.pkl', 'rb') as f:
#    with open('/mount/src/jmapp/trained_booster_3.pkl', 'rb') as f:
        bst = pickle.load(f)
    d = xgb.DMatrix(X)
    predicted_risk = bst.predict(d)
    return predicted_risk[0]

