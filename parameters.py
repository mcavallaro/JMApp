# coding: utf-8

import numpy as np

fixedEffects = {
    'aged': 0.004745540728017758,
    'White': 0.08638973990196011,
    'Black': 0.22958769339743168,
    'South_East_Asian': 0.42478216574391114,
    'Other_Asian': 0.4302495022117645,
    'Mixed': 0.14378449324373335,
    'Chinese': 0.07299347582455766,
    'Other': 0.1392801498731118,
    'male': 0.3135815459496682,
    "intercept": 5.778220602230432,
    "tyears": 0.03518103796475563
    }
    
# random effect covariance matrix
G = np.array([[1.390534, -0.043312501402308475],[-0.043312501402308475, 0.014563]])

# Residual variance
sigma2 = 0.8382328677496468

baseline = np.load("baseline.npz")

