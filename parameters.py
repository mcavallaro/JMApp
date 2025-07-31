# coding: utf-8

import numpy as np

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

# random effect covariance matrix
G = np.array([[1.390534, -0.043312501402308475],[-0.043312501402308475, 0.014563]])

# Residual variance
sigma2 = 0.8382328677496468

baseline = np.load("baseline.npz")

