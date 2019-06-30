# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:55:29 2019

@author: nishi
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# =============================================================================
# Prepare Data
# =============================================================================
grad_data = pd.read_csv("C:\\Users\\nishi\\OneDrive\\Kaggle\\Graduate Admission\\Admission_Predict_Ver1.1.csv")
grad_data.columns


# =============================================================================
# EDA
# =============================================================================
# Scatter Plat
for col in grad_data.columns[1:8]:
    plt.scatter(grad_data[col], grad_data["Chance of Admit "])
    plt.title(col)
    plt.show()

# Correlation
grad_data.drop(columns = ["Serial No."]).corr()


# =============================================================================
# Fit Model
# =============================================================================
import statsmodels.api as sm
import statsmodels.stats
X = grad_data.drop(columns = ["Serial No.", "Chance of Admit "])
X = sm.add_constant(X)
Y = grad_data["Chance of Admit "]

model = sm.OLS(Y,X)
reg_fit = model.fit()

# Summary of simple linear regression model
reg_fit.summary()

# Residual Analysis
residuals = reg_fit.resid
plt.plot(residuals, 'bo')
plt.hist(residuals)
abs(residuals).mean() #average absolute error made by model
(sum([i**2 for i in residuals])/len(residuals))**0.5 #mean square error made by model
residuals.describe()


statsmodels.stats.diagnostic.kstest_normal(residuals)




















