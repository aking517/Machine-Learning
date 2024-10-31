import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from datetime import datetime
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler

# Load the data
counts = pd.read_csv(
    'C:\\Users\\andre\\Downloads\\FremontBridge.csv',
    index_col='Date',
    parse_dates=['Date'],  # Specify the column to parse
    date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%Y %I:%M:%S %p')
)
weather = pd.read_csv('C:\\Users\\andre\\Downloads\\BicycleWeather.csv', index_col='DATE', parse_dates=True)

# Define function to calculate hours of daylight
def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - datetime(2000, 12, 21)).days 
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

# Resample the data to daily frequency and add features
daily = counts.resample('d').sum()
daily['Total'] = daily.sum(axis=1)  

# Calculate daylight hours for each day
daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))

# Add columns for days of the week
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(len(days)):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

# Add holiday feature
cal = USFederalHolidayCalendar()
holidays = cal.holidays('2012', '2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'] = daily['holiday'].fillna(0)

# Set up feature matrix `x` and target `y`
x = daily[['holiday', 'daylight_hrs'] + days]
y = daily['Total']  

# Scale features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Initialize models
linear_reg = LinearRegression()
lasso = Lasso(max_iter=6000, tol= 0.1)
ridge = Ridge(max_iter=6000, tol= 0.1)

# Define range of alpha values
alpha_values = np.logspace(-4, 4, num=100)
param_dist = {'alpha': alpha_values}

# Perform RandomizedSearchCV for Lasso and Ridge models
lasso_cv = RandomizedSearchCV(lasso, param_dist, cv=10, n_iter=100, random_state=0)
ridge_cv = RandomizedSearchCV(ridge, param_dist, cv=10, n_iter=100, random_state=0)

# Fit models and get scores
lin_reg_score = cross_val_score(linear_reg, x, y, cv=10).mean()
lasso_cv.fit(x, y)
ridge_cv.fit(x, y)

lasso_best_score = lasso_cv.best_score_
ridge_best_score = ridge_cv.best_score_
lasso_best_alpha = lasso_cv.best_params_['alpha']  
ridge_best_alpha = ridge_cv.best_params_['alpha']

# Print results
print("The linear regression best cross-validation score is:", lin_reg_score)
print("The lasso best cross-validation score is: ", lasso_best_score)
print("The best lasso alpha is: ", lasso_best_alpha)
print("The ridge best cross-validation score is: ", ridge_best_score)
print("The best ridge alpha is: ", ridge_best_alpha)

# Find and print the best model
best_model = max((lin_reg_score, "Linear Regression"),
                 (lasso_best_score, "Lasso"),
                 (ridge_best_score, "Ridge"))
print("The best model is: ", best_model[1])


