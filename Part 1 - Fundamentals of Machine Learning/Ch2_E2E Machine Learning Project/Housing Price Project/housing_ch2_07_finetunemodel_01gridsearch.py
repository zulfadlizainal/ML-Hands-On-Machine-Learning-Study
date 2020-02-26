# Practice Code Note by github.com/zulfadlizainal (Based on Hands On Machine Learning Book written by Aurélien Géron)

# Import Library
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

###################################Check Data###################################

# Define Importing Function - Import Data from .py Directory

HOUSING_PATH = ""


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Calling Function - To import Data
housing = load_housing_data()

#########################Creating a Training + Test Set#########################

# Using Scikit-Learn (BEST -RECOMMENDED!) (Method 4)

# One liner..
# Benefit: Can input multiple data set, can input random_state (So the training set will not change)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

################################Exploring data#################################

# Create a copy of Training Database
housing = train_set.copy()

##########################Seperating Predictors & Label#########################

housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# Since Imputer can only process numerical data, we need to seperate text attributes
housing_prepared = housing.drop("ocean_proximity", axis=1)

#############################Random Forest Regressor############################

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)

################################Measure Error###################################

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print('\nMSE:\t', forest_mse)
print('\nRMSE:\t', forest_rmse)

#############################Cross Validation###################################

#CV=10 means split the train set to 10 randomly and retrain the model
scores = cross_val_score(forest_reg, housing_prepared,
                         housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print('Scores: ', scores)
    print('Mean: ', scores.mean())
    print('Standard Deviation: ', scores.std())


display_scores(forest_rmse_scores)

##############Test Hyperparameters (Grid Search Method)#########################

param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, {
    'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, ]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# To get best parameter
print(grid_search.best_params_)

# To get best estimator
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
