# Practice Code Note by github.com/zulfadlizainal (Based on Hands On Machine Learning Book written by Aurélien Géron)

# Import Library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

# Visualize Geographical Data
housing.plot(kind="scatter", x="longitude", y="latitude")

# Adding Alpha to see Scatter Plot based on the Density of samples
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

# Adding
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population", c="median_house_value",
             cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
plt.show()

#######################Finding Correlation (Normal Method)######################

# Standard Correlation Equation
corr_matrix = housing.corr()

# How much each attrivutes correlates with median house values
corr_matrix["median_house_value"].sort_values(ascending=False)


############Finding Correlation (Pandas Scatter Matrix Method)##################

attributes = ["median_house_value", "median_income",
              "total_rooms", "housing_median_age"]

# Scatter selected attributes with each other
scatter_matrix(housing[attributes], figsize=(12, 8))


#######################Zoom in Chart - Based in Interest########################

housing.plot(kind="scatter", x="median_income",
             y="median_house_value", alpha=0.1)


############################Creating new Attributes#############################

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / \
    housing["total_rooms"]
housing["population_per_household"] = housing["population"] / \
    housing["households"]

# Finding correlation with new created Attributes
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

##########################Seperating Predictors & Label#########################

housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

# Since Imputer can only process numerical data, we need to seperate text attributes
housing_prepared = housing.drop("ocean_proximity", axis=1)

################################Linear Regression###############################

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#################################Test your model################################

some_data = housing_prepared.iloc[:5]
some_labels = housing_labels.iloc[:5]

print('\n\nPredictions:', lin_reg.predict(some_data))
print('\nLabels:', list(some_labels))

################################Measure Error###################################

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('\nMSE:\t', lin_mse)
print('\nRMSE:\t', lin_rmse)
