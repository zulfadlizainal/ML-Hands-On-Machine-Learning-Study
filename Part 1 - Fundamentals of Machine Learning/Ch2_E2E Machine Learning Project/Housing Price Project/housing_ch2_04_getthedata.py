#Practice Code Note by github.com/zulfadlizainal (Based on Hands On Machine Learning Book written by Aurélien Géron)

#Import Library
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

###################################Check Data###################################

#Define Importing Function - Import Data from .py Directory

HOUSING_PATH = ""

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#Calling Function - To import Data
housing = load_housing_data()

#########################Creating a Training + Test Set#########################
###Using Scikit-Learn (Categorization/Strata) (Method 5)

#Method 04 is the best, but if your data is small -> Sample Bias could happen
#Method 05 is good when samples is small and we want to select samples based on categorized main features.

#Categorize Samples Based on Important Features
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#Split the data using Strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    train_set_05 = housing.loc[train_index]
    test_set_05 = housing.loc[test_index]

#Check Distribution of Categorized/Stratified Samples
housing["income_cat"].value_counts() / len(housing)

del train_index, test_index

#################################Exploring data#################################

#Create a copy of Training Database
housing = train_set_05.copy()

#Visualize Geographical Data
housing.plot(kind="scatter", x="longitude", y="latitude")

#Adding Alpha to see Scatter Plot based on the Density of samples
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

#Adding
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
            s=housing["population"]/100, label="population", c="median_house_value",
            cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
plt.show()

###############################Finding Correlation##############################

#Standard Correlation Equation
corr_matrix = housing.corr()

#Howmuch each attrivutes correlates with median house values
corr_matrix["median_house_value"].sort_values(ascending=False)
