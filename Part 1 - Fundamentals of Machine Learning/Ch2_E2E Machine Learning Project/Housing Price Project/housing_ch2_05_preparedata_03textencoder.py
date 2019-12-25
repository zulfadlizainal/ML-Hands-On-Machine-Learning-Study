# Practice Code Note by github.com/zulfadlizainal (Based on Hands On Machine Learning Book written by Aurélien Géron)

# Import Library
import os
import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import hashlib
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
#from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

###################################Check Data###################################

# Define Importing Function - Import Data from .py Directory

HOUSING_PATH = ""


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Calling Function - To import Data
housing = load_housing_data()

#########################Creating a Training + Test Set#########################
# Using Scikit-Learn (Categorization/Strata) (Method 5)

# Method 04 is the best, but if your data is small -> Sample Bias could happen
# Method 05 is good when samples is small and we want to select samples based on categorized main features.

# Categorize Samples Based on Important Features
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# Split the data using Strata
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    train_set_05 = housing.loc[train_index]
    test_set_05 = housing.loc[test_index]

# Check Distribution of Categorized/Stratified Samples
housing["income_cat"].value_counts() / len(housing)

del train_index, test_index


##########################Seperating Predictors & Label#########################

housing = train_set_05.drop("median_house_value", axis=1)
housing_labels = train_set_05["median_house_value"].copy()


#################Cleaning Data - Using Scikit Learn Imputer#####################

# Define Imputer strategy
imputer = SimpleImputer(strategy="median")

# Since Imputer can only process numerical data, we need to seperate text attributes
housing_num = housing.drop("ocean_proximity", axis=1)

# Fit imputer instance to training data
option4 = SimpleImputer.fit(imputer, housing_num)

# Now can use the trained Imputer to Transform training set, by replacing missing valued by learned medians
option4 = SimpleImputer.transform(imputer, housing_num)
# Output is just numpy array

# Change back to pandas dataframe
housing_tr = pd.DataFrame(option4, columns=housing_num.columns)


###############Switch Text to Numerical (Label Encoder Method)##################

# Define encoder function
encoder = LabelEncoder()

# Transform column to numerical category
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

# Check category definition
print(encoder.classes_)

#########Switch Integer Cat Value to One Hot (One Hot Encoder Method)###########

# Define encoder function
encoder = OneHotEncoder()

# Transform integer category to 1hot
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
#Output in Sparse array

# Convert to Numpy Array
housing_cat_1hot = housing_cat_1hot.toarray()
