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

#Take a quick look of the Data
housing.head()                  #Top 5 Rows view

#Get some info about the Data
housing.info()

#To get better overview on Object Data
housing["ocean_proximity"].value_counts()

#To get better overview on numerical Data
housing.describe()

#To understand data using Plot (Histogram)
housing.hist(bins=50, figsize=(20,15))
plt.show()

print("\n\n\n\n")

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


#################################Exploring data#################################
