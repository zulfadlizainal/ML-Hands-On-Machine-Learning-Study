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
###Using RAND Method (Method 1)

#Define Split Training + Test Set Function (Based on Test Ratio)
#This code will split it Randomly
#Ideal Rule: 80% Training Samples, 20% Test Sample
#Problem: If you run this program again, it will generate a different test set.
#Because of this, over time, your ML algorithm will get to see your overall data (We need to avoid this)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#Split Training + Test Set
train_set_01, test_set_01 = split_train_test(housing, 0.2)
print(len(train_set_01), "Train Set +", len(test_set_01), "Test Set")


#########################Creating a Training + Test Set#########################
###Using HASH ID Method (Method 2)

#This method using Hash unique identifier for each Rows
#By doing this, the test set will remain consistent across multiple runs, even if you refresh the dataset.
#Problem: The housing data set does not have identifier column, so need to create id column


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

#Add ID column
housing_with_id = housing.reset_index() # adds an `index` column
train_set_02, test_set_02 = split_train_test_by_id(housing_with_id, 0.2, "index")

###Using HASH ID Method (Method 3)

#Using Current Data to create ID
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set_03, test_set_03 = split_train_test_by_id(housing_with_id, 0.2, "id")


#########################Creating a Training + Test Set#########################
###Using Scikit-Learn (BEST -RECOMMENDED!) (Method 4)

#One liner..
#Benefit: Can input multiple data set, can input random_state (So the training set will not change)

train_set_04, test_set_04 = train_test_split(housing, test_size=0.2, random_state=42)


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
