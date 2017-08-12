# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display  # Allows the use of display() for DataFrames
import matplotlib.pyplot as pl
# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
import matplotlib

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))

print data.head(n=1).size
# age,workclass,education_level,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income
# 39,State-gov,Bachelors,13.0,Never-married,0,Adm-clerical,Not-in-family,White,Male,2174.0,0.0,0,40.0,United-States,<=50K


# Total number of records
# note : data.size == rows * columns
print data.shape
n_records = data.shape[0]

# Number of records where individual's income is more than $50,000
income = data["income"]
n_greater_50k = income[income == ">50K"].size

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = income[income == "<=50K"].size

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = float(n_greater_50k) / (n_records)

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

# ======================================================================================
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis=1)

# Visualize skewed continuous features of original data
vs.distribution(data)

# ======================================================================================
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed=True)
# ======================================================================================
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()  # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n=5))
# ======================================================================================

# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

# Adds
workcls = pd.get_dummies(features_log_minmax_transform["workclass"])
edlevel = pd.get_dummies(features_log_minmax_transform["education_level"])
mstatus = pd.get_dummies(features_log_minmax_transform["marital-status"])
occup   = pd.get_dummies(features_log_minmax_transform["occupation"])
relat   = pd.get_dummies(features_log_minmax_transform["relationship"])
race    = pd.get_dummies(features_log_minmax_transform["race"])
sex     = pd.get_dummies(features_log_minmax_transform["sex"])
country = pd.get_dummies(features_log_minmax_transform["native-country"])

# Drops
drops = ["workclass", "education_level", "marital-status", "occupation", "relationship", "race", "sex","native-country"]

# adds
adds = [workcls,edlevel,mstatus,occup,relat,race,sex,country]

encoded = list(features_log_minmax_transform.columns)
print "{} total features before drops.".format(len(encoded))

# drop
features_final = features_log_minmax_transform.drop(drops, axis=1)
encoded = list(features_final.columns)
print "{} total features after drops.".format(len(encoded))

# add
for a in adds:
    features_final = features_final.add(a, axis=1)

encoded = list(features_final.columns)
print "{} total features after adds.".format(len(encoded))

# Encode the 'income_raw' data to numerical values
income = pd.Series(np.where(income_raw == "<=50K", 0, 1))

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
# print encoded

# ======================================================================================
'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# Calculate accuracy, precision and recall
#from sklearn.metrics import accuracy_score,recall_score,precision_score,fbeta_score

total    = float(income.count()) # 45222
tp       = float(np.sum(income)) # 11208 (predicted positives that are positive)
fp       = total - tp            # 34014 (predicted positives that are negative)
tn       = 0.0                   # 0     (predicted negatives that are negative)
fn       = 0.0                   # 0     (predicted negatives that are positive)

# print total,tp,fp,tn,fn

# accuracy = true positive / total
accuracy  = tp / total

# recall = true_positive / (true_positive + false_negatives)
recall    = tp / (tp + fn)

# precision = true_positive / (true_positive + false_positives)
precision = tp / (tp + fp)

# print accuracy, recall, precision

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
# HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
beta2 = 0.5 * 0.5
fscore = (1.0 + beta2) * (precision * recall) / ((beta2 * precision) + recall)

# Print the results
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)


pl.show()
