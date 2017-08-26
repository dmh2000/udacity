# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display  # Allows the use of display() for DataFrames
import matplotlib.pyplot as pl
# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
import matplotlib.pyplot as plt

plt.interactive(False)

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
# ======================================================================================
# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis=1)

# Visualize skewed continuous features of original data
vs.distribution(data)

# ======================================================================================
# ======================================================================================
# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data=features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed=True)
# ======================================================================================
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
# ======================================================================================

# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

# Adds
# it's easier for me to see results if I perform each transform individually
workcls = pd.get_dummies(features_log_minmax_transform["workclass"])
edlevel = pd.get_dummies(features_log_minmax_transform["education_level"])
mstatus = pd.get_dummies(features_log_minmax_transform["marital-status"])
occup = pd.get_dummies(features_log_minmax_transform["occupation"])
relat = pd.get_dummies(features_log_minmax_transform["relationship"])
race = pd.get_dummies(features_log_minmax_transform["race"])
sex = pd.get_dummies(features_log_minmax_transform["sex"])
country = pd.get_dummies(features_log_minmax_transform["native-country"])

# Drops
drops = ["workclass", "education_level", "marital-status", "occupation", "relationship", "race", "sex",
         "native-country"]

# Adds
adds = [workcls, edlevel, mstatus, occup, relat, race, sex, country]

# just checking, should be 13
# encoded = list(features_log_minmax_transform.columns)
# print "{} total features before drops.".format(len(encoded))

# drop : this worked using a vector
features_final = features_log_minmax_transform.drop(drops, axis=1)

# just checking, should be 5
# encoded = list(features_final.columns)
# print "{} total features before adds.".format(len(encoded))

# add the resulting dummy features
adds = [workcls, edlevel, mstatus, occup, relat, race, sex, country]
for a in adds:
    features_final = pd.concat([features_final, a], axis=1)

# Encode the 'income_raw' data to numerical values
income = pd.Series(np.where(income_raw == "<=50K", 0, 1))

# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
# print encoded
# ======================================================================================
# ======================================================================================

# Import train_test_split
# updated sklearn to remove deprecation warning
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size=0.2,
                                                    random_state=0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])
# ======================================================================================

'''
TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case
'''
# Calculate accuracy, precision and recall

total = float(income.count())  # 45222
tp = float(np.sum(income))  # 11208 (predicted positives that are positive)
fp = total - tp  # 34014 (predicted positives that are negative)
tn = 0.0  # 0     (predicted negatives that are negative)
fn = 0.0  # 0     (predicted negatives that are positive)

# print total,tp,fp,tn,fn

# accuracy = true positive / total
accuracy = tp / total

# recall = true_positive / (true_positive + false_negatives)
recall = tp / (tp + fn)

# precision = true_positive / (true_positive + false_positives)
precision = tp / (tp + fp)

# print accuracy, recall, precision

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
# HINT: The formula above can be written as (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
beta2 = 0.5 * 0.5
fscore = (1.0 + beta2) * (precision * recall) / ((beta2 * precision) + recall)

# Print the results
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
# ======================================================================================
# ======================================================================================

# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score, fbeta_score


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time()  # Get start time
    X = X_train[0:sample_size]
    y = y_train[0:sample_size]
    learner.fit(X_train[0:sample_size], y_train[0:sample_size])
    end = time()  # Get end time

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[0:300])
    end = time()  # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[0:300], predictions_train)

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[0:300], predictions_train, 0.5)

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)

    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # Return the results
    return results


# ======================================================================================
# ======================================================================================
# Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC

# Initialize the three models
clf_A = DecisionTreeClassifier(random_state=0)
clf_B = LinearSVC(random_state=0)
clf_C = AdaBoostClassifier(random_state=0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100
# HINT: samples_1 is 1% of samples_100
samples_100 = y_train.count()
samples_10 = samples_100 / 10
samples_1 = samples_10 / 10

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)
# ======================================================================================
# ======================================================================================
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Initialize the classifier
clf = DecisionTreeClassifier(random_state=0)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 10, 20, 30, 40, 50],
              'max_depth': [4, 8, 12, 16],
              'min_samples_leaf': [4, 8, 12, 16],
              'max_features': [25, 50, 75, None]
              }

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

print "======================================================================================"
# Get the estimator
best_clf = grid_fit.best_estimator_
print best_clf

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5))
print "======================================================================================"

# ======================================================================================
# ======================================================================================
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

t0 = time()

# Initialize the classifier
clf = AdaBoostClassifier(random_state=0)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'n_estimators': [100, 200, 500, 1000, 5000],
              'learning_rate': [0.5, 1.0, 4.0]
              }

# Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

print "======================================================================================"
# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

t1 = time()

# print "Elapsed Time : " + str(t1 - t0)

# Report the before-and-afterscores
print "Unoptimized model\n------"
# print clf
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5))
print "\nOptimized Model\n------"
# print best_clf
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5))
print "======================================================================================"
# Import a supervised learning model that has 'feature_importances_'
from sklearn.tree import DecisionTreeClassifier


# Train the supervised model on the training set using .fit(X_train, y_train)
model = DecisionTreeClassifier()

model.fit(X_train,y_train)

# TODO: Extract the feature importances using .feature_importances_
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)

print "======================================================================================"
# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))

plt.plot()
