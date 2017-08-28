# Import libraries necessary for this project
import numpy as np
import pandas as pd
# from IPython.display import display # Allows the use of display() for DataFrames
import IPython.display as dp

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
import matplotlib

# matplotlib inline

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis=1, inplace=True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

dp.display(data.describe())

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [0, 1, 2]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
print "Chosen samples of wholesale customers dataset:"
dp.display(samples)

# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

# which feature to drop
drop_feature = "Delicatessen"

# labels
y = data[drop_feature]

# features minus dropped value
new_data = data.drop(drop_feature, axis=1)

# Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data, y, test_size=.20, random_state=42)

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(max_depth=10)

cv = cross_val_score(regressor, new_data, y, cv=10)
print(cv)

# fit the model
regressor.fit(X_train, y_train)

print(regressor.feature_importances_)

# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)

print(score)

log_data = np.log(data)
log_samples = np.log(samples)

# For each feature find the data points with extreme high or low values
all_outliers = []
index = 0
for feature in log_data.keys():
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)

    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    # display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

    # accumulate outlier indices
    ol = log_data.index[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    [all_outliers.append(i) for i in ol]


# OPTIONAL: Select the indices for data points you wish to remove
duplicate_outliers = []
all_outliers.sort()
for i in range(1,len(all_outliers)):
    if all_outliers[i-1] == all_outliers[i]:
        duplicate_outliers.append(all_outliers[i])
print(duplicate_outliers)

# get unique outliers
unique_outliers = np.unique(all_outliers)

print(unique_outliers)

# threshold the outliers
outliers = unique_outliers


print(len(outliers))
print(440 - len(outliers))

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)

from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(random_state=42)
pca.fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)

print 0.4993
component 2 variance  = 0.2259
component 3 variance  = 0.1049
component 4 variance  = 0.0978

print 0.4993 + 0.2259 + 0.1049 + 0.0978