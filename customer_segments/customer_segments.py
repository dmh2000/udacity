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

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2, random_state=42)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.fit_transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

# Apply your clustering algorithm of choice to the reduced data
clusterer = KMeans(n_clusters=2, random_state=2)

clusterer.fit(reduced_data)

# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer.cluster_centers_
print(centers)
# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data,preds)

print(score)

# n_clusters = 2 0.447157742293
# n_clusters = 3 0.364874035612
# n_clusters = 4 0.331150954285
# n_clusters = 5 0.350794538773



# Apply your clustering algorithm of choice to the reduced data
clusterer = GaussianMixture(n_components=2,random_state=42)

clusterer.fit(reduced_data)

# Predict the cluster for each data point
preds = clusterer.predict(reduced_data)

# Find the cluster centers
centers = clusterer.means_

# Predict the cluster for each transformed sample data point
sample_preds = clusterer.predict(pca_samples)

# Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data,preds)

print(score)

print("----------------------------")
log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)
print(centers)
print(log_centers)
print(true_centers)
