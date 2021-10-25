# Import modules required from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Read the iris data from the comma separated file with the pandas library
data = pd.read_csv("iris.data", skiprows=0)

# Print the data
print(data)

# Set the number of nearest neighbors the model will use to make predictions
# Note: This is usually odd numbers to prevent ties
clf = KNeighborsClassifier(n_neighbors=3)

# X uses all rows (:) and the first 4 columns as attribute data (0-3)
X = data.values[:, 0:3]

# Y uses all rows (:) and the final column as the label data (4) 
Y = data.values[:,4]

# X_train is the training dataset without labels
# X_test is the test set without labels
# Y_train is the label dataset for the X_train set
# Y_test is the label dataset for the X_test set
# Set the test set size to a percentage (.2 = 20%) of the full dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Add the test dataset to the k-nearest neighbors classifier
clf = clf.fit(X_train, Y_train)

# Stores the prediction using the test set
Y_prediction = clf.predict(X_test)

# Print the accuracy of the prediction compared to the actual results
print("Train/test accuracy:",accuracy_score(Y_test,Y_prediction))

# Import shuffle split from scikit-learn
from sklearn.model_selection import ShuffleSplit

# Set shuffle_split params
cv = ShuffleSplit(n_splits=5, test_size=0.2)


from sklearn.model_selection import cross_val_score

# Add the kNN model to the shuffle split params
# Run the test using a (test_size) percentage of the data and run it (n_splits) number of times
scores = cross_val_score(clf, X, Y, cv=cv)

print()

# Print the scores from the tests
print("Cross fold validation accuracy scores:",scores)

# Print the mean value for all tests
print("Cross fold validation accuracy mean:",scores.mean())