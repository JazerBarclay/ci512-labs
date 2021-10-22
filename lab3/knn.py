from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_predict

# Read the iris data from the comma separated file with the pandas library
data = pd.read_csv("iris.data")

print("----------------- CSV DATA ------------------")

# Print all rows of data
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data)

print("--------------- END OF DATA ----------------")
print()

# Number of neighbors used
clf = KNeighborsClassifier(n_neighbors=3)

# X uses all rows (:) and the first 4 columns as features (0-3)
X = data.values[:, 0:3]

# Y uses all rows (:) and the final column as the label (4) 
Y = data.values[:,4]

# X_train is the training dataset without labels
# X_test is the test set without labels
# Y_train is the label dataset for the X_train set
# Y_test is the label dataset for the X_test set
# Set the test set size to a percentage (.2 = 20%) of the full dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# Add the test dataset to the k-nearest neighbors classifier
clf = clf.fit(X_train, Y_train)

# Stores teh prediction using the test set
Y_prediction = clf.predict(X_test)

# Print the prediction made compared to the actual results ('O' => Correct , 'X' => Incorrect)
print("=========== PREDICTION VS TEST SET ===========")

for i in range(0,len(Y_prediction)):
    print(Y_prediction[i] + " | " + Y_test[i] + " -> " + ("O" if Y_prediction[i] == Y_test[i] else "X"))

print("============== END OF PREDICTION ==============")
print()

# Print the accuracy of the prediction compared to the actual results
print("Train/test accuracy:",accuracy_score(Y_test,Y_prediction))


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