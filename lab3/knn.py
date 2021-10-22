from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set randomness based on seed 95
np.random.seed(95)

# Read the iris data from the comma separated file with the pandas library
data = pd.read_csv("iris.data", skiprows=0)

print("----------------- CSV DATA ------------------")

# Print all rows of data
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data)

print("--------------- END OF DATA ----------------")
print()

# Number of neighbors used to classify a test datapoint
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

# Feature scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Add the test dataset to the k-nearest neighbors classifier
clf = clf.fit(X_train, Y_train)

# Stores the prediction using the test set
Y_prediction = clf.predict(X_test)

# Print the prediction made compared to the actual results ('O' => Correct , 'X' => Incorrect)
print("=========== PREDICTION VS TEST SET ===========")

for i in range(0,len(Y_prediction)):
    print(Y_prediction[i] + " | " + Y_test[i] + " -> " + ("O" if Y_prediction[i] == Y_test[i] else "X"))

print("============== END OF PREDICTION ==============")
print()

# Print the accuracy of the prediction compared to the actual results
print("Train/test accuracy:",accuracy_score(Y_test,Y_prediction))

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


precision = precision_score(Y_test, Y_prediction, average="macro")
print("Macro Precission Score: ", end='')
print(precision)

# Print confusion matrix for test data
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, Y_prediction))
print(classification_report(Y_test, Y_prediction))

# ------------ Drawing windows with data :D ------------ #

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# Create color maps
cmap_bold = ['darkorange', 'c', 'darkblue']

# step size in the mesh
h = .02  

# Draw graphs for sepal and petal comparisons for all data
for vals in [[0,3], [1,4]]:

    X = data.values[:, vals[0]:vals[1]]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a plot
    plt.figure(figsize=(8, 7))

    # Plot also the training points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y,
                    palette=cmap_bold, alpha=1.0, edgecolor="black")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification of length vs width")
    plt.xlabel(("sepal" if vals[0] == 0 else "petal") + " length (cm)")
    plt.ylabel(("sepal" if vals[0] == 0 else "petal") + " width (cm)")

# Display all stats in their own windows
plt.show()