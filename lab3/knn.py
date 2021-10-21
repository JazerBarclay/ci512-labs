from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_predict

data = pd.read_csv("iris.data")
print(data)

clf = KNeighborsClassifier(n_neighbors=3)

X = data.values[:, 0:3]
Y = data.values[:,4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
clf = clf.fit(X_train, Y_train)
Y_prediction = clf.predict(X_test)
print("Train/test accuracy:",accuracy_score(Y_test,Y_prediction))

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=5, test_size=0.2)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X, Y, cv=cv)
print()
print("Cross fold validation accuracy scores:",scores)
print("Cross fold validation accuracy mean:",scores.mean())