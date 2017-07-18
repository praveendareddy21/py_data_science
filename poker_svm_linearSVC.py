import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC


X_train, y_train = load_svmlight_file('data/poker.bz2')
from sklearn.preprocessing import OneHotEncoder
hot_encoding = OneHotEncoder(sparse=True)
X_train = hot_encoding.fit_transform(X_train.toarray())

print(X_train)
print(y_train)

hypothesis = LinearSVC(dual=False)

scores = cross_val_score(hypothesis, X_train, y_train, cv=3,scoring='accuracy', n_jobs=-1)

print "LinearSVC -> cross validation accuracy: mean = %0.3f std = %0.3f" % (np.mean(scores), np.std(scores))