import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, 2:]  # we only take the first two features.
Y = iris.target




print("X")
print(X)

print("Y")
print(Y)


rng_state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(rng_state)
np.random.shuffle(Y)

length = len(Y)
split_int = int (length * 0.75)

X_train = X[:split_int]

X_test = X[split_int+1:]

Y_train = Y[:split_int]

Y_test = Y[split_int+1:]

h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, Y_train)

pred = logreg.predict(X_test)

print("predict output")
print(pred)


print("actual output")
print(Y_test)


pred_proba = logreg.predict_proba(X_test)
print(pred_proba)

