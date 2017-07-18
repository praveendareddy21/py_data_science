import numpy as np

data = np.genfromtxt("data/forest_cover/train.csv", delimiter=',')

print(data)
print(data.shape)