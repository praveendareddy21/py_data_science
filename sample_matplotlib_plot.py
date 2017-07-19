from matplotlib import pyplot
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


test_losses = []
test_accuracies = []
indep_test_axis = []
batch_size = 300

for i in range(batch_size):
    indep_test_axis.append(i)
    test_losses.append(3.5 - 1.6 * sigmoid(i / 10))
    test_accuracies.append(0.5 + 0.4 * sigmoid(i / 10))

indep_test_axis = np.array(indep_test_axis)
test_losses = np.array(test_losses)
test_accuracies = np.array(test_accuracies)

pyplot.plot(indep_test_axis, test_losses)



pyplot.title("sample plot")
pyplot.legend(loc='upper right', shadow=True)
pyplot.ylabel("loss")
pyplot.xlabel("x index")
pyplot.show()

