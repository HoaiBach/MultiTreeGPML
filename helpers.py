import numpy as np
from sklearn import metrics
from deap import gp


# Define new functions
def protectedDiv(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# Define new functions
def Sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def test_score(y_test, prediction):
    hamming_loss = metrics.hamming_loss(y_test, prediction)
    f1 = metrics.f1_score(y_test, prediction, average="macro")
    acc = metrics.accuracy_score(y_test, prediction)
    print("Hamming loss", hamming_loss)
    print("F1 Score", f1)
    print("Accuracy ", acc)
    return hamming_loss, f1, acc


def multiCrossover(ind1, ind2):
    assert len(ind1) == len(ind2)
    for idx in range(len(ind1)):
        ind1[idx], ind2[idx] = gp.cxOnePoint(ind1[idx], ind2[idx])
    return ind1, ind2


def multiMutation(ind, toolbox):
    idx = np.random.randint(0, len(ind))
    ind[idx] = toolbox.sub_mutate(ind[idx])[0]
    # for i in range(len(ind)):
    #     ind[i] = toolbox.sub_mutate(ind[i])[0]
    return ind,


