import numpy as np
import matplotlib.pyplot as plt

def fun(x, w, b):
    return (b + x * -w[0] ) / w[1]

def print_statistics(classifier):
    print("Best parameters: w = {}, b = {}".format(classifier.get_w_values()[-1],
                                                   classifier.get_b_values()[-1]))
    print("Solution found in {} iterations".format(classifier.get_iterations()))

def draw_statistics(X, classifier):
    w_values = classifier.get_w_values()
    b_values = classifier.get_b_values()
    w = w_values[-1]
    b = b_values[-1]

    X1 = X[:len(X)//2]
    X2 = X[len(X)//2:]
    classifier_X = np.linspace(min(X[:, 0]), max(X[:, 0]))
    classifier_Y = fun(classifier_X, w, b)

    _, axs = plt.subplots(2, 2)

    axs[0][0].plot(X1[:, 0], X1[:, 1], "ro", label="1")
    axs[0][0].plot(X2[:, 0], X2[:, 1], 'bo', label="-1")
    axs[0][0].plot(classifier_X, classifier_Y, label="Classifier")
    axs[0][0].set_ylabel("x2")
    axs[0][0].set_xlabel("x1")
    axs[0][0].legend()

    axs[0][1].plot(range(len(classifier.get_accuracies())), classifier.get_accuracies(), "r-")
    axs[0][1].set_xlabel("Iteration")
    axs[0][1].set_ylabel("Accuracy")

    axs[1][0].plot(range(len(b_values)), b_values, "b-", label="b")
    axs[1][0].set_xlabel("Iteration")
    axs[1][0].set_ylabel("Value")
    axs[1][0].legend()

    axs[1][1].plot(range(len(w_values)), w_values[:, 0], "r-", label="w1")
    axs[1][1].plot(range(len(w_values)), w_values[:, 1], "g-", label="w2")
    axs[1][1].set_xlabel("Iteration")
    axs[1][1].set_ylabel("Value")
    axs[1][1].legend()

    plt.show()

def count_max_norm(X):
    max_norm = 0
    for x in X:
        norm = np.linalg.norm(x)
        max_norm = max(norm, max_norm)
    return max_norm

def generate_point_set(N=20):
    if N % 2 != 0:
        N += 1

    col1 = np.random.normal(0, 1, N)
    col2a = np.random.random(N // 2) + 0.25
    col2b = -np.random.random(N // 2) - 0.25
    col2 = np.append(col2a, col2b)

    A = np.array([col1, col2])

    angle = np.random.normal()
    x = []

    x.append(A[0] * np.cos(angle) - A[1] * np.sin(angle) + np.random.normal())
    x.append(A[0] * np.sin(angle) + A[1] * np.cos(angle) + np.random.normal())

    x = np.array(x).transpose()
    y = np.append(np.ones(N // 2), -np.ones(N // 2))

    return x, y
