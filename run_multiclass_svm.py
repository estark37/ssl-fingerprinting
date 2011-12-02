import sys
import os
from cvxopt import solvers
import cvxopt
from feature_vectors import (scale, load_feature_vectors, select_test_set,
get_target_sites)
from numpy import *

def SVM_fit(X, Y, k):
    n = len(Y) # number of samples
    d = len(X[0]) # number of features
    # k = number of classes

    # The solver solves:
    # min 1/2 xPx + qx subject to h >= Gx
    # We set x = [theta_1 || theta_2 || ... || theta_k]
    # (each theta has an extra feature for an intercept)
    #
    # P = I, q = [0, ..., 0]
    # h = [-1, ..., -1]
    #
    # G is the matrix that enforces the constraints:
    # for all i:
    # theta^(yi).x^i >= theta^(yj).x^i + 1 for all j != i
    #
    # P is k(d+1) x k(d+1)

    print k*(d+1)

    P = zeros((k*(d+1), k*(d+1)))
    for i in range(0, k*(d+1)):
        P[i, i] = 1.0
    q = zeros((k*(d+1), 1))
    h = zeros((n*(n-1), 1))
    for i in range(n*(n-1)):
        h[i, 0] = -1

    G = zeros((n*(n-1), k*(d+1)))
    next_row = 0
    for i in range(n):
        for j in range(n):
            if j != i:
                xi = X[i]
                xi.append(1)
                yi = Y[i]
                yj = Y[j]
                for m in range(d+1):
                    G[next_row, yi * (d+1) + m] = -xi[m]
                    G[next_row, yj * (d+1) + m] = xi[m]
                next_row = next_row + 1

    sol = solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                     cvxopt.matrix(G),
                     cvxopt.matrix(h))

    thetas = []
    bs = [] # plural of b

    for i in range(k):
        thetas.append(sol['x'][i * (d+1) : i * (d+1) + d])
        bs.append(sol['x'][i * (d+1) + d])
                
    return thetas, bs

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_sites = get_target_sites(input_dir)

    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))

    X, Y = load_feature_vectors(input_dir, num_samples_per_site, labels)
    X, Y, testX, testY = select_test_set(X, Y, (num_samples_per_site / 2) * len(target_sites))

    # X, testX = scale(X, testX)

    X = [[0.0, 0.0], [0.01, 0.01], [1.0, 1.0], [1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [2.1, 2.1]]
    Y=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    labels = {0: 0, 1: 1, 2: 2}

    thetas, bs = SVM_fit(X, Y, len(labels))

main()
