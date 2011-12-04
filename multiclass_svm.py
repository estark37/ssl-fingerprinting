import json
from cvxopt import solvers
import cvxopt
from numpy import *
import os

def SVM_classify(x, thetas, bs):
    ys = []
    for (i, (theta, b)) in enumerate(zip(thetas, bs)):
        ys.append((dot(matrix(x), theta) + b, i))

    return (max(ys))[1]

def SVM_fit(xfile, Y, k, d, c = 1.0):
    n = len(Y) # number of samples

    print "n: %d, d: %d, k: %d"%(n,d,k)

    # k = number of classes

    # The solver solves:
    # min 1/2 xPx + qx subject to h >= Gx
    # We set x = [theta_1 || theta_2 || ... || theta_k ||
    #     slack_1 || ... || slack_n]
    # (each theta has an extra feature for an intercept)
    #
    # The top block of P is I. The bottom block is 0's.
    # q = [0, ..., 0, c, ..., c]
    # h = [-1, ..., -1]
    #
    # G is the matrix that enforces the constraints:
    # for all i:
    # theta^(yi).x^i >= theta^(yj).x^i + 1 - slack_i for all j != i
    #
    # P is k(d+1)+n x k(d+1)+n

    P = zeros((k*(d+1)+n, k*(d+1)+n))
    for i in range(0, k*(d+1)):
        P[i, i] = 1.0

    print "Built P"

    q = zeros((k*(d+1) + n, 1))
    for i in range(n):
        q[k*(d+1) + i] = c

    print "Built q"

    h = zeros((n*(n-1) + n, 1))
    for i in range(n*(n-1)):
        h[i, 0] = -1.0

    print "Built h"

    f = open(xfile)

    G = zeros((n*(n-1) + n, k*(d+1) + n))
    next_row = 0
    i = 0
    for l in f:
        for j in range(n):
            if j != i:
                xi = json.loads(l)
                xi.append(1.0)
                yi = Y[i]
                yj = Y[j]
                for m in range(d+1):
                    G[next_row, yi * (d+1) + m] = -xi[m]
                    G[next_row, yj * (d+1) + m] = xi[m]
                
                G[next_row, k*(d+1) + i] = -1.0
                next_row = next_row + 1
        i = i + 1

    # The final rows of G enforces that slacks are nonnegative
    for i in range(n):
        G[next_row, k*(d+1) + i] = -1.0
        next_row = next_row + 1

    print "Built G"

    sol = solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                     cvxopt.matrix(G),
                     cvxopt.matrix(h))

    thetas = []
    bs = [] # plural of b

    for i in range(k):
        thetas.append(sol['x'][i * (d+1) : i * (d+1) + d])
        bs.append(sol['x'][i * (d+1) + d])
                
    return thetas, bs, sol['x'][-n:]
