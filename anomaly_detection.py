import sys
from cvxopt import solvers
import cvxopt
from numpy import *
import cmath

def rbf(Xi, Xj):
    beta = 0.005
    norm = sum(map(lambda (xi, xj): (xi-xj)**2, zip(Xi, Xj)))
    return cmath.exp(-1*beta * norm).real

def linK(Xi, Xj):
    return sum(map(lambda (xi, xj): xi*xj, zip(Xi, Xj)))

def test():
    X = [[100.0, 100.0], [102.25, 102.25], [101.5, 101.5]]
    rho, alphas = AnomDet_fit(X, 0.00005, linK)

    print rho

    for x in X:
        print AnomDet_classify(x, alphas, rho, X, linK)
    print "Should be an anomaly:"
    print AnomDet_classify([90.0, 90.1], alphas, rho, X, linK)

# Returns 1 for an anomaly, 0 otherwise
def AnomDet_classify(x, alphas, rho, X, K = linK):
    t = sum(map(lambda (i, a): a*K(X[i], x), enumerate(alphas)))
    if t >= rho:
        return 0
    else:
        return 1

def AnomDet_fit(X, v, K = linK):
    n = len(X)
    d = len(X[0])

    # x is [alpha1, ..., alphan, alpha1, ..., alphan]
    # P is the kernel matrix appended with zeros
    # G enforces the constraints that 
    #    0 <= alphai <= 1/vn
    # A enforces the constraint that alpha1+...+alphan = 1

    P = zeros((2*n, 2*n))
    for i in range(n):
        for j in range(n):
            P[i, j] = K(X[i], X[j])

    q = zeros((2*n, 1))
    
    G = zeros((3*n, 2*n))
    h = zeros((3*n, 1))

    A = zeros((1, 2*n))
    b = zeros((1, 1))
    b[0, 0] = 1.0

    for i in range(n):
        G[i, i] = 1.0
        G[n+i, n+i] = -1.0
        G[2*n+i, i] = 1.0
        G[2*n+i, n+i] = -1.0
        h[i, 0] = 1.0/(v*n)
        A[0, i] = 1.0

    solvers.options['show_progress'] = False
    sol = solvers.qp(cvxopt.matrix(P), cvxopt.matrix(q),
                     cvxopt.matrix(G), cvxopt.matrix(h),
                     cvxopt.matrix(A), cvxopt.matrix(b))

    rhos = []
    alphas = []
    for j in range(n):
        alphas.append(sol['x'][j])
        sum = 0.0
        for i in range(n):
           sum = sum + sol['x'][i] * K(X[i], X[j])
        rhos.append(sum)
    
    return min(rhos), alphas

#test()
