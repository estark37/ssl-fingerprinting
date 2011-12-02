import os
from sklearn import preprocessing
import json
from random import sample

def load_feature_vectors(input_dir, num_samples_per_site, labels):
    X = []
    Y = []

    input_files = os.listdir(input_dir)

    for site, yi in labels.iteritems():
        for i in range(num_samples_per_site):
            Y.append(yi)

            inp = open("%s/%s_%d.dat"%(input_dir, site, i))
            feature_vector = json.loads(inp.read())
            inp.close()
            X.append(feature_vector)

    # Pad every row of X with -1s to the maximum length of any row
    # (All feature vectors have to be the same length)
    max_len = max(map(lambda v: len(v), X))
    X = map(lambda v: v + ([-1] * (max_len - len(v))), X)

    return X, Y

def scale(X, testX = None):
    newX = map(lambda l: map(lambda x: x*1.0, l), X)
    if testX is not None:
        newTestX = map(lambda l: map(lambda x: x*1.0, l), testX)

    scalerX = preprocessing.Scaler().fit(newX)
    newX = scalerX.transform(newX)
    if testX is not None:
        newTestX = scalerX.transform(newTestX)
        return newX, newTestX

    return newX

def select_test_set(X, Y, n_test):
    tests = sample(range(len(Y)), n_test)
    print "test samples:"
    print tests

    newX = []
    newY = []
    testX = []
    testY = []
    for i in range(len(Y)):
        if i in tests:
            testX.append(X[i])
            testY.append(Y[i])
        else:
            newX.append(X[i])
            newY.append(Y[i])

    return newX, newY, testX, testY

def get_target_sites(input_dir):
    input_files = os.listdir(input_dir)
    input_files = map(lambda f: f[ : f.find("_")], input_files)
    return list(set(input_files))
