import json
import sys
from sklearn import svm
from sklearn import preprocessing
import os
from random import randint, sample

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
    X = map(lambda v: v + ([0] * (max_len - len(v))), X)

    return X, Y

def scale(X, testX):
    newX = map(lambda l: map(lambda x: x*1.0, l), X)
    newTestX = map(lambda l: map(lambda x: x*1.0, l), testX)

    scalerX = preprocessing.Scaler().fit(newX)
    newX = scalerX.transform(newX)
    newTestX = scalerX.transform(newTestX)

    return newX, newTestX

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

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_sites = get_target_sites(input_dir)

    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))
    
    X, Y = load_feature_vectors(input_dir, num_samples_per_site, labels)
    X, Y, testX, testY = select_test_set(X, Y, (num_samples_per_site / 2) * len(target_sites))

    X, testX = scale(X, testX)

    clf = svm.SVC(kernel = 'linear')
    clf.fit(X, Y)

    correct = 0
    for ind, test in enumerate(testX):
        result = clf.predict(test)[0]
        print "Prediction: %f, real label: %f"%(result, testY[ind])
        if result == testY[ind]:
            correct = correct + 1
    
    print "Num correct: %d/%d"%(correct, len(testY))

main()
