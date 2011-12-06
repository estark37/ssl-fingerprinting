import sys
from sklearn import svm
import os
from feature_vectors import (scale, load_feature_vectors, select_test_set,
get_target_sites, select_cross_validation_subsets)

def classify(X, Y, testX, testY, onevone = True, c=1.0, kernel='linear'):
    if onevone:
        clf = svm.SVC(kernel = kernel, C=c)
    else:
        clf = svm.sparse.LinearSVC(C=c)

    clf.fit(X, Y)

    correct = 0
    for ind, test in enumerate(testX):
        result = clf.predict(test)[0]
        if result == testY[ind]:
            correct = correct + 1
    
    print "Num correct: %d/%d"%(correct, len(testY))    

def run(input_dir, num_samples_per_site, onevone = True):
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])

    target_sites = get_target_sites(input_dir)

    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))
    
    X, Y = load_feature_vectors(input_dir, num_samples_per_site, labels)
    #X, Y, testX, testY = select_test_set(X, Y, (num_samples_per_site / 2) * len(target_sites))


    #X = [[100.0, 100.0], [101.0, 101.0], [200.0, 200.0], [201.0, 201.0], [300.0, 301.0], [300.0, 301.0]]
    #Y = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]

    iters = select_cross_validation_subsets(X, Y, 6)

    #crange = map(lambda c: c/10.0, range(1, 11))
    #crange = map(lambda c: c/1000.0, range(1, 11))
    #crange.extend(map(lambda c: c/100.0, range(2, 11)))
    #crange = [0.00001, 0.01, 1.0, 100.0, 10000.0]
    crange = [1.0, 100.0]
    for c in crange:
        print "C=%f"%c
        print "Linear kernel:"
        for d in iters:
            trainX, testX = scale(d["train"][0], d["test"][0])
            classify(trainX, d["train"][1], testX, d["test"][1], onevone, c)
        print "RBF kernel:"
        for d in iters:
            trainX, testX = scale(d["train"][0], d["test"][0])
            classify(trainX, d["train"][1], testX, d["test"][1], onevone, c, 'rbf')
