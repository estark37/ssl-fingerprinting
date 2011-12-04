import sys
from sklearn import svm
import os
from feature_vectors import (scale, load_feature_vectors, select_test_set,
get_target_sites)

def classify(X, Y, testX, testY, onevone = True, c=1.0):
    if onevone:
        clf = svm.SVC(kernel = 'linear', C=c)
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
    X, Y, testX, testY = select_test_set(X, Y, (num_samples_per_site / 2) * len(target_sites))

    X, testX = scale(X, testX)

    classify(X, Y, testX, testY, onevone, 0.002)
