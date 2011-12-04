import sys
import os
from feature_vectors import (scale, load_feature_vectors, select_test_set,
get_target_sites)
from multiclass_svm import SVM_fit, SVM_classify

# NOTE: this file is outdated, was just used for testing purposes

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_sites = get_target_sites(input_dir)

    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))

    X, Y = load_feature_vectors(input_dir, num_samples_per_site, labels)
    X, Y, testX, testY = select_test_set(X, Y, (num_samples_per_site / 2) * len(target_sites))

    # X, testX = scale(X, testX)

    X = [[0.0, 0.0], [0.01, 0.01], [1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [2.1, 2.1], [2.05, 2.05]]#, [2.05, 2.05], [2.05, 2.07], [2.01, 2.01]]
    Y=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0]#, 1.0, 2.0, 2.0]
    labels = {0: 0, 1: 1, 2: 2}

    thetas, bs, slacks = SVM_fit(X, Y, len(labels), 5.0)

    print "thetas"
    print thetas[0]
    print thetas[1]
    print thetas[2]
    print "bs"
    print bs
    print "slacks"
    print slacks


    print "Should be class 0:"
    print SVM_classify([0.05, 0.05], thetas, bs)
    print "Should be class 1:"
    print SVM_classify([1.05, 1.05], thetas, bs)
    print "Should be class 2:"
    print SVM_classify([1.98, 1.98], thetas, bs)



main()
