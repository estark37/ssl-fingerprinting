import sys
import os
from feature_vectors import (scale, load_feature_vectors, select_test_set,
get_target_sites)
from multiclass_svm import SVM_fit

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_sites = get_target_sites(input_dir)

    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))

    X, Y = load_feature_vectors(input_dir, num_samples_per_site, labels)
    X, Y, testX, testY = select_test_set(X, Y, (num_samples_per_site / 2) * len(target_sites))

    # X, testX = scale(X, testX)

    # X = [[0.0, 0.0], [0.01, 0.01], [1.0, 1.0], [1.0, 1.0], [1.1, 1.1], [2.0, 2.0], [2.1, 2.1]]
    # Y=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    # labels = {0: 0, 1: 1, 2: 2}

    thetas, bs = SVM_fit(X, Y, len(labels))

main()
