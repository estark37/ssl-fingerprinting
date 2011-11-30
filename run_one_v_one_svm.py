import sys
from sklearn import svm
import os
from feature_vectors import scale, load_feature_vectors, select_test_set

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
