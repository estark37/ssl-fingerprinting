import json
import sys
from sklearn import svm
import os

def load_feature_vectors(input_dir, num_samples_per_site, target_sites):
    X = []
    Y = []
    testX = []
    testY = []

    for i in range(num_samples_per_site/2):
        for site in target_sites:
            print "Loading feature vector for %s_%d.dat"%(site, i)
            if site == target_sites[0]:
                Y.append(0)
            else:
                Y.append(1)

            inp = open("%s/%s_%d.dat"%(input_dir, site, i))
            feature_vector = json.loads(inp.read())
            inp.close()
            X.append(feature_vector)

    for i in range(num_samples_per_site/2, num_samples_per_site):
        for site in target_sites:
            print "Loading test vector for %s_%d.dat"%(site, i)
            if site == target_sites[0]:
                testY.append(0)
            else:
                testY.append(1)

            inp = open("%s/%s_%d.dat"%(input_dir, site, i))
            feature_vector = json.loads(inp.read())
            inp.close()
            testX.append(feature_vector)

    # Pad every row of X with -1s
    max_len = max(map(lambda v: len(v), X) + map(lambda v: len(v), testX))
    X = map(lambda v: v + ([-1] * (max_len - len(v))), X)
    testX = map(lambda v: v + ([-1] * (max_len - len(v))), testX)

    return X, Y, testX, testY
    

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_site1 = sys.argv[3]
    target_site2 = sys.argv[4]

    X, Y, testX, testY = load_feature_vectors(input_dir, num_samples_per_site, [target_site1, target_site2])

    clf = svm.SVC(kernel='linear')
    clf.fit(X, Y)

    correct = 0
    predictions = []
    for ind, test in enumerate(testX):
        result = clf.predict(test)[0]
        predictions.append(result)
        if result == testY[ind]:
            correct = correct + 1

    print "Num correct: %d"%correct
        

main()
