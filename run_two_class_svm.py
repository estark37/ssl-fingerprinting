import json
import sys
from sklearn import svm
import os
from random import randint

def load_feature_vectors(input_dir, num_samples_per_site, target_sites):
    X = []
    Y = []
    testX = []
    testY = []

    train_samples = []
    while len(train_samples) < num_samples_per_site / 2:
        next = randint(0, num_samples_per_site - 1)
        if not (next in train_samples):
            train_samples.append(next)

    for i in range(num_samples_per_site):
        for site in target_sites:
            if i in train_samples:
                useY = Y
                useX = X
            else:
                useY = testY
                useX = testX
            if site == target_sites[0]:
                useY.append(0)
            else:
                useY.append(1)
            
            inp = open("%s/%s_%d.dat"%(input_dir, site, i))
            feature_vector = json.loads(inp.read())
            inp.close()
            useX.append(feature_vector)

    # Pad every row of X with -1s to the maximum length of any row
    # (All feature vectors have to be the same length)
    max_len = max(map(lambda v: len(v), X) + map(lambda v: len(v), testX))
    X = map(lambda v: v + ([-1] * (max_len - len(v))), X)
    testX = map(lambda v: v + ([-1] * (max_len - len(v))), testX)

    return X, Y, testX, testY
 

def get_target_site_pairs(input_dir):
    input_files = os.listdir(input_dir)
    input_files = map(lambda f: f[ : f.find("_")], input_files)
    return list(set(input_files))

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
#    target_site1 = sys.argv[3]
#    target_site2 = sys.argv[4]

    target_site_pairs = get_target_site_pairs(input_dir)

    for ind1, s1 in enumerate(target_site_pairs):
        for ind2, s2 in enumerate(target_site_pairs):
            if ind1 < ind2:
                X, Y, testX, testY = load_feature_vectors(input_dir, num_samples_per_site, [s1, s2])

                clf = svm.SVC(kernel='linear')
                clf.fit(X, Y)

                correct = 0
                predictions = []
                for ind, test in enumerate(testX):
                    result = clf.predict(test)[0]
                    predictions.append(result)
                    if result == testY[ind]:
                        correct = correct + 1

                print "Pair: %s, %s"%(s1, s2)
                print "Num correct: %d"%correct
        

main()
