import json
import sys
from sklearn import svm
import os

def load_feature_vectors(input_dir, exclude_sample, target_site):
    X = []
    Y = []
    input_files = os.listdir(input_dir)

    for f in input_files:
        # leave one out for training
        print "Loading feature vector for %s"%f
        exclude = False
        if f.find(target_site) == -1:
            Y.append(0)
        else:
            if f.find("%s_%d"%(target_site, exclude_sample)) != -1:
                exclude = True
            else:
                Y.append(1)

        inp = open("%s/%s"%(input_dir, f))
        feature_vector = json.loads(inp.read())
        inp.close()
        if not exclude:
            X.append(feature_vector)
        else:
            test = feature_vector

    # Pad every row of X with -1s
    max_len = max(map(lambda v: len(v), X))
    X = map(lambda v: v + ([-1] * (max_len - len(v))), X)        
    test.extend([-1] * (max_len - len(test)))

    return X, Y, test
    

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_site = sys.argv[3]

    X, Y, test = load_feature_vectors(input_dir, num_samples_per_site-1, target_site)

    clf = svm.SVC()
    clf.fit(X, Y)

    # Test on the example we left out
    print "Testing on:"
    print test
    print clf.predict(test)

main()
