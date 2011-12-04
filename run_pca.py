import sys
from sklearn.decomposition import PCA
from feature_vectors import(scale, load_feature_vectors, get_target_sites,
select_test_set)
from multiclass_on_binary_svm import classify
from multiclass_svm import SVM_fit, SVM_classify
from anomaly_detection import AnomDet_fit, AnomDet_classify, rbf

def translate(translateDims, X):
    Xnew = []
    for x in X:
        Xnew.append(map(lambda (i, xi):
                            xi + translateDims[i], enumerate(x)))
    return Xnew

def anomaly_detection(labels, Xnew, Y, testXnew, testY):

    print "Anomaly detection"

    for test_class in range(len(labels)):

        anomX = []
        for (i, x) in enumerate(Xnew):
            if Y[i] == test_class:
                anomX.append(x)

        minD = []
        # Translate so all coordinates are positive
        for d in range(len(anomX[0])):
            minD.append(min(map(lambda x: x[d], anomX)))

        anomX = translate(minD, anomX)
        anomX = scale(anomX)
        rho, alphas = AnomDet_fit(anomX, 0.1, rbf)

        test = testXnew
        test = translate(minD, test)
        test = scale(test)
        num_correct = [0, 0]
        predictions = [0, 0]
        for (i, x) in enumerate(test):
            classify = AnomDet_classify(x, alphas, rho, anomX, rbf)
            #print "Label: %d, classification: %d"%(testY[i], classify)
            if testY[i] == test_class:
                predictions[0] = predictions[0] + 1
                if classify == 0.0:
                    num_correct[0] = num_correct[0] + 1
            else:
                predictions[1] = predictions[1] + 1
                if classify != 0.0:
                    num_correct[1] = num_correct[1] + 1

        print "Test class %d. Normal correct: %d/%d, anomaly correct: %d/%d"%(
            test_class, num_correct[0], predictions[0], num_correct[1],
            predictions[1])

def multiclass_svm(Xnew, testXnew, Y, testY, labels):
    print "Classifying with a multiclass SVM"

    Xnew, testXnew = scale(Xnew, testXnew)

    thetas, bs, slacks = SVM_fit(Xnew, Y, len(labels), 0.05)

    num_correct = 0
    for (i, x) in enumerate(testXnew):
        if (SVM_classify(x, thetas, bs) == testY[i]):
            num_correct = num_correct + 1

    print "Num correct: %d/%d"%(num_correct, len(testY))

def multiclass_on_binary_svms(Xnew, Y, testXnew, testY):

    print "Classifying with a one vs one SVM"
    classify(Xnew, Y, testXnew, testY, True, 0.2)

    print "Classifying with a one vs all SVM"
    classify(Xnew, Y, testXnew, testY, False, 0.2)

def main():
    cmd = sys.argv[3]
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])
    target_sites = get_target_sites(input_dir)

    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))
    X, Y = load_feature_vectors(input_dir, num_samples_per_site, labels)
    X, Y, testX, testY = select_test_set(X, Y,
                           (num_samples_per_site / 2) * len(target_sites))
    Y = map(lambda v: v*1.0, Y)
    testY = map(lambda v: v*1.0, testY)

    pca = PCA(n_components = 50)

    print "Fitting X"
    pca.fit(X)

    print "Transforming X and testX"
    Xnew = pca.transform(X)
    testXnew = pca.transform(testX)

    del X
    del testX

    if cmd == "monb":
        multiclass_on_binary_svms(Xnew, Y, testXnew, testY)
    elif cmd == "multiclass":
        multiclass_svm(Xnew, testXnew, Y, testY, labels)
    elif cmd == "anomdet":
        anomaly_detection(labels, Xnew, Y, testXnew, testY)

main()
