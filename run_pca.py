import sys
from sklearn.decomposition import PCA
from feature_vectors import(scale, load_feature_vectors, get_target_sites,
select_test_set)
from one_v_one_svm import classify
from multiclass_svm import SVM_fit, SVM_classify

def main():
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

    Xnew = pca.transform(X)
    testXnew = pca.transform(testX)

    X, testX = scale(Xnew, testXnew)

    #print "Classifying with a one vs one SVM"

    #classify(Xnew, Y, testXnew, testY, 0.002)

    print "Classifying with a multiclass SVM"

    thetas, bs, slacks = SVM_fit(X, Y, len(labels), 15.0)

    num_correct = 0
    for (i, x) in enumerate(testX):
        if (SVM_classify(x, thetas, bs) == testY[i]):
            num_correct = num_correct + 1

    print "Num correct: %d/%d"%(num_correct, len(testY))

main()
