import math, os
from feature_vectors import (load_feature_vectors, get_target_sites,
                             select_test_set, scale,
                             select_cross_validation_subsets)
from multiclass_on_binary_svm import classify
import json
import shutil
import sys
from multiclass_svm import SVM_fit, SVM_classify

# nb is number of blocks to divide each packet trace into
# e is measured in bytes
def reduce(X_multi, X_single, n_features, nb, e):
    # 3 features per packet
    if len(X_multi[0]) % 3 != 0 or len(X_single[0]) % 3 != 0:
        return []

    # Precompute the total number of packets of each length and direction 
    # in each block of each single visit.
    print "Precomputing..."
    score_lookup = {}
    for i in range(0, nb):
        sl = {0: {}, 1: {}}
        for x in X_single:
            # Get the i'th block of x
            lx = 0
            try:
                lx = x.index(-1)
            except:
                lx = len(x)

            ppb = (lx / 3) / nb # number of packets divided by number blocks
            if i == nb - 1:
                # last block eats up all leftover packets
                block = x[ppb * i * 3 : lx]
            else:
                block = x[ppb * i * 3 : ppb * i * 3 + ppb * 3]

            lp = len(block) / 3
            for j in range(lp):
                direction = block[j * 3 + 1]
                length = block[j * 3 + 2]
                if (sl[direction].has_key(length)):
                    sl[direction][length] = sl[direction][length] + 1
                else:
                    sl[direction][length] = 1
        score_lookup[i] = sl
            
    # Now we can compute the score of each feature in each vector in X_multi

    Xnew = []
    for x in X_multi:
        scores = [] # a score for each packet
        try:
            lx = x.index(-1)
        except:
            lx = len(x)

        np = lx / 3
        ppb = np / nb
        for i in range(np):
            # figure out which block we're in
            try:
                b = i / ppb
            except:
                b = 0
            if b >= nb:
                b = nb - 1
            score = 0.0
            direction = x[i*3 + 1]
            length = x[i*3 + 2]
            lcounts = score_lookup[b][direction]
            total = sum(map(lambda l: lcounts[l], lcounts.keys()))
            for l in lcounts.keys():
                if l >= length - e and l <= length + e:
                    score = score + (lcounts[l] * 1.0) / total
            scores.append(score)

        scores = map(lambda (i, s): (s, i), enumerate(scores))
        scores.sort()
        scores.reverse()

        newx = []
        packets = []
        i = 0
        while len(packets)*2 < n_features and i < len(scores):
            p = scores[i][1]
            packet = (x[i*3 + 1], x[i*3 + 2])
            try:
                packets.index(packet)
            except:
                newx.append(packet[0])
                newx.append(packet[1])
                packets.append(packet)
            i = i + 1
        Xnew.append(newx)

    # just in case something went wrong, pad all the X's to the max length
    max_len = max(map(lambda v: len(v), Xnew))
    Xnew = map(lambda v: v + ([-1] * (max_len - len(v))), Xnew)

    return Xnew

def multiclass_svm_crossvalidate(X, Y, labels, c):
    iters = select_cross_validation_subsets(X, Y, 6)
    for d in iters:
        X = d["train"][0]
        Y = d["train"][1]
        testX = d["test"][0]
        testY = d["test"][1]
        multiclass_svm(X, Y, testX, testY, labels, c)

        
def multiclass_svm(X, Y, testX, testY, labels, c=1.0):
    X, testX = scale(X, testX)

    try:
        shutil.rmtree("tmp_x")
    except:
        print "No tmp directory to delete"
    os.mkdir("tmp_x")
    out = open("tmp_x/x.dat", mode="w+")
    for x in X:
        out.write(json.dumps(x.tolist()))
        out.write("\n")
    out.close()
    
    thetas, bs, slacks = SVM_fit("tmp_x/x.dat", Y, len(labels), len(X[0]), c)
    num_correct = 0
    for (i, x) in enumerate(testX):
        result = SVM_classify(x, thetas, bs)
        if result == testY[i]:
            num_correct = num_correct + 1
    print "Num correct: %d/%d"%(num_correct, len(testY))

def multiclass_svm_test(X, Y, labels, c, ntests):
    X, Y, testX, testY = select_test_set(X, Y, ntests)
    multiclass_svm(X, Y, testX, testY, labels, c)

def monb_test(X, Y, c, onevone, ntest):
    X, Y, testX, testY = select_test_set(X, Y, ntest)
    Y = map(lambda v: v*1.0, Y)
    classify(X, Y, testX, testY, onevone, c)

def main():
    cmd = sys.argv[5]
    sinput_dir = sys.argv[1]
    minput_dir = sys.argv[2]

    nssamples = int(sys.argv[3])
    nmsamples = int(sys.argv[4])

    target_sites = get_target_sites(sinput_dir)
    labels = dict(map(lambda (i, s): (s, i), enumerate(target_sites)))

    Xs, Ys = load_feature_vectors(sinput_dir, nssamples, labels)
    Xm, Ym = load_feature_vectors(minput_dir, nmsamples, labels)

    if cmd == "multiclass-crossvalidate":
        d = 50
        nb = 15
        e = 100
        c = 0.1
        Xm = reduce(Xm, Xs, d, nb, e)
        multiclass_svm_crossvalidate(Xm, Ym, labels, c)
    elif cmd == "multiclass-test":
        d = 50
        nb = 15
        e = 100
        c = 0.1
        Xm = reduce(Xm, Xs, d, nb, e)
        multiclass_svm_test(Xm, Ym, labels, c, nmsamples*len(labels) / 2)
    elif cmd == "onevone-test":
        c = 1.0
        print "Original dimensionality"
        monb_test(Xm, Ym, c, True, nmsamples*len(labels) / 2)
        d = 50
        nb = 15
        e = 100
        print "Reduced dimensionality"
        Xm = reduce(Xm, Xs, d, nb, e)
        monb_test(Xm, Ym, c, True, nmsamples*len(labels) / 2)
    elif cmd == "onevall-test":
        c = 1.0
        print "Original dimensionality"
        monb_test(Xm, Ym, c, False, nmsamples*len(labels) / 2)
        d = 50
        nb = 15
        e = 100
        print "Reduced dimensionality"
        Xm = reduce(Xm, Xs, d, nb, e)
        monb_test(Xm, Ym, c, False, nmsamples*len(labels) / 2)
        


    #print "Reduced dimensionality test"
    #Xmnew = reduce(Xm, Xs, 50, 15, 100)
    #classify(X, Y, testX, testY, True, 1.0)

    #print "Original dimensionality"
    #X, Y, testX, testY = select_test_set(Xm, Ym, 30*len(labels)/2)
    #classify(X, Y, testX, testY, False, 1.0)

main()
