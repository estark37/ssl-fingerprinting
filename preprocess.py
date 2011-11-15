import os, sys, time, shutil, subprocess
from dateutil import parser
import json

def main():
    input_dir = sys.argv[1]
    data_src = sys.argv[2]
    output_dir = sys.argv[3]
    input_files = os.listdir(input_dir)

    # Some place to put the tcpdump-d raw input files
    try:
        shutil.rmtree("tmp_preprocess")
    except:
        print "No directory to delete"
    os.mkdir("tmp_preprocess")
    
    # Process raw packet traces
    for f in input_files:
        subprocess.call("tcpdump -r %s/%s > tmp_preprocess/%s"%(input_dir, f, f), shell=True)

    # Parse packet trace to feature vector
    for input in input_files:
        feature_vector = []
        f = open("tmp_preprocess/%s"%input)
        start = None
        for p in f:
            fields = p.split(" ")
            # time relative to first packet's time
            t = parser.parse(fields[0])
            if start is None:
                start = t
            tdelta = (t - start).total_seconds()

            # direction of packet
            src = fields[2]
            forwards = 0
            if src.find(data_src) != -1:
                forwards = 1

            # and length
            length = int(fields[-1])

            feature_vector = feature_vector + [tdelta, forwards, length]
        f.close()
        print "Feature vector for %s (length %s)"%(input, len(feature_vector))
        print feature_vector

        out = open("%s/%s"%(output_dir, input), mode="w+")
        out.write(json.dumps(feature_vector))
        out.close()

main()
    
