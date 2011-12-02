import sys
from one_v_one_svm import run

def main():
    input_dir = sys.argv[1]
    num_samples_per_site = int(sys.argv[2])

    run(input_dir, num_samples_per_site)

main()
