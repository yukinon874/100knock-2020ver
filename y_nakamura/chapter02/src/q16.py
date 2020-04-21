import sys
import math
from itertools import islice

file_path = './data/popular-names.txt'

def get_file_length(file_path):
    with open(file_path, mode = 'r') as f:
        return len(f.readlines())

n = int(sys.argv[1])
with open(file_path, mode = 'r') as f:
    num_line = get_file_length(file_path)
    out_num_line = math.ceil(num_line / n)
    for file_number in range(n):
        with open(f'./work/splitpy/splited_popular-names-{file_number:02}.txt', mode = 'w') as out_file:
            for line in islice(f, out_num_line):
                out_file.write(line)
