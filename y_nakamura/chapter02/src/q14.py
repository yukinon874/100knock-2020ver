import sys
from itertools import islice

n = int(sys.argv[1])
with open('./data/popular-names.txt', mode = 'r') as f:
    for data in islice(f, n):
        print(data.strip())
