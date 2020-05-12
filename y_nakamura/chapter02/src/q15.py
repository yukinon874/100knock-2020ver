import sys
from collections import deque

n = int(sys.argv[1])
with open('./data/popular-names.txt', mode = 'r') as f:
    dq = deque(f, n)
    for data in dq:
        print(data.strip())
