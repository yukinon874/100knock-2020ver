import sys

n = int(sys.argv[1])
with open('./data/popular-names.txt', mode = 'r') as f:
    data_list = f.readlines()
    for data in data_list[-n:]:
        print(data.strip())
