def n_gram(s, n):
    return [s[i:i + n] for i in range(len(s) - n + 1)]

if __name__ == '__main__':
    s = 'I am an NLPer'
    print(n_gram(s, 2))
    print(n_gram(s.split(), 2))

