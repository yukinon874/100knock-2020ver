from nlp100_5 import n_gram

if __name__ == '__main__':
    x = 'paraparaparadise'
    y = 'paragraph'
    X = set(n_gram(x, 2))
    Y = set(n_gram(y, 2))
    print(X | Y)
    print(X & Y)
    print(X - Y)
    print('se' in X)
    print('se' in Y)


