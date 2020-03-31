from functools import reduce

if __name__ == '__main__':
    a = "パトカー"
    b = "タクシー"
    s = ''.join(char_a + char_b for (char_a, char_b) in zip(a, b))
    print(s)
    s = reduce(lambda word, char_ab: word + ''.join(char_ab), zip(a, b), '')
    print(s)
