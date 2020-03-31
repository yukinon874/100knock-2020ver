from random import sample
from functools import reduce

def shuffleWord(s):
    if len(s) <= 4: return s
    else: return s[0] + ''.join(sample(s[1:-1], len(s) - 2)) + s[-1]

def Typoglycemia(s):
    return reduce(lambda words, word: words + ' ' + shuffleWord(word) , s.split())

if __name__ == '__main__':
    s = 'I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind .'
    print(Typoglycemia(s))
