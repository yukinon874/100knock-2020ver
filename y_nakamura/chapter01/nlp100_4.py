import re

if __name__ == '__main__':
    s = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can.'
    a = {1, 5, 6, 7, 8, 9, 15, 16, 19}
    wordIndex = {word[:1] if i + 1 in a else word[:2]:i + 1 for (i, word) in enumerate([word for word in re.split('[. ]', s) if word != ''])}
    print(wordIndex)


