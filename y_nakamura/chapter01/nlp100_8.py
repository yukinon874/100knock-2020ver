def chipher(s):
    return ''.join(map(lambda x:chr(219 - ord(x)) if x.islower() else x, s))

if __name__ == '__main__':
    print(chipher('AaBb-0:!'))
