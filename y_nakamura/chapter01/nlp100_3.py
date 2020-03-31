import re

if __name__ == '__main__':
    s = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    wordSizeList = [len(word) for word in re.split('[,. ]', s) if word != '']
    print(wordSizeList)
