import csv
import operator

chs = {}
with open('data/training.1600000.processed.noemoticon.csv', 'rU', encoding='utf8') as trainingfile:
    spamreader = csv.reader(trainingfile, delimiter=',', quotechar='"')
    for row in spamreader:
        for char in row[5]:
            if char not in chs:
                chs[char] = 0
            chs[char] = chs[char] + 1

sorted_chs = sorted(chs.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_chs[:100])

with open('data/chs.data', 'w', encoding='utf8') as chsfile:
    for char in sorted_chs[:100]:
        chsfile.write(char[0] + '')