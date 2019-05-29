short_pos = open("positive.csv", "r")
short_neg = open("negative.csv", "r")


documents_pos = []
documents_neg = []

for i in range(60000):
    _str = short_pos.readline().split(';')

    if len(_str) > 4:
        if len(_str[3]) > 4:
            documents_pos.append(''.join((_str[3].strip(), '\n')))

for i in range(60000):
    _str = short_neg.readline().split(';')
    if len(_str) > 4:
        if len(_str[3]) > 4:
            documents_neg.append(''.join((_str[3].strip(), '\n')))

short_pos.close()
short_neg.close()

short_pos = open("positive.txt", "w")
short_neg = open("negative.txt", "w")

for i in range(14000):
    short_pos.write(documents_pos[i])
    short_neg.write(documents_neg[i])

short_neg.close()
short_pos.close()
