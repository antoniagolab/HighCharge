import pandas as pd
import csv
counter = 0
all = []
with open('raststationen.csv', 'r', encoding='utf-8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\n')
    for row in spamreader:
        all.append(row[0])


header = all[0:8]
header[0] = 'ID'
rest = all[8:len(all)]
table = pd.DataFrame()
for ij in range(0, len(rest), 8):
    table = table.append({header[0]:int(rest[ij]), header[1]: rest[ij+1], header[2]: rest[ij+2],
                          header[3]: float(rest[ij+3]), header[4]: int(rest[ij+4]), header[5]: int(rest[ij+5]),
                          header[6]: float(rest[ij+6]), header[7]: float(rest[ij+7])}, ignore_index=True)

table.to_csv("ordered_raststaetten.csv", index=False, encoding='utf-8')




