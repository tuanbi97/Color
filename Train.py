import numpy as np
import Constant as cons
import csv

def read_data(csv_file):
    data = {}
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        cnt = -1
        for row in csv_reader:
            if cnt == -1:
                cnt += 1
                continue
            data[cnt] = {'dataname': row[0], 'number': row[1], 'name': row[2], 'RGB': [int(row[3]), int(row[4]), int(row[5])], 'LAB': [float(row[6]), float(row[7]), float(row[8])]}
            ref = {}
            for i in range(9, len(row)):
                ref[400 + (i - 9) * 10] = float(row[i])
            data[cnt]['ref'] = ref
            cnt += 1
    return data

def reflectance2XYZ(data):
    obsFunc = cons.deg10ObsFunc
    tmp = 0.0
    for i in range(400, 701, 10):
        tmp += cons.d65[i] * obsFunc[i][1]
    K = 1.0 / tmp
    tX = tY = tZ = 0.0
    for i in range(400, 701, 10):
        tX += obsFunc[i][0] * cons.d65[i] * data[i]
        tY += obsFunc[i][1] * cons.d65[i] * data[i]
        tZ += obsFunc[i][2] * cons.d65[i] * data[i]
    return [K * tX, K * tY, K * tZ]

data = read_data('VOC 2014 Color Data.csv')
print(len(data))
for i in range(0, len(data)):
    data[i]['XYZ'] = reflectance2XYZ(data[i]['ref'])
print(data[0]['XYZ'])
print(data[1]['XYZ'])
