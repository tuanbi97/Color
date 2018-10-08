import Constant as cons
import csv

def read_data(csv_file):
    data = {}
    with open(csv_file, encoding='cp932', errors='ignore') as csv_file:
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
    obsFunc = cons.deg2ObsFunc
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

def XYZ2LAB(xyz, ill):
    if ill == 'a':
        wp = [1.0985, 1.0000, 0.3558]
    if ill == 'c':
        wp = [0.9807, 1.0000, 1.1822]
    if ill == 'e':
        wp = [1.000, 1.000, 1.000]
    if ill == 'd50':
        wp = [0.9642, 1.0000, 0.8251]
    if ill == 'd55':
        wp = [0.9568, 1.0000, 0.9214]
    if ill == 'd65':
        wp = [0.9504, 1.0000, 1.0888]
    if ill == 'icc':
        wp = [0.9642, 1.000, 0.8249]
    e = 0.008856
    k = 903.3
    xr = xyz[0] / wp[0]
    yr = xyz[1] / wp[1]
    zr = xyz[2] / wp[2]
    if xr > e:
        fx = xr**(1.0/3.0)
    else:
        fx = (k*xr + 16)/116.0

    if yr > e:
        fy = yr**(1.0/3.0)
    else:
        fy = (k*yr + 16)/116.0

    if zr > e:
        fz = zr**(1.0/3.0)
    else:
        fz = (k*zr + 16)/116.0

    return [116*fy - 16, 500*(fx - fy), 200*(fy - fz)]
