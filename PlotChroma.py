import csv
import colour
import matplotlib.pyplot as plt
import colour.plotting as cplot
cplot.RGB_colourspaces_chromaticity_diagram_plot_CIE1931(['sRGB'], standalone=False)

csv_result = './ResultIOS.csv'

with open(csv_result, 'r') as fp:
    n = int(fp.readline())
    for i in range(0, n):
        row = fp.readline().split(',')
        #print(row)
        color_name = row[0].replace('_', ' ')
        c_xyz = [float(x) for x in row[-3:]]
        d_xyz = [float(x) for x in row[-6: -3]]

        xy = colour.XYZ_to_xy(c_xyz, illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
        x, y = xy
        plt.plot(x, y, 'o-', color='black')

        print(d_xyz, ' ', c_xyz)

cplot.render(
    standalone=True,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True)


