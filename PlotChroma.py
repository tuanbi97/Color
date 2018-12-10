import csv
import colour
import matplotlib.pyplot as plt
import colour.plotting as cplot
import Reflectance2XYZ as ref2xyz
import Camera2XYZ as cam2xyz

ppg_data = ref2xyz.read_data('VOC 2014 Color Data.csv')
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
        j_rgb = [float(x) for x in row[-9: -6]]
        #print(j_rgb)
        j_xyz = colour.sRGB_to_XYZ(j_rgb)
        ppg_lab = ppg_data[color_name]['LAB']
        ppg_xyz = colour.Lab_to_XYZ(ppg_lab, illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])
        ppg_srgb = colour.XYZ_to_sRGB(ppg_xyz)
        for tt in range(0, 3):
            if (ppg_srgb[tt] > 1.0 or ppg_srgb[tt] < 0.0):
                print(color_name)
                break
        #print(ppg_srgb)
        xy = colour.XYZ_to_xy(ppg_xyz, illuminant=colour.ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65'])

        x, y = xy
        plt.plot(x, y, 'o-', color='black', markersize=4)

        #print(d_xyz, ' ', c_xyz)

cplot.render(
    standalone=True,
    limits=(-0.1, 0.9, -0.1, 0.9),
    x_tighten=True,
    y_tighten=True)


