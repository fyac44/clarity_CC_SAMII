import json
import csv
from matplotlib.markers import MarkerStyle
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.optimize import curve_fit
from clarity_core.config import CONFIG
from lib_samii import experiment

def sigmoid(x ,x0, k):
    y = 100 / (1 + np.exp(-k*(x-x0)))
    return (y)

data_path = 'data/clarity_data/clarity_data/mutualinfo/train_indep/'
test_path = 'data/clarity_data/clarity_data/mutualinfo/test_indep/'
figures_path = 'projects/SAMII/figures/'
metadata_file = 'data/clarity_data/metadata/CPC1.train_indep.json'
baseline_file = 'scripts/mbstoi.train_indep.baseline.csv'
output_file = 'scripts/samii.test_indep.predictions.csv'

data_list = os.listdir(data_path)
pre_silence = CONFIG.pre_duration

baseline = dict()
with open(baseline_file, mode='r') as csv_file:
    baseline_raw = csv.DictReader(csv_file)
    for row in baseline_raw:
        bl_key = row['scene'] + '_' + row['listener'] + '_' + row['system']
        baseline[bl_key] = row['mbstoi']

with open(metadata_file) as json_metadata:
    metadata_raw = json.load(json_metadata)

metadata = dict()
for md in metadata_raw:
    md_key = md['signal']
    metadata[md_key] = md

del metadata_raw
del baseline_raw

mbstoi_dict = dict()
samii_dict = dict()
correctness_dict = dict()
mbstoi_lst = list()
samii_lst = list()
validation_mbstoi_lst = list()
validation_samii_lst = list()
validation_correctness_lst = list()
correctness_lst = list()

validate = 0
plotit = 0

for file in data_list:
#for _ in range(1):
    #file = 'S09352_L0229_E001.json'
    filename = file.split('.')[0]
    with open(data_path+file) as json_data:
        data = json.load(json_data)
    
    exp = experiment(data, filename, pre_silence)

    samii = exp.get_samii()
    correctness = metadata[filename]['correctness']
    listener = metadata[filename]['listener']
    system = metadata[filename]['system']
    scene = metadata[filename]['scene']
    if baseline.get(filename) is None:
        continue

    if plotit < 0 and system == 'E003':
        exp.generate_plots('./projects/SAMII/figures/')
        plotit += 1

    mbstoi = float(baseline[filename])

    #
    mbstoi_dict[listener] = mbstoi_dict.get(listener, list())
    mbstoi_dict[system] = mbstoi_dict.get(system, list())
    mbstoi_dict[scene] = mbstoi_dict.get(scene, list())
    samii_dict[listener] = samii_dict.get(listener, list())
    samii_dict[system] = samii_dict.get(system, list())
    samii_dict[scene] = samii_dict.get(scene, list())
    correctness_dict[listener] = correctness_dict.get(listener, list())
    correctness_dict[system] = correctness_dict.get(system, list())
    correctness_dict[scene] = correctness_dict.get(scene, list())

    mbstoi_dict[listener].append(mbstoi)
    mbstoi_dict[system].append(mbstoi)
    mbstoi_dict[scene].append(mbstoi)
    samii_dict[listener].append(samii)
    samii_dict[system].append(samii)
    samii_dict[scene].append(samii)
    correctness_dict[listener].append(correctness)
    correctness_dict[system].append(correctness)
    correctness_dict[scene].append(correctness)

    if validate == -1:
        validation_mbstoi_lst.append(mbstoi)
        validation_samii_lst.append(samii)
        validation_correctness_lst.append(correctness)
        validate = 0
    else:
        mbstoi_lst.append(mbstoi)
        samii_lst.append(samii)
        correctness_lst.append(correctness)
        validate = validate + 1

    print(filename+':', samii, mbstoi, correctness)
'''
i = 0
for k in samii_col.keys():
    if k.startswith('S') and i<10:
        i = i + 1
        plt.scatter(samii_dict[k], correctness_dict[k])
'''

mbstoi_lst[mbstoi_lst != mbstoi_lst] = 0.5

samii_p0 = [np.median(samii_lst),100] # this is an mandatory initial guess
mbstoi_p0 = [np.median(samii_lst),1] # this is an mandatory initial guess

samii_popt, samii_pcov = curve_fit(sigmoid, samii_lst, correctness_lst, samii_p0, method='lm')
mbstoi_popt, mbstoi_pcov = curve_fit(sigmoid, mbstoi_lst, correctness_lst, mbstoi_p0, method='lm')

samii_points = np.arange(len(samii_lst))/(len(samii_lst)-1) * max(samii_lst)
mbstoi_points = np.arange(len(mbstoi_lst))/(len(mbstoi_lst)-1) * max(mbstoi_lst)

samii_sigmoid = sigmoid(samii_points, samii_popt[0], samii_popt[1])
mbstoi_sigmoid = sigmoid(mbstoi_points, mbstoi_popt[0], mbstoi_popt[1])
'''
samii_fit = sigmoid(validation_samii_lst, samii_popt[0], samii_popt[1])
mbstoi_fit = sigmoid(validation_mbstoi_lst, mbstoi_popt[0], mbstoi_popt[1])

samii_rmse = np.sqrt(np.square(np.subtract(samii_fit, validation_correctness_lst)).mean())
mbstoi_rmse = np.sqrt(np.square(np.subtract(mbstoi_fit, validation_correctness_lst)).mean())

fig1, ax_sf = plt.subplots()
fig2, ax_sp = plt.subplots()
fig3, (ax_mf, ax_mp) = plt.subplots(1, 2, sharey=True)

ax_sf.scatter(samii_lst, correctness_lst, 5, c='gray', marker='x', label='fitting scenes')
ax_sf.plot(samii_points, samii_sigmoid, color='black',  label='fitted sigmoid')
ax_sf.set_title('SAMII Fitting')
ax_sf.set_xlim((0.9*min(samii_lst), 0.1*min(samii_lst)+max(samii_lst)))
ax_sf.set_ylim((-5, 105))
ax_sf.set_yticks([0, 25, 50, 75, 100])

ax_sp.scatter(samii_fit, validation_correctness_lst, 5, c='gray', marker='x', label='validation scenes')
ax_sp.plot([0, 100], [0, 100], color='black', label='ideal values')
ax_sp.set_title('SAMII Predictions')
ax_sp.set_xlim((-5, 105))
ax_sp.set_xticks([0, 25, 50, 75, 100])
ax_sp.set_ylim((-5, 105))
ax_sp.set_yticks([0, 25, 50, 75, 100])

ax_mf.scatter(mbstoi_lst, correctness_lst, 5, c='gray', marker='x')
ax_mf.plot(mbstoi_points, mbstoi_sigmoid, color='black')
ax_mf.set_title('MBSTOI Fitting')

ax_mp.scatter(mbstoi_fit, validation_correctness_lst, 5, c='gray', marker='x')
ax_mp.set_title('MBSTOI Predictions')

ax_sf.set_xlabel('SAMII')
ax_sp.set_xlabel('Prediction [%] (SAMII) - rmse: {:.2f}'.format(samii_rmse))
ax_sf.set_ylabel('Correctness [%]')
ax_sp.set_ylabel('Correctness [%]')
ax_sf.legend()
ax_sp.legend()
ax_mf.set_xlabel('MBSTOI')
ax_mp.set_xlabel('Prediction (MBSTOI) - rmse: {:.2f}'.format(mbstoi_rmse))
ax_mf.set_ylabel('Correctness')

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

fig1.savefig(figures_path + 'samii_fit.png')
fig2.savefig(figures_path + 'samii_val.png')
fig3.savefig(figures_path + 'samii_mbstoi.png')
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)

'''

test_list = os.listdir(test_path)
filename_lst = list()
predictions_lst = list()

for file in test_list:
#for _ in range(1):
    #file = 'S09352_L0229_E001.json'
    filename = file.split('.')[0]
    with open(test_path+file) as json_data:
        data = json.load(json_data)
    
    exp = experiment(data, filename, pre_silence)
    #exp.generate_plots('./projects/SAMII/figures/')
    samii = exp.get_samii()
    prediction = sigmoid(samii, samii_popt[0], samii_popt[1])

    filename_lst.append(filename)
    predictions_lst.append(prediction)

    print(filename+':', samii, prediction)

fieldnames = ['signal', 'prediction']
with open(output_file, mode='w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predictions_lst)):
        writer.writerow({'signal': filename_lst[i], 'prediction': predictions_lst[i]})
