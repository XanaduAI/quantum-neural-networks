# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script for creating Plots"""
import numpy as np
import matplotlib.pyplot as plt
import plot_confusion_matrix

# Label for simulation
simulation_label = 1

# Loading confusion table
confusion_table = np.load('./outputs/confusion/' + str(simulation_label) + '/confusion_table.npy')

# Defining array of thresholds from 0 to 1 to consider in the ROC curve
thresholds_points = 101
thresholds = np.linspace(0, 1, num=thresholds_points)

# false/true positive/negative rates
fp_rate = []
tp_rate = []
fn_rate = []
tn_rate = []

# Creating rates
for i in range(thresholds_points):
    fp_rate.append(confusion_table[i, 0, 1] / (confusion_table[i, 0, 1] + confusion_table[i, 0, 0]))
    tp_rate.append(confusion_table[i, 1, 1] / (confusion_table[i, 1, 1] + confusion_table[i, 1, 0]))

    fn_rate.append(confusion_table[i, 1, 0] / (confusion_table[i, 1, 1] + confusion_table[i, 1, 0]))
    tn_rate.append(confusion_table[i, 0, 0] / (confusion_table[i, 0, 0] + confusion_table[i, 0, 1]))

# Distance of each threshold from ideal point at (0, 1)
distance_from_ideal = (np.array(tn_rate) - 1)**2 + (np.array(fn_rate) - 0)**2

# Threshold closest to (0, 1)
closest_threshold = np.argmin(distance_from_ideal)

# Area under ROC curve
area_under_curve = np.trapz(np.sort(tn_rate), x=np.sort(fn_rate))

print("Area under ROC curve: " + str(area_under_curve))
print("Closest threshold to optimal ROC: " + str(thresholds[closest_threshold]))

# Plotting ROC curve
straight_line = np.linspace(0, 1, 1001)

plt.gcf().subplots_adjust(bottom=0.15)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='New Century Schoolbook')
plt.gcf().subplots_adjust(bottom=0.15)
plt.plot(fn_rate, tn_rate, color='#056eee', linewidth=2.2)
plt.plot(straight_line, straight_line, color='#070d0d', linewidth=1.5, dashes=[6, 2])
plt.plot(0.0, 1.0, 'ko')
plt.plot(fn_rate[closest_threshold], tn_rate[closest_threshold], 'k^')
plt.ylim(-0.05, 1.05)
plt.xlim(-0.05, 1.05)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('False negative rate', fontsize=15)
plt.ylabel('True negative rate', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=14, length=6, width=1)
plt.tick_params(axis='both', which='minor', labelsize=14, length=6, width=1)
plt.savefig('./roc.pdf')
plt.close()

# Selecting ideal confusion table and plotting
confusion_table_ideal = confusion_table[closest_threshold]

plt.figure()
plot_confusion_matrix.plot_confusion_matrix(confusion_table_ideal, classes=['Genuine', 'Fraudulent'], title='')

plt.savefig('./confusion.pdf')
