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
"""For processing data from https://www.kaggle.com/mlg-ulb/creditcardfraud"""
import csv
import numpy as np
import random

# creditcard.csv downloaded from https://www.kaggle.com/mlg-ulb/creditcardfraud
with open('creditcard.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    data = list(csv_reader)

data = data[1:]
data_genuine = []
data_fraudulent = []

# Splitting genuine and fraudulent data
for i in range(len(data)):
    if int(data[i][30]) == 0:
        data_genuine.append([float(i) for i in data[i]])
    if int(data[i][30]) == 1:
        data_fraudulent.append([float(i) for i in data[i]])

fraudulent_data_points = len(data_fraudulent)

# We want the genuine data points to be 3x the fraudulent ones
undersampling_ratio = 3

genuine_data_points = fraudulent_data_points * undersampling_ratio

random.shuffle(data_genuine)
random.shuffle(data_fraudulent)

# Fraudulent and genuine transactions are split into two datasets for cross validation

data_fraudulent_1 = data_fraudulent[:int(fraudulent_data_points / 2)]
data_fraudulent_2 = data_fraudulent[int(fraudulent_data_points / 2):]

data_genuine_1 = data_genuine[:int(genuine_data_points / 2)]
data_genuine_2 = data_genuine[int(genuine_data_points / 2):genuine_data_points]
data_genuine_remaining = data_genuine[genuine_data_points:]

random.shuffle(data_fraudulent_1)
random.shuffle(data_fraudulent_2)
random.shuffle(data_genuine_1)
random.shuffle(data_genuine_2)

np.savetxt('creditcard_genuine_1.csv', data_genuine_1, delimiter=',')
np.savetxt('creditcard_genuine_2.csv', data_genuine_2, delimiter=',')
np.savetxt('creditcard_fraudulent_1.csv', data_fraudulent_1, delimiter=',')
np.savetxt('creditcard_fraudulent_2.csv', data_fraudulent_2, delimiter=',')
# Larger datasets are used for testing, including genuine transactions unseen in training
np.savetxt('creditcard_combined_1_big.csv', data_fraudulent_1 + data_genuine_1 + data_genuine_remaining, delimiter=',')
np.savetxt('creditcard_combined_2_big.csv', data_fraudulent_2 + data_genuine_2 + data_genuine_remaining, delimiter=',')
