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
"""Fraud detection fitting script"""
import numpy as np
import os

import tensorflow as tf

import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate, Kgate, Sgate, Rgate

# ===================================================================================
#                                   Hyperparameters
# ===================================================================================

# Two modes required: one for "genuine" transactions and one for "fradulent"
mode_number = 2
# Number of photonic quantum layers
depth = 6

# Fock basis truncation
cutoff = 10

# Label for simulation
simulation_label = 1

# Random initialization of gate parameters
sdev_photon = 0.1
sdev = 1

# Variable clipping values
disp_clip = 5
sq_clip = 5
kerr_clip = 1

# If loading from checkpoint, previous batch number reached
ckpt_val = 0

model_string = str(simulation_label)

# Target location of output
folder_locator = './outputs/'

# Locations of model saves and where confusion matrix will be saved
checkpoint_string = folder_locator + 'models/' + model_string + '/'
confusion_string = folder_locator + 'confusion/' + model_string + '/'

# ===================================================================================
#                                   Loading the training data
# ===================================================================================

# Loading combined dataset with extra genuine datapoints unseen in training
data_combined = np.loadtxt('./creditcard_combined_2_big.csv', delimiter=',')

# Set to a size so that the data can be equally split up with no remainder
batch_size = 29

data_combined_points = len(data_combined)

# ===================================================================================
#                                   Setting up the classical NN input
# ===================================================================================

# Input neurons
input_neurons = 10
# Widths of hidden layers
nn_architecture = [10, 10]
# Output neurons of classical part
output_neurons = 14

# Defining classical network parameters
input_classical_layer = tf.placeholder(tf.float32, shape=[batch_size, input_neurons])

layer_matrix_1 = tf.Variable(tf.random_normal(shape=[input_neurons, nn_architecture[0]]))
offset_1 = tf.Variable(tf.random_normal(shape=[nn_architecture[0]]))

layer_matrix_2 = tf.Variable(tf.random_normal(shape=[nn_architecture[0], nn_architecture[1]]))
offset_2 = tf.Variable(tf.random_normal(shape=[nn_architecture[1]]))

layer_matrix_3 = tf.Variable(tf.random_normal(shape=[nn_architecture[1], output_neurons]))
offset_3 = tf.Variable(tf.random_normal(shape=[output_neurons]))

# Creating hidden layers and output
layer_1 = tf.nn.elu(tf.matmul(input_classical_layer, layer_matrix_1) + offset_1)
layer_2 = tf.nn.elu(tf.matmul(layer_1, layer_matrix_2) + offset_2)

output_layer = tf.nn.elu(tf.matmul(layer_2, layer_matrix_3) + offset_3)

# ===================================================================================
#                                   Defining QNN parameters
# ===================================================================================

# Number of beamsplitters in interferometer
bs_in_interferometer = int(1.0 * mode_number * (mode_number - 1) / 2)

with tf.name_scope('variables'):
    bs_variables = tf.Variable(tf.random_normal(shape=[depth, bs_in_interferometer, 2, 2]
                                                , stddev=sdev))
    phase_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number, 2], stddev=sdev))

    sq_magnitude_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                          , stddev=sdev_photon))
    sq_phase_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                      , stddev=sdev))
    disp_magnitude_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                            , stddev=sdev_photon))
    disp_phase_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number]
                                                        , stddev=sdev))
    kerr_variables = tf.Variable(tf.random_normal(shape=[depth, mode_number], stddev=sdev_photon))

parameters = [layer_matrix_1, offset_1, layer_matrix_2, offset_2, layer_matrix_3, offset_3, bs_variables,
              phase_variables, sq_magnitude_variables, sq_phase_variables, disp_magnitude_variables,
              disp_phase_variables, kerr_variables]


# ===================================================================================
#                                   Constructing quantum layers
# ===================================================================================


# Defining input QNN layer, whose parameters are set by the outputs of the classical network
def input_qnn_layer():
    with tf.name_scope('inputlayer'):
        Sgate(tf.clip_by_value(output_layer[:, 0], -sq_clip, sq_clip), output_layer[:, 1]) | q[0]
        Sgate(tf.clip_by_value(output_layer[:, 2], -sq_clip, sq_clip), output_layer[:, 3]) | q[1]

        BSgate(output_layer[:, 4], output_layer[:, 5]) | (q[0], q[1])

        Rgate(output_layer[:, 6]) | q[0]
        Rgate(output_layer[:, 7]) | q[1]

        Dgate(tf.clip_by_value(output_layer[:, 8], -disp_clip, disp_clip), output_layer[:, 9]) \
        | q[0]
        Dgate(tf.clip_by_value(output_layer[:, 10], -disp_clip, disp_clip), output_layer[:, 11]) \
        | q[0]

        Kgate(tf.clip_by_value(output_layer[:, 12], -kerr_clip, kerr_clip)) | q[0]
        Kgate(tf.clip_by_value(output_layer[:, 13], -kerr_clip, kerr_clip)) | q[0]


# Defining standard QNN layers
def qnn_layer(layer_number):
    with tf.name_scope('layer_{}'.format(layer_number)):
        BSgate(bs_variables[layer_number, 0, 0, 0], bs_variables[layer_number, 0, 0, 1]) \
        | (q[0], q[1])

        for i in range(mode_number):
            Rgate(phase_variables[layer_number, i, 0]) | q[i]

        for i in range(mode_number):
            Sgate(tf.clip_by_value(sq_magnitude_variables[layer_number, i], -sq_clip, sq_clip),
                  sq_phase_variables[layer_number, i]) | q[i]

        BSgate(bs_variables[layer_number, 0, 1, 0], bs_variables[layer_number, 0, 1, 1]) \
        | (q[0], q[1])

        for i in range(mode_number):
            Rgate(phase_variables[layer_number, i, 1]) | q[i]

        for i in range(mode_number):
            Dgate(tf.clip_by_value(disp_magnitude_variables[layer_number, i], -disp_clip,
                                   disp_clip), disp_phase_variables[layer_number, i]) | q[i]

        for i in range(mode_number):
            Kgate(tf.clip_by_value(kerr_variables[layer_number, i], -kerr_clip, kerr_clip)) | q[i]


# ===================================================================================
#                                   Defining QNN
# ===================================================================================

# construct the two-mode Strawberry Fields engine
eng, q = sf.Engine(num_subsystems=mode_number)

# construct the circuit
with eng:
    input_qnn_layer()

    for i in range(depth):
        qnn_layer(i)

# run the engine (in batch mode)
state = eng.run('tf', cutoff_dim=cutoff, eval=False, batch_size=batch_size)
# extract the state
ket = state.ket()

# ===================================================================================
#                                   Extracting probabilities
# ===================================================================================

# Classifications for whole batch: rows act as data points in the batch and columns
# are the one-hot classifications
classification = tf.placeholder(shape=[batch_size, 2], dtype=tf.int32)

prob = []

for i in range(batch_size):
    # Finds the probability of a photon being in either mode
    prob.append([tf.abs(ket[i, 1, 0]) ** 2, tf.abs(ket[i, 0, 1]) ** 2])

# ===================================================================================
#                                   Testing performance
# ===================================================================================

# Defining array of thresholds from 0 to 1 to consider in the ROC curve
thresholds_points = 101
thresholds = np.linspace(0, 1, num=thresholds_points)

# Saver/Loader for outputting model
saver = tf.train.Saver(parameters)

session = tf.Session()
session.run(tf.global_variables_initializer())

saver.restore(session, checkpoint_string + 'sess.ckpt-' + str(ckpt_val))

# Split up data to process in batches
data_split = np.split(data_combined, data_combined_points / batch_size)

# Defining confusion table
confusion_table = np.zeros((thresholds_points, 2, 2))

for batch in data_split:
    # Input data (provided as principal components)
    data_points_principal_components = batch[:, 1:input_neurons + 1]
    # Data classes
    classes = batch[:, -1]

    # Probabilities outputted from circuit
    prob_run = session.run(prob, feed_dict={input_classical_layer: data_points_principal_components})

    for i in range(batch_size):
        # Calculate probabilities of photon coming out of either mode
        p = prob_run[i]
        # Normalize to these two events (i.e. ignore all other outputs)
        p = p / np.sum(p)

        # Predicted class is a list corresponding to threshold probabilities
        predicted_class = []

        for j in range(thresholds_points):
            # If probability of a photon exiting first mode is larger than threshold, attribute as genuine
            if p[0] > thresholds[j]:
                predicted_class.append(0)
            else:
                predicted_class.append(1)

        actual_class = classes[i]

        # Constructing confusion table
        for j in range(2):
            for k in range(2):
                for l in range(thresholds_points):
                    if actual_class == j and predicted_class[l] == k:
                        confusion_table[l, j, k] += 1

# Renormalizing confusion table
for i in range(thresholds_points):
    confusion_table[i] = confusion_table[i] / data_combined_points * 100

if not os.path.exists(confusion_string):
    os.makedirs(confusion_string)

# Save as numpy array
np.save(confusion_string + 'confusion_table.npy', confusion_table)
