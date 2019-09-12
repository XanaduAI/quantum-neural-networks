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
depth = 4

# Fock basis truncation
cutoff = 10
# Number of batches in optimization
reps = 50000

# Label for simulation
simulation_label = 1

# Number of batches to use in the optimization
batch_size = 24

# Random initialization of gate parameters
sdev_photon = 0.1
sdev = 1

# Variable clipping values
disp_clip = 5
sq_clip = 5
kerr_clip = 1

# If loading from checkpoint, previous batch number reached
ckpt_val = 0

# Number of repetitions between each output to TensorBoard
tb_reps = 100
# Number of repetitions between each model save
savr_reps = 1000

model_string = str(simulation_label)

# Target location of output
folder_locator = './outputs/'

# Locations of TensorBoard and model save outputs
board_string = folder_locator + 'tensorboard/' + model_string + '/'
checkpoint_string = folder_locator + 'models/' + model_string + '/'

# ===================================================================================
#                                   Loading the training data
# ===================================================================================

# Data outputted from data_processor.py
data_genuine = np.loadtxt('creditcard_genuine_1.csv', delimiter=',')
data_fraudulent = np.loadtxt('creditcard_fraudulent_1.csv', delimiter=',')

# Combining genuine and fraudulent data
data_combined = np.append(data_genuine, data_fraudulent, axis=0)
data_points = len(data_combined)

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
        | q[1]

        Kgate(tf.clip_by_value(output_layer[:, 12], -kerr_clip, kerr_clip)) | q[0]
        Kgate(tf.clip_by_value(output_layer[:, 13], -kerr_clip, kerr_clip)) | q[1]


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
#                                   Setting up cost function
# ===================================================================================

# Classifications for whole batch: rows act as data points in the batch and columns
# are the one-hot classifications
classification = tf.placeholder(shape=[batch_size, 2], dtype=tf.int32)

func_to_minimise = 0

# Building up the function to minimize by looping through batch
for i in range(batch_size):
    # Probabilities corresponding to a single photon in either mode
    prob = tf.abs(ket[i, classification[i, 0], classification[i, 1]]) ** 2
    # These probabilities should be optimised to 1
    func_to_minimise += (1.0 / batch_size) * (prob - 1) ** 2

# Defining the cost function
cost_func = func_to_minimise
tf.summary.scalar('Cost', cost_func)

# ===================================================================================
#                                   Training
# ===================================================================================

# We choose the Adam optimizer
optimiser = tf.train.AdamOptimizer()
training = optimiser.minimize(cost_func)

# Saver/Loader for outputting model
saver = tf.train.Saver(parameters)

session = tf.Session()
session.run(tf.global_variables_initializer())

# Load previous model if non-zero ckpt_val is specified
if ckpt_val != 0:
    saver.restore(session, checkpoint_string + 'Sess.ckpt-' + str(ckpt_val))

# TensorBoard writer
writer = tf.summary.FileWriter(board_string)
merge = tf.summary.merge_all()

counter = ckpt_val

# Tracks optimum value found (set high so first iteration encodes value)
opt_val = 1e20
# Batch number in which optimum value occurs
opt_position = 0
# Flag to detect if new optimum occured in last batch
new_opt = False

while counter <= reps:

    # Shuffles data to create new epoch
    np.random.shuffle(data_combined)

    # Splits data into batches
    split_data = np.split(data_combined, data_points / batch_size)

    for batch in split_data:

        if counter > reps:
            break

        # Input data (provided as principal components)
        data_points_principal_components = batch[:, 1:input_neurons + 1]
        # Data classes
        classes = batch[:, -1]

        # Encoding classes into one-hot form
        one_hot_input = np.zeros((batch_size, 2))

        for i in range(batch_size):
            if int(classes[i]) == 0:
                # Encoded such that genuine transactions should be outputted as a photon in the first mode
                one_hot_input[i] = [1, 0]
            else:
                one_hot_input[i] = [0, 1]

        # Output to TensorBoard
        if counter % tb_reps == 0:
            [summary, training_run, func_to_minimise_run] = session.run([merge, training, func_to_minimise],
                                                                        feed_dict={
                                                                            input_classical_layer:
                                                                                data_points_principal_components,
                                                                            classification: one_hot_input})
            writer.add_summary(summary, counter)

        else:
            # Standard run of training
            [training_run, func_to_minimise_run] = session.run([training, func_to_minimise], feed_dict={
                input_classical_layer: data_points_principal_components, classification: one_hot_input})

        # Ensures cost function is well behaved
        if np.isnan(func_to_minimise_run):
            compute_grads = session.run(optimiser.compute_gradients(cost_func),
                                        feed_dict={input_classical_layer: data_points_principal_components,
                                                   classification: one_hot_input})
            if not os.path.exists(checkpoint_string):
                os.makedirs(checkpoint_string)
            # If cost function becomes NaN, output value of gradients for investigation
            np.save(checkpoint_string + 'NaN.npy', compute_grads)
            print('NaNs outputted - leaving at step ' + str(counter))
            raise SystemExit

        # Test to see if new optimum found in current batch
        if func_to_minimise_run < opt_val:
            opt_val = func_to_minimise_run
            opt_position = counter
            new_opt = True

        # Save model every fixed number of batches, provided a new optimum value has occurred
        if (counter % savr_reps == 0) and (i != 0) and new_opt and (not np.isnan(func_to_minimise_run)):
            if not os.path.exists(checkpoint_string):
                os.makedirs(checkpoint_string)
            saver.save(session, checkpoint_string + 'sess.ckpt', global_step=counter)
            # Saves position of optimum and corresponding value of cost function
            np.savetxt(checkpoint_string + 'optimum.txt', [opt_position, opt_val])

        counter += 1
