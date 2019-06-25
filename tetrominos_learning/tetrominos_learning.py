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
""" This script trains a quantum network for encoding Tetris images in the quantum state of two bosonic modes."""

import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate, Kgate, Sgate, Rgate
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# =============================================
#         Settings and hyperparameters
# =============================================

# Model name
model_string = 'tetris'

# Output folder
folder_locator = './outputs/'

# Locations of TensorBoard and model saving outputs
board_string = folder_locator + 'tensorboard/' + model_string + '/'
save_string = folder_locator + 'models/' + model_string + '/'


# Record initial time
init_time = time.time()

# Set seed for random generator
tf.set_random_seed(1)

# Depth of the quantum network (suggested: 25)
depth = 2

# number of modes
modes = 2

# Fock basis truncation
cutoff = 11  # suggested value: 11

# Image size (im_dim X im_dim)
im_dim = 4

# Number of optimization steps (suggested: 50000)
reps = 20

# Number of steps between data logging/saving.
partial_reps = 2

# Number of images to encode (suggested: 7)
num_images = 7

# Clipping of training parameters
disp_clip = 5
sq_clip = 5
kerr_clip = 1

# Weight for quantum state normalization
norm_weight = 100.0

# ====================================================
#        Manual definition of target images
# ====================================================

train_images = np.zeros((num_images, im_dim, im_dim))

# Target images: L,O,T,I,S,J,Z tetrominos.
L = O = T = I = S = J = Z = np.zeros((im_dim, im_dim))

L[0, 0] = L[1, 0] = L[2, 0] = L[2, 1] = 1 / np.sqrt(4)
O[0, 0] = O[1, 1] = O[0, 1] = O[1, 0] = 1 / np.sqrt(4)
T[0, 0] = T[0, 1] = T[0, 2] = T[1, 1] = 1 / np.sqrt(4)
I[0, 0] = I[1, 0] = I[2, 0] = I[3, 0] = 1 / np.sqrt(4)
S[1, 0] = S[1, 1] = S[0, 1] = S[0, 2] = 1 / np.sqrt(4)
J[0, 1] = J[1, 1] = J[2, 1] = J[2, 0] = 1 / np.sqrt(4)
Z[0, 0] = Z[0, 1] = Z[1, 1] = Z[1, 2] = 1 / np.sqrt(4)

train_images = [L, O, T, I, S, J, Z]

# ====================================================
#    Initialization of TensorFlow variables
# ====================================================

print('Initializing TensorFlow graph...')

# Initial standard deviation of parameters
sdev = 0.1

# Coherent state amplitude
alpha = 1.4

# Combinations of two-mode amplitudes corresponding to different final images
disps_alpha = tf.constant(
    [alpha, -alpha, alpha, -alpha, 1.0j * alpha, -1.0j * alpha, 1.0j * alpha]
)
disps_beta = tf.constant(
    [alpha, alpha, -alpha, -alpha, 1.0j * alpha, 1.0j * alpha, -1.0j * alpha]
)  # , 1.65615])

# Trainable weights of the quantum network.
with tf.name_scope('variables'):
    r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

    theta1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    phi1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

    theta2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    phi2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

    sqr1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    sqphi1 = tf.Variable(tf.random_normal(shape=[depth]))

    sqr2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    sqphi2 = tf.Variable(tf.random_normal(shape=[depth]))

    dr1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    dphi1 = tf.Variable(tf.random_normal(shape=[depth]))

    dr2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    dphi2 = tf.Variable(tf.random_normal(shape=[depth]))

    kappa1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    kappa2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

# List of all the weights
parameters = [
    r1,
    r2,
    theta1,
    phi1,
    theta2,
    phi2,
    sqr1,
    sqphi1,
    sqr2,
    sqphi2,
    dr1,
    dphi1,
    dr2,
    dphi2,
    kappa1,
    kappa2,
]

# ====================================================
#   Definition of the quantum neural network
# ====================================================

# Single quantum variational layer


def layer(l):
    with tf.name_scope('layer_{}'.format(l)):
        BSgate(theta1[l], phi1[l]) | (q[0], q[1])
        Rgate(r1[l]) | q[0]
        Sgate(tf.clip_by_value(sqr1[l], -sq_clip, sq_clip), sqphi1[l]) | q[0]
        Sgate(tf.clip_by_value(sqr2[l], -sq_clip, sq_clip), sqphi2[l]) | q[1]
        BSgate(theta2[l], phi2[l]) | (q[0], q[1])
        Rgate(r2[l]) | q[0]
        Dgate(tf.clip_by_value(dr1[l], -disp_clip, disp_clip), dphi1[l]) | q[0]
        Dgate(tf.clip_by_value(dr2[l], -disp_clip, disp_clip), dphi2[l]) | q[1]
        Kgate(tf.clip_by_value(kappa1[l], -kerr_clip, kerr_clip)) | q[0]
        Kgate(tf.clip_by_value(kappa2[l], -kerr_clip, kerr_clip)) | q[1]


# StrawberryFields quantum simulator
engine, q = sf.Engine(num_subsystems=modes)

# Definition of the CV quantum network
with engine:
    # State preparation
    Dgate(disps_alpha) | q[0]
    Dgate(disps_beta) | q[1]
    # Sequence of variational layers
    for i in range(depth):
        layer(i)

# Symbolic evaluation of the output state
state = engine.run('tf', cutoff_dim=cutoff, eval=False, batch_size=num_images)
ket = state.ket()

# Projection on the subspace of up to im_dim-1 photons for each mode.
ket_reduced = ket[:, :im_dim, :im_dim]
norm = tf.sqrt(tf.abs(tf.reduce_sum(tf.conj(ket_reduced) * ket_reduced, axis=[1, 2])))
norm_inv = tf.expand_dims(
    tf.expand_dims(tf.cast(1.0 / norm, dtype=tf.complex64), axis=[1]), axis=[1]
)
ket_processed = ket_reduced * norm_inv

# ====================================================
# 		 Definition of the loss function
# ====================================================

# Target images
data_states = tf.placeholder(tf.complex64, shape=[num_images, im_dim, im_dim])

# Overlaps with target images
overlaps = tf.abs(tf.reduce_sum(tf.conj(ket_processed) * data_states, axis=[1, 2])) ** 2

# Overlap cost function
overlap_cost = tf.reduce_mean((overlaps - 1) ** 2)

# State norm cost function
norm_cost = tf.reduce_sum(
    (tf.abs(tf.reduce_sum(tf.conj(ket) * ket, axis=[1, 2])) ** 2 - 1) ** 2
)

cost = overlap_cost + norm_weight * norm_cost

# ====================================================
# 	TensorBoard logging of cost functions and images
# ====================================================

tf.summary.scalar('Cost', cost)
tf.summary.scalar('Norm cost', norm_cost)
tf.summary.scalar('Overlap cost', overlap_cost)

# Output images with and without subspace projection.
images_out = tf.abs(ket_processed)
images_out_big = tf.abs(ket)
tf.summary.image('image_out', tf.expand_dims(images_out, axis=3), max_outputs=num_images)
tf.summary.image('image_out_big', tf.expand_dims(images_out_big, axis=3), max_outputs=num_images)

# TensorBoard writer and summary
writer = tf.summary.FileWriter(board_string)
merge = tf.summary.merge_all()


# ====================================================
# 					Training
# ====================================================

# Optimization algorithm (Adam optimizer)
optim = tf.train.AdamOptimizer()
training = optim.minimize(cost)

print('Graph building time: {:3f}'.format(time.time() - init_time))

# TensorFlow session
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    start_time = time.time()

    for i in range(reps):
        rep_time = time.time()
        # make an optimization step
        [_training] = session.run(
            [training], feed_dict={data_states: train_images}
        )

        if (i + 1) % partial_reps == 0:
            # evaluate tensors for saving and logging
            [summary, params_numpy,_images_out,_images_out_big] = session.run(
                [merge, tf.squeeze(parameters),images_out,images_out_big], feed_dict={data_states: train_images}
            )
            # save tensorboard data
            writer.add_summary(summary, i+1)
            
            # save trained weights
            os.makedirs(save_string,exist_ok=True)
            np.save(save_string+'trained_params.npy', params_numpy)
            
            # save output images as numpy arrays
            np.save(save_string+'images_out.npy',_images_out)
            np.save(save_string+'images_out_big.npy',_images_out_big)
            
            print(
                'Iteration: {:d} Single iteration time {:.3f}'.format(
                    i+1, time.time() - rep_time
                )
            )

print('Script completed. Total time: {:3f}'.format(time.time() - init_time))
