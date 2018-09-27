import strawberryfields as sf
from strawberryfields.ops import *
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import pi
import os

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Computer Modern Roman']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tic = time.time()


# Initialization values
cutoff = 10
interval = 1
batch_size = 50
depth = 6
disp_clip = 100
sq_clip = 50
kerr_clip = 50
reps = 1000
regularization = 0.0
reg_variance = 0.0


# this is the function that we want to fit; we use it to generate data points
def f(x, eps=0.0):
    # return np.abs(x) + eps * np.random.normal(size=x.shape)
    # return np.sin(x*pi)/(pi*x) + eps * np.random.normal(size=x.shape)
    return 1.0*(np.sin(1.0 * x * np.pi) + eps * np.random.normal(size=x.shape))
    # return np.exp(x) + eps * np.random.normal(size=x.shape)
    # return np.tanh(4*x) + eps * np.random.normal(size=x.shape)
    # return x**3 + eps * np.random.normal(size=x.shape)


train_data = np.load('sine_train_data.npy')
test_data = np.load('sine_test_data.npy')
data_y = np.load('sine_outputs.npy')


# Random initialization of gate parameters
sdev = 0.05

with tf.name_scope('variables'):
    d_r = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    d_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    r1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    sq_r = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    sq_phi = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    r2 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))
    kappa1 = tf.Variable(tf.random_normal(shape=[depth], stddev=sdev))

engine, qMode = sf.Engine(1)


# Gate layer: D-R1-S-R2-K
def layer(i):
    with tf.name_scope('layer_{}'.format(i)):
        Dgate(tf.clip_by_value(d_r[i], -disp_clip, disp_clip), d_phi[i]) | qMode[0]
        Rgate(r1[i]) | qMode[0]
        Sgate(tf.clip_by_value(sq_r[i], -sq_clip, sq_clip), sq_phi[i]) | qMode[0]
        Rgate(r2[i]) | qMode[0]
        Kgate(tf.clip_by_value(kappa1[i], -kerr_clip, kerr_clip)) | qMode[0]


input_data = tf.placeholder(tf.float32, shape=[batch_size])
with engine:
    Dgate(input_data) | qMode[0]
    for k in range(depth):
        layer(k)


state = engine.run('tf', cutoff_dim=cutoff, eval=False, batch_size=batch_size)
ket = state.ket()
mean_x, svd_x = state.quad_expectation(0)
errors_y = tf.sqrt(svd_x)

output_data = tf.placeholder(tf.float32, shape=[batch_size])
mse = tf.reduce_mean(tf.abs(mean_x - output_data) ** 2)
var = tf.reduce_mean(errors_y)

state_norm = tf.abs(tf.reduce_mean(state.trace()))
cost = mse + regularization * (tf.abs(state_norm - 1) ** 2) + reg_variance*var
tf.summary.scalar('cost', cost)

optimiser = tf.train.AdamOptimizer()
min_op = optimiser.minimize(cost)

session = tf.Session()
session.run(tf.global_variables_initializer())

print('Beginning optimization')

mse_vals = []
error_vals = []


for i in range(reps+1):
    # data_x = np.random.uniform(-interval, interval, size=batch_size)
    # data_y = f(data_x)

    mse_, predictions, errors, mean_error, ket_norm, _ = session.run([mse, mean_x, errors_y, var, state_norm, min_op],
                                       feed_dict={input_data: train_data, output_data: data_y})

    mse_vals.append(mse_)
    error_vals.append(mean_error)

    if i % 100 == 0:
        print(i, mse_)

toc = time.time()

test_predictions = session.run(mean_x, feed_dict={input_data: test_data})

np.save('sine_test_predictions', test_predictions)


print("Elapsed time is {} seconds".format(np.round(toc - tic)))

plt.figure()
x = np.linspace(-interval, interval, 200)
plt.plot(x, f(x), color='#3f9b0b', zorder=1, linewidth=2)
# plt.errorbar(data_x, predictions, yerr=errors, fmt='o', color='#fb2943')
plt.scatter(train_data, data_y, color='#fb2943', marker='o', zorder=2, s=75)
plt.scatter(test_data, test_predictions, color='#0165fc', marker='x', zorder=3, s=75)
plt.xlabel('Input', fontsize=18)
plt.ylabel('Output', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
# plt.savefig('x3.pdf', format='pdf', bbox_inches='tight')
plt.show()




