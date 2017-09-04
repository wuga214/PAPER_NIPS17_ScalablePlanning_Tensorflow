import os
import gc
import json
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from optimizer.reservoir import ReservoirOptimizer
from domains.reservoir import RESERVOIR_NONLINEAR
from instances.reservoir import RESERVOIR_20

with tf.Session() as sess:
    initial_a = tf.truncated_normal(shape=[100, 120, 20], mean=0.0, stddev=0.5).eval(session=sess)
    a = tf.Variable(initial_a, name="action")
    rnn_inst = ReservoirOptimizer(a, 120, 100, RESERVOIR_NONLINEAR, RESERVOIR_20, sess)
    rnn_inst.Optimize(timing=True)

gc.collect()
tf.reset_default_graph()

with tf.Session() as sess:
    initial_a = tf.truncated_normal(shape=[100, 60, 20], mean=0.0, stddev=0.5).eval(session=sess)
    a = tf.Variable(initial_a, name="action")
    rnn_inst = ReservoirOptimizer(a, 60, 100, RESERVOIR_NONLINEAR, RESERVOIR_20, sess)
    rnn_inst.Optimize(timing=True)
