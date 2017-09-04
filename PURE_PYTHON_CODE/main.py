import os
import gc
import json
import tensorflow as tf
from optimizer.hvac import HVACOptimizer
from optimizer.nav import NAVOptimizer
from optimizer.reservoir import ReservoirOptimizer
from domains.hvac import HVAC
from domains.nav import NAVI_NONLINEAR, NAVI_BILINEAR, NAVI_LINEAR
from domains.reservoir import RESERVOIR_LINEAR, RESERVOIR_NONLINEAR

from instances.nav import NAV_30
from instances.reservoir import RESERVOIR_20
from instances.hvac import HVAC_60

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# sess = tf.Session()
# initial_a = tf.truncated_normal(shape=[100, 96, 60], mean=5.0, stddev=1.0).eval(session=sess)
# a = tf.Variable(initial_a, name="action")
# rnn_inst = HVACOptimizer(a, 96, 100, HVAC, HVAC_60, sess)
# rnn_inst.Optimize(1000)

# sess = tf.Session()
# initial_a = tf.truncated_normal(shape=[100, 120, 2], mean=0.0, stddev=0.005).eval(session=sess)
# a = tf.Variable(initial_a, name="action")
# rnn_inst = NAVOptimizer(a, 120, 100, NAVI_LINEAR, NAV_30, sess)
# rnn_inst.Optimize(1000)

# sess = tf.Session()
# initial_a = tf.truncated_normal(shape=[100, 120, 20], mean=0.0, stddev=0.5).eval(session=sess)
# a = tf.Variable(initial_a, name="action")
# rnn_inst = ReservoirOptimizer(a, 120, 100, RESERVOIR_NONLINEAR, RESERVOIR_60, sess)
# rnn_inst.Optimize(300)
#
from experiments.configuration import CONFIGURATIONS

Data = {}
for config in CONFIGURATIONS:
    for index, step in enumerate(config['step']):
        gc.collect()
        tf.reset_default_graph()
        with tf.Session() as sess:
            name = (config['optimizer'].__name__
                    + ' ' + config['domain'].__name__
                    + ' ' + config['instance'][index]['name']
                    + ' Planning Step:' + str(step))
            print name
            initial_a = tf.truncated_normal(shape=[config['batch'], step, config['dimension']],
                                            mean=config['initial_mean'],
                                            stddev=config['initial_std'])
            a = tf.Variable(initial_a, name="action")
            rnn_inst = config['optimizer'](a, step, config['batch'], config['domain'], config['instance'][index], sess=sess)
            mean, std = rnn_inst.Optimize(config['epoch'])
            print 'mean: {0}, std: {1}'.format(mean, std)
            Data[name] = unicode([mean, std])
with open('result.json', 'w') as fp:
    json.dump(Data, fp)
