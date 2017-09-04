import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np
from cells.hvac import HVACCell
from instances.hvac import HVAC_60
from tqdm import tqdm


DEFAULT_SETTINGS = {
    "cap": 80,
    "outside_resist": 2.0,
    "hall_resist": 1.3,
    "wall_resist": 1.1,
    "cap_air": 1.006,
    "cost_air": 1.0,
    "time_delta": 1.0,
    "temp_air": 40.0,
    "temp_up": 23.5,
    "temp_low": 20.0,
    "temp_outside": 6.0,
    "temp_hall": 10.0,
    "penalty": 1000.0,
    "air_max": 10.0
   }


class HVACOptimizer(object):
    def __init__(self,
                 action,  # Actions
                 num_step,  # Number of RNN step, this is a fixed step RNN sequence, 12 for navigation
                 batch_size,
                 domain,
                 instance,
                 sess,
                 learning_rate=0.1):
        self.action = action
        # print(self.action)
        self.num_step = num_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.previous_output = np.zeros((batch_size, num_step))
        self.weights = np.ones((batch_size, num_step, 1))
        self._p_create_rnn_graph(instance, domain)
        self._p_create_loss()
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def _p_create_rnn_graph(self, instance, domain):
        cell = HVACCell(domain, instance['adj_outside'], instance['adj_hall'], instance['adj'], instance['rooms'],
                        self.batch_size, DEFAULT_SETTINGS)
        initial_state = cell.zero_state(self.action.get_shape()[0], dtype=tf.float32) \
                        + tf.constant(10, dtype=tf.float32)
        # +tf.constant([RandomInitialandWriteFile(rooms)],dtype=tf.float32)
        # print('action batch size:{0}'.format(array_ops.shape(self.action)[0]))
        # print('Initial_state shape:{0}'.format(initial_state))
        rnn_outputs, state = tf.nn.dynamic_rnn(cell, self.action, dtype=tf.float32, initial_state=initial_state)
        # need output intermediate states as well
        concated = tf.concat(axis=0, values=rnn_outputs)
        something_unpacked = tf.unstack(concated, axis=2)
        self.outputs = tf.reshape(something_unpacked[0], [-1, self.num_step, 1])
        self.intern_states = tf.stack(something_unpacked[1:61], axis=2)
        self.last_state = state
        self.pred = tf.reduce_sum(self.outputs, 1)
        self.average_pred = tf.reduce_mean(self.pred)
        # print("self.pred:{0}".format(self.pred))

    def _p_create_loss(self):
        # print("MSE-loss")
        objective = tf.reduce_mean(tf.square(self.pred))
        self.loss = objective
        # print(self.loss.get_shape())
        # self.loss = -objective
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.action])

    def Optimize(self, epoch=100):
        new_loss = self.sess.run([self.average_pred])
        # print('Loss in epoch {0}: {1}'.format("Initial", new_loss))
        for epoch in tqdm(xrange(epoch)):
            training = self.sess.run([self.optimizer])
            self.sess.run(tf.assign(self.action, tf.clip_by_value(self.action, 0, 10)))
            if True:
                new_loss = self.sess.run([self.average_pred])
                # print('Loss in epoch {0}: {1}'.format(epoch, new_loss))
        minimum_costs_id = self.sess.run(tf.argmax(self.pred, 0))
        # print(minimum_costs_id)
        # print('Optimal Action Squence:{0}'.format(self.sess.run(self.action)[minimum_costs_id[0]]))
        action = self.sess.run(self.action)[minimum_costs_id[0]]
        # np.savetxt("HVAC_ACTION.csv", action, delimiter=",", fmt='%2.5f')
        pred_list = self.sess.run(self.pred)
        pred_list = np.sort(pred_list.flatten())[::-1]
        pred_list = pred_list[:10]
        pred_mean = np.mean(pred_list)
        pred_std = np.std(pred_list)
        # print('Best Cost: {0}'.format(pred_list[0]))
        # print('Sorted Costs:{0}'.format(pred_list))
        # print('MEAN: {0}, STD:{1}'.format(pred_mean, pred_std))
        # print('The last state:{0}'.format(self.sess.run(self.last_state)))
        # print('Rewards each time step:{0}'.format(self.sess.run(self.outputs)))
        # reward = self.sess.run(self.outputs)[minimum_costs_id[0]]
        # np.savetxt("HVAC_REWARD.csv", reward, delimiter=",", fmt='%7.5f')
        # print('Intermediate states:{0}'.format(self.sess.run(self.intern_states)[minimum_costs_id[0]]))
        # interm = self.sess.run(self.intern_states)[minimum_costs_id[0]]
        # np.savetxt("HVAC_INTERM.csv", interm, delimiter=",", fmt='%2.5f')
        # print 'END'
        return pred_mean, pred_std
