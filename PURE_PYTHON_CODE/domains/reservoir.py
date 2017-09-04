import tensorflow as tf
import numpy as np


class RESERVOIR_NONLINEAR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.batch_size = batch_size
        self.reservoirs = default_settings['reservoirs']
        self.reservoir_num = len(default_settings['reservoirs'])
        self.biggestmaxcap = tf.constant(default_settings["biggestmaxcap"], dtype=tf.float32)
        self.zero = tf.constant(0, shape=[self.batch_size, self.reservoir_num], dtype=tf.float32)
        self._high_bounds(default_settings["high_bound"])
        self._low_bounds(default_settings["low_bound"])
        self._rains(default_settings["rain"])
        self._max_cap(default_settings["max_cap"])
        self._downstream(default_settings["downstream"])
        self._downtosea(default_settings["downtosea"])

    def _max_cap(self, max_cap_list):
        self.max_cap = tf.constant(max_cap_list, dtype=tf.float32)

    def _high_bounds(self, high_bound_list):
        self.high_bound = tf.constant(high_bound_list, dtype=tf.float32)

    def _low_bounds(self, low_bound_list):
        self.low_bound = tf.constant(low_bound_list, dtype=tf.float32)

    def _rains(self, rain_list):
        self.rain = tf.constant(rain_list, dtype=tf.float32)

    def _downstream(self, downstream):
        np_downstream = np.zeros((self.reservoir_num, self.reservoir_num))
        for i in downstream:
            m = self.reservoirs.index(i[0])
            n = self.reservoirs.index(i[1])
            np_downstream[m, n] = 1
        self.downstream = tf.constant(np_downstream, dtype=tf.float32)

    def _downtosea(self, downtosea):
        np_downtosea = np.zeros((self.reservoir_num,))
        for i in downtosea:
            m = self.reservoirs.index(i)
            np_downtosea[m] = 1
        self.downtosea = tf.constant(np_downtosea, dtype=tf.float32)

    def MAXCAP(self):
        return self.max_cap

    def HIGH_BOUND(self):
        return self.high_bound

    def LOW_BOUND(self):
        return self.low_bound

    def RAIN(self):
        return self.rain

    def DOWNSTREAM(self):
        return self.downstream

    def DOWNTOSEA(self):
        return self.downtosea

    def BIGGESTMAXCAP(self):
        return self.biggestmaxcap

    def Transition(self, states, actions):
        previous_state = states
        vaporated = 0.5 * tf.sin(previous_state / self.BIGGESTMAXCAP()) * previous_state
        upstreamflow = tf.transpose(tf.matmul(tf.transpose(self.DOWNSTREAM()), tf.transpose(actions)))
        new_state = previous_state + self.RAIN() - vaporated - actions + upstreamflow
        return new_state

    # Reward for Reservoir is computed on 'Next State'
    def Reward(self, states):
        new_rewards = tf.where(
            tf.logical_and(tf.greater_equal(states, self.LOW_BOUND()), tf.less_equal(states, self.HIGH_BOUND())),
            self.zero,
            tf.where(tf.less(states, self.LOW_BOUND()),
                     -5 * (self.LOW_BOUND() - states),
                     -100 * (states - self.HIGH_BOUND()))
            )
        new_rewards += tf.abs(((self.HIGH_BOUND() + self.LOW_BOUND()) / 2.0) - states) * (-0.1)
        return tf.reduce_sum(new_rewards, 1, keep_dims=True)


class RESERVOIR_LINEAR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.batch_size = batch_size
        self.reservoirs = default_settings['reservoirs']
        self.reservoir_num = len(default_settings['reservoirs'])
        self.biggestmaxcap = tf.constant(default_settings["biggestmaxcap"], dtype=tf.float32)
        self.zero = tf.constant(0, shape=[self.batch_size, self.reservoir_num], dtype=tf.float32)
        self._high_bounds(default_settings["high_bound"])
        self._low_bounds(default_settings["low_bound"])
        self._rains(default_settings["rain"])
        self._max_cap(default_settings["max_cap"])
        self._downstream(default_settings["downstream"])
        self._downtosea(default_settings["downtosea"])

    def _max_cap(self, max_cap_list):
        self.max_cap = tf.constant(max_cap_list, dtype=tf.float32)

    def _high_bounds(self, high_bound_list):
        self.high_bound = tf.constant(high_bound_list, dtype=tf.float32)

    def _low_bounds(self, low_bound_list):
        self.low_bound = tf.constant(low_bound_list, dtype=tf.float32)

    def _rains(self, rain_list):
        self.rain = tf.constant(rain_list, dtype=tf.float32)

    def _downstream(self, downstream):
        np_downstream = np.zeros((self.reservoir_num, self.reservoir_num))
        for i in downstream:
            m = self.reservoirs.index(i[0])
            n = self.reservoirs.index(i[1])
            np_downstream[m, n] = 1
        self.downstream = tf.constant(np_downstream, dtype=tf.float32)

    def _downtosea(self, downtosea):
        np_downtosea = np.zeros((self.reservoir_num,))
        for i in downtosea:
            m = self.reservoirs.index(i)
            np_downtosea[m] = 1
        self.downtosea = tf.constant(np_downtosea, dtype=tf.float32)

    def MAXCAP(self):
        return self.max_cap

    def HIGH_BOUND(self):
        return self.high_bound

    def LOW_BOUND(self):
        return self.low_bound

    def RAIN(self):
        return self.rain

    def DOWNSTREAM(self):
        return self.downstream

    def DOWNTOSEA(self):
        return self.downtosea

    def BIGGESTMAXCAP(self):
        return self.biggestmaxcap

    def Transition(self, states, actions):
        previous_state = states
        vaporated = 0.1 * previous_state
        upstreamflow = tf.transpose(tf.matmul(tf.transpose(self.DOWNSTREAM()), tf.transpose(actions)))
        new_state = previous_state + self.RAIN() - vaporated - actions + upstreamflow
        return new_state

    # Reward for Reservoir is computed on 'Next State'
    def Reward(self, states):
        new_rewards = tf.where(
            tf.logical_and(tf.greater_equal(states, self.LOW_BOUND()), tf.less_equal(states, self.HIGH_BOUND())),
            self.zero,
            tf.where(tf.less(states, self.LOW_BOUND()),
                     -5 * (self.LOW_BOUND() - states),
                     -100 * (states - self.HIGH_BOUND()))
            )
        new_rewards += tf.abs(((self.HIGH_BOUND() + self.LOW_BOUND()) / 2.0) - states) * (-0.1)
        return tf.reduce_sum(new_rewards, 1, keep_dims=True)
