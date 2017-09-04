import tensorflow as tf


class RESERVOIRCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, domain, batch_size, default_settings):
        self._num_state_units = len(default_settings["reservoirs"])
        self._num_reward_units = self._num_state_units + 1
        self.reservoir = domain(batch_size, default_settings)

    @property
    def state_size(self):
        return self._num_state_units

    @property
    def output_size(self):
        return self._num_reward_units

    def __call__(self, inputs, state, scope=None):
        next_state = self.reservoir.Transition(state, inputs)
        reward = self.reservoir.Reward(next_state)
        return tf.concat(axis=1, values=[reward, next_state]), next_state
