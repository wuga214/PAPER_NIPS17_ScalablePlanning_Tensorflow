import tensorflow as tf


class NAVI_BILINEAR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.batch_size = batch_size
        self.zero = tf.constant(0, shape=[batch_size, 2], dtype=tf.float32)
        self.four = tf.constant(4.0, dtype=tf.float32)
        self.one = tf.constant(1.0, shape=[batch_size], dtype=tf.float32)

    def MINMAZEBOUND(self):
        return self.min_maze_bound

    def MAXMAZEBOUND(self):
        return self.max_maze_bound

    def MINACTIONBOUND(self):
        return self.min_act_bound

    def MAXACTIONBOUND(self):
        return self.max_act_bound

    def GOAL(self):
        return self.goal

    def CENTER(self):
        return self.centre

    def Transition(self, states, actions):
        previous_state = states
        distance = tf.reduce_sum(tf.abs(states - self.CENTER()), 1)
        scalefactor = tf.where(tf.less(distance, self.four), distance / self.four, self.one)
        proposedLoc = previous_state + tf.matrix_transpose(scalefactor * tf.matrix_transpose(actions))
        new_states = tf.where(tf.logical_and(tf.less_equal(proposedLoc, self.MAXMAZEBOUND()),
                                             tf.greater_equal(proposedLoc, self.MINMAZEBOUND())),
                              proposedLoc,
                              tf.where(tf.greater(proposedLoc, self.MAXMAZEBOUND()),
                                       self.zero + self.MAXMAZEBOUND(),
                                       self.zero + self.MINMAZEBOUND())
                              )
        return new_states

    def Reward(self, states, actions):
        new_reward = -tf.reduce_sum(tf.abs(states - self.GOAL()), 1, keep_dims=True)
        return new_reward


class NAVI_NONLINEAR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.batch_size = batch_size
        self.zero = tf.constant(0, shape=[batch_size, 2], dtype=tf.float32)
        self.two = tf.constant(2.0, dtype=tf.float32)
        self.one = tf.constant(1.0, dtype=tf.float32)
        self.lessone = tf.constant(0.99, dtype=tf.float32)

    def MINMAZEBOUND(self):
        return self.min_maze_bound

    def MAXMAZEBOUND(self):
        return self.max_maze_bound

    def MINACTIONBOUND(self):
        return self.min_act_bound

    def MAXACTIONBOUND(self):
        return self.max_act_bound

    def GOAL(self):
        return self.goal

    def CENTER(self):
        return self.centre

    def Transition(self, states, actions):
        previous_state = states
        distance = tf.sqrt(tf.reduce_sum(tf.square(states - self.CENTER()), 1))
        scalefactor = self.two / (self.one + tf.exp(-self.two * distance)) - self.lessone
        proposedLoc = previous_state + tf.matrix_transpose(scalefactor * tf.matrix_transpose(actions))
        new_states = tf.where(tf.logical_and(tf.less_equal(proposedLoc, self.MAXMAZEBOUND()),
                                             tf.greater_equal(proposedLoc, self.MINMAZEBOUND())),
                               proposedLoc,
                               tf.where(tf.greater(proposedLoc, self.MAXMAZEBOUND()),
                                         self.zero + self.MAXMAZEBOUND(),
                                         self.zero + self.MINMAZEBOUND())
                               )
        return new_states

    def Reward(self, states, actions):
        new_reward = -tf.reduce_sum(tf.abs(states - self.GOAL()), 1, keep_dims=True)
        return new_reward


class NAVI_LINEAR(object):
    def __init__(self,
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.batch_size = batch_size
        self.zero = tf.constant(0, dtype=tf.float32)
        self.two = tf.constant(2.0, dtype=tf.float32)
        self.one = tf.constant(1.0, dtype=tf.float32)
        self.onedsix = tf.constant(1.6, dtype=tf.float32)
        self.onedtwo = tf.constant(1.2, dtype=tf.float32)
        self.dotnifi = tf.constant(0.95, dtype=tf.float32)
        self.doteight = tf.constant(0.8, dtype=tf.float32)
        self.dotsix = tf.constant(0.6, dtype=tf.float32)
        self.dotfour = tf.constant(0.4, dtype=tf.float32)
        self.dottwo = tf.constant(0.2, dtype=tf.float32)
        self.dotone = tf.constant(0.1, dtype=tf.float32)

    def MINMAZEBOUND(self, dim):
        return self.min_maze_bound

    def MAXMAZEBOUND(self, dim):
        return self.max_maze_bound

    def MINACTIONBOUND(self, dim):
        return self.min_act_bound

    def MAXACTIONBOUND(self, dim):
        return self.max_act_bound

    def GOAL(self):
        return self.goal

    def CENTER(self, dim):
        return self.centre

    def PENALTY(self):
        return self.penalty

    def _transition(self, dim, states_packed, actions_packed):

        states = tf.unstack(states_packed)
        actions = tf.unstack(actions_packed)

        previous_state = states[dim]

        # distance to centre Manhattan
        distance = self.zero
        for i in range(len(states)):
            distance += tf.abs(states[i] - self.CENTER(i))

        discountfactor = tf.cond(tf.logical_and(distance <= self.two, distance > self.onedsix),
                                 lambda: self.dotone,
                                 lambda: tf.cond(tf.logical_and(distance <= self.onedsix, distance > self.onedtwo),
                                                 lambda: self.dottwo,
                                                 lambda: tf.cond(
                                                     tf.logical_and(distance <= self.onedtwo, distance > self.doteight),
                                                     lambda: self.dotfour,
                                                     lambda: tf.cond(tf.logical_and(distance <= self.doteight,
                                                                                    distance > self.dotfour),
                                                                     lambda: self.dotsix,
                                                                     lambda: tf.cond(distance <= self.dotfour,
                                                                                     lambda: self.dotnifi,
                                                                                     lambda: self.zero
                                                                                     )

                                                                     )
                                                     )
                                                 )
                                 )

        # proposed location
        proposedLoc = tf.cond(tf.logical_and(actions[dim] >= self.zero, actions[dim] >= discountfactor),
                              lambda: previous_state + actions[dim] - discountfactor,
                              lambda: tf.cond(tf.logical_and(actions[dim] >= self.zero, actions[dim] < discountfactor),
                                              lambda: previous_state,
                                              lambda: tf.cond(tf.logical_and(actions[dim] < self.zero,
                                                                             -actions[dim] >= discountfactor),
                                                              lambda: previous_state + actions[dim] + discountfactor,
                                                              lambda: previous_state
                                                              )
                                              )
                              )

        # new state
        new_state = tf.cond(
            tf.logical_and(proposedLoc <= self.MAXMAZEBOUND(dim), proposedLoc >= self.MINMAZEBOUND(dim)), \
            lambda: proposedLoc,
            lambda: tf.cond(proposedLoc > self.MAXMAZEBOUND(dim), lambda: self.MAXMAZEBOUND(dim),
                            lambda: self.MINMAZEBOUND(dim))
            )

        return new_state

    # For single data point
    def _vector_trans(self, state_size, states_packed, actions_packed):
        new_states = []
        for i in range(state_size):
            new_states.append(self._transition(i, states_packed, actions_packed))
        return tf.stack(new_states)

    def Transition(self, states, actions):
        new_states = []
        batch_size, state_size = states.get_shape()
        states_list = tf.unstack(states)
        actions_list = tf.unstack(actions)
        for i in range(batch_size):
            new_states.append(self._vector_trans(state_size, states_list[i], actions_list[i]))
        return tf.stack(new_states)

    # def _reward(self, state_size, states_packed, actions_packed):
    #     reward = self.zero
    #     states = tf.unstack(states_packed)
    #     actions = tf.unstack(actions_packed)
    #
    #     for i in range(state_size):
    #         reward -= tf.abs(states[i] - self.GOAL(i))
    #     return tf.stack([reward])
    #
    # def Reward(self, states, actions):
    #     new_rewards = []
    #     batch_size, state_size = states.get_shape()
    #     states_list = tf.unstack(states)
    #     actions_list = tf.unstack(actions)
    #     for i in range(batch_size):
    #         new_rewards.append(self._reward(state_size, states_list[i], actions_list[i]))
    #     return tf.stack(new_rewards)
    def Reward(self, states, actions):
        new_reward = -tf.reduce_sum(tf.abs(states - self.GOAL()), 1, keep_dims=True)
        return new_reward


# class NAVI_LINEAR_REFACTOR(object):
#     def __init__(self,
#                  batch_size,
#                  default_settings):
#         self.__dict__.update(default_settings)
#         self.batch_size = batch_size
#         self.zero = tf.constant(0, dtype=tf.float32)
#         self.two = tf.constant(2.0, dtype=tf.float32)
#         self.one = tf.constant(1.0, dtype=tf.float32)
#         self.onedsix = tf.constant(1.6, dtype=tf.float32)
#         self.onedtwo = tf.constant(1.2, dtype=tf.float32)
#         self.dotnifi = tf.constant(0.95, dtype=tf.float32)
#         self.doteight = tf.constant(0.8, dtype=tf.float32)
#         self.dotsix = tf.constant(0.6, dtype=tf.float32)
#         self.dotfour = tf.constant(0.4, dtype=tf.float32)
#         self.dottwo = tf.constant(0.2, dtype=tf.float32)
#         self.dotone = tf.constant(0.1, dtype=tf.float32)
#
#     def MINMAZEBOUND(self, dim):
#         return self.min_maze_bound
#
#     def MAXMAZEBOUND(self, dim):
#         return self.max_maze_bound
#
#     def MINACTIONBOUND(self, dim):
#         return self.min_act_bound
#
#     def MAXACTIONBOUND(self, dim):
#         return self.max_act_bound
#
#     def GOAL(self, dim):
#         return self.goal
#
#     def CENTER(self, dim):
#         return self.centre
#
#     def PENALTY(self):
#         return self.penalty
#
#     def _transition(self, dim, states_packed, actions_packed):
#
#         states = tf.unstack(states_packed)
#         actions = tf.unstack(actions_packed)
#
#         previous_state = states[dim]
#
#         # distance to centre Manhattan
#         distance = self.zero
#         for i in range(len(states)):
#             distance += tf.abs(states[i] - self.CENTER(i))
#
#         discountfactor = tf.cond(tf.logical_and(distance <= self.two, distance > self.onedsix),
#                                  lambda: self.dotone,
#                                  lambda: tf.cond(tf.logical_and(distance <= self.onedsix, distance > self.onedtwo),
#                                                  lambda: self.dottwo,
#                                                  lambda: tf.cond(
#                                                      tf.logical_and(distance <= self.onedtwo, distance > self.doteight),
#                                                      lambda: self.dotfour,
#                                                      lambda: tf.cond(tf.logical_and(distance <= self.doteight,
#                                                                                     distance > self.dotfour),
#                                                                      lambda: self.dotsix,
#                                                                      lambda: tf.cond(distance <= self.dotfour,
#                                                                                      lambda: self.dotnifi,
#                                                                                      lambda: self.zero
#                                                                                      )
#
#                                                                      )
#                                                      )
#                                                  )
#                                  )
#
#         # proposed location
#         proposedLoc = tf.cond(tf.logical_and(actions[dim] >= self.zero, actions[dim] >= discountfactor),
#                               lambda: previous_state + actions[dim] - discountfactor,
#                               lambda: tf.cond(tf.logical_and(actions[dim] >= self.zero, actions[dim] < discountfactor),
#                                               lambda: previous_state,
#                                               lambda: tf.cond(tf.logical_and(actions[dim] < self.zero,
#                                                                              -actions[dim] >= discountfactor),
#                                                               lambda: previous_state + actions[dim] + discountfactor,
#                                                               lambda: previous_state
#                                                               )
#                                               )
#                               )
#
#         # new state
#         new_state = tf.cond(
#             tf.logical_and(proposedLoc <= self.MAXMAZEBOUND(dim), proposedLoc >= self.MINMAZEBOUND(dim)), \
#             lambda: proposedLoc,
#             lambda: tf.cond(proposedLoc > self.MAXMAZEBOUND(dim), lambda: self.MAXMAZEBOUND(dim),
#                             lambda: self.MINMAZEBOUND(dim))
#             )
#
#         return new_state
#
#     # For single data point
#     def _vector_trans(self, state_size, states_packed, actions_packed):
#         new_states = []
#         for i in range(state_size):
#             new_states.append(self._transition(i, states_packed, actions_packed))
#         return tf.stack(new_states)
#
#     def Transition(self, states, actions):
#         new_states = []
#         batch_size, state_size = states.get_shape()
#         states_list = tf.unstack(states)
#         actions_list = tf.unstack(actions)
#         for i in range(batch_size):
#             new_states.append(self._vector_trans(state_size, states_list[i], actions_list[i]))
#         return tf.stack(new_states)
#
#     def _reward(self, state_size, states_packed, actions_packed):
#         reward = self.zero
#         states = tf.unstack(states_packed)
#         actions = tf.unstack(actions_packed)
#
#         for i in range(state_size):
#             reward -= tf.abs(states[i] - self.GOAL(i))
#         return tf.stack([reward])
#
#     def Reward(self, states, actions):
#         new_rewards = []
#         batch_size, state_size = states.get_shape()
#         states_list = tf.unstack(states)
#         actions_list = tf.unstack(actions)
#         for i in range(batch_size):
#             new_rewards.append(self._reward(state_size, states_list[i], actions_list[i]))
#         return tf.stack(new_rewards)
