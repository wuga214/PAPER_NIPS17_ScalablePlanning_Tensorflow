import tensorflow as tf
import numpy as np


class HVAC(object):
    def __init__(self,
                 adj_outside,  # Adjacent to outside
                 adj_hall,  # Adjacent to hall
                 adj,  # Adjacent between rooms
                 rooms,  # Room names
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.rooms = rooms
        self.batch_size = batch_size
        self.room_size = len(rooms)
        self.zero = tf.constant(0, shape=[self.batch_size ,self.room_size], dtype=tf.float32)
        self._init_ADJ_Matrix(adj)
        self._init_ADJOUT_MATRIX(adj_outside)
        self._init_ADJHALL_MATRIX(adj_hall)

    def _init_ADJ_Matrix(self ,adj):
        np_adj = np.zeros((self.room_size, self.room_size))
        for i in adj:
            m=self.rooms. index(i[0])
            n=self.rooms. index(i[1])
            np_adj[m, n] = 1
            np_adj[n, m] = 1
        self.adj = tf.constant(np_adj, dtype=tf. float32)
        # print('self.adj shape:{0}'.format(self.adj.get_shape()))

    def _init_ADJOUT_MATRIX(self, adj_outside):
        np_adj_outside = np.zeros((self.room_size,))
        for i in adj_outside:
            m=self.rooms. index(i)
            np_adj_outside[m] = 1
        self.adj_outside = tf.constant(np_adj_outside,dtype=tf. float32)

    def _init_ADJHALL_MATRIX(self, adj_hall):
        np_adj_hall = np.zeros((self.room_size,))
        for i in adj_hall:
            m=self.rooms. index(i)
            np_adj_hall[m] = 1
        self.adj_hall = tf.constant(np_adj_hall,dtype=tf. float32)

    def ADJ(self):
        return self.adj

    def ADJ_OUTSIDE(self):
        return self.adj_outside

    def ADJ_HALL(self):
        return self.adj_hall

    def R_OUTSIDE(self):
        return self.outside_resist

    def R_HALL(self):
        return self.hall_resist

    def R_WALL(self):
        return self.wall_resist

    def CAP(self):
        return self.cap

    def CAP_AIR(self):
        return self.cap_air

    def COST_AIR(self):
        return self.cost_air

    def TIME_DELTA(self):
        return self.time_delta

    def TEMP_AIR(self):
        return self.temp_air

    def TEMP_UP(self):
        return self.temp_up

    def TEMP_LOW(self):
        return self.temp_low

    def TEMP_OUTSIDE(self):
        return self.temp_outside

    def TEMP_HALL(self):
        return self.temp_hall

    def PENALTY(self):
        return self.penalty

    def AIR_MAX(self):
        return self.air_max

    def ZERO(self):
        return self.zero

    def Transition(self, states, actions):
        previous_state = states
        # print('state shape:{0}'.format(states.get_shape()))
        heating_info = actions*self.CAP_AIR() * (self.TEMP_AIR()-previous_state)
        neighbor_info = (tf.transpose(tf.matmul(self.ADJ(), tf.transpose(states)))
                         - previous_state
                         *tf.reduce_sum(self.ADJ(), 1))/self.R_WALL()
        outside_info = (self. TEMP_OUTSIDE() - previous_state)*self. ADJ_OUTSIDE()/self.R_OUTSIDE()
        hall_info = (self. TEMP_HALL() - previous_state)*self. ADJ_HALL()/self.R_HALL()
        # print('neighbor_info shape:{0}'.format(neighbor_info.get_shape()))
        # print('hall_info shape:{0}'.format(hall_info.get_shape()))
        new_state = previous_state+self. TIME_DELTA()/self. CAP()*(heating_info +
                                                                   neighbor_info + outside_info + hall_info)
        return new_state

    def Reward(self, states, actions):
        # batch_size,state_size = states.get_shape()
        # break_penalty = tf.select(tf.logical_or(tf.less(states,self.TEMP_LOW()),\
        #                                        tf.greater(states,self.TEMP_UP())),self.PENALTY()+self.ZERO(),self.ZERO())
        dist_penalty = tf.abs(((self.TEMP_UP()+self. TEMP_LOW())/tf. constant(2.0, dtype=tf.float32))-states)
        ener_penalty = actions*self. COST_AIR()
        new_rewards = -tf.reduce_sum(tf.constant(10.0, tf.float32) * dist_penalty + ener_penalty, 1, keep_dims=True)
        return new_rewards

