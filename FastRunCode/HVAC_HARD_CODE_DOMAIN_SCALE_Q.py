
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import pandas as pd
#Functional coding
import random
import functools
from functools import partial
from tensorflow.python.ops import array_ops 
from scipy.stats import truncnorm


# In[2]:

'a' in ['a','b','c']

# In[4]:

#Given local path, find full path
def PathFinder(path):
    #python 2
    #script_dir = os.path.dirname('__file__')
    #fullpath = os.path.join(script_dir,path)
    #python 3
    fullpath=os.path.abspath(path)
    print(fullpath)
    return fullpath

#Read Data for Deep Learning
def ReadData(path):
    fullpath=PathFinder(path)
    return pd.read_csv(fullpath, sep=',', header=0)

def RandomInitialandWriteFile(rooms):
    num_rooms = len(rooms)
    initial_state=[truncnorm.rvs(-5/3,5/3,loc=15, scale=2.5) for _ in range(60)]
    for i,room in enumerate(rooms):
        print('TEMP(r{}) = {:2.6f};'.format(room,initial_state[i]))
    return initial_state


# In[6]:

default_settings = {                
    "cap": tf.constant(80.0,dtype=tf.float32), 
    "outside_resist" : tf.constant(2.0,dtype=tf.float32),
    "hall_resist" : tf.constant(1.3,dtype=tf.float32),
    "wall_resist" : tf.constant(1.1,dtype=tf.float32),
    "cap_air" : tf.constant(1.006,dtype=tf.float32), 
    "cost_air" : tf.constant(1.0,dtype=tf.float32), 
    "time_delta" : tf.constant(1.0,dtype=tf.float32),
    "temp_air" : tf.constant(40.0,dtype=tf.float32),
    "temp_up" : tf.constant(23.5,dtype=tf.float32),
    "temp_low" : tf.constant(20.0,dtype=tf.float32),
    "temp_outside" : tf.constant(6.0,dtype=tf.float32),
    "temp_hall" : tf.constant(10.0,dtype=tf.float32),
    "penalty" : tf.constant(20000.0,dtype=tf.float32),
    "air_max" : tf.constant(10.0,dtype=tf.float32)
   }


# In[7]:

class HVAC(object):
    def __init__(self, 
                 adj_outside, #Adjacent to outside 
                 adj_hall, #Adjacent to hall
                 adj, #Adjacent between rooms
                 rooms, #Room names
                 default_settings):
        self.__dict__.update(default_settings)
        self.adj_outside = adj_outside
        self.adj_hall = adj_hall
        self.adj = adj
        self.rooms = rooms
        self.zero = tf.constant(0,dtype=tf.float32)
        
    def ADJ(self, space1, space2):
        for pair in self.adj:
            if space1 in pair and space2 in pair:
                return True
        return False
                 
    def ADJ_OUTSIDE(self,  space):
        if space in self.adj_outside:
            return True
        else:
            return False
            
    def ADJ_HALL(self, space):
        if space in self.adj_hall:
            return True
        else:
            return False  
        
    def R_OUTSIDE(self, space):
        return self.outside_resist
    
    def R_HALL(self, space):
        return self.hall_resist
    
    def R_WALL(self, space1, space2):
        return self.wall_resist
        
    def CAP(self, space):
        return self.cap
    
    def CAP_AIR(self):
        return self.cap_air
    
    def COST_AIR(self):
        return self.cost_air
    
    def TIME_DELTA(self):
        return self.time_delta
    
    def TEMP_AIR(self):
        return self.temp_air
    
    def TEMP_UP(self, space):
        return self.temp_up
    
    def TEMP_LOW(self, space):
        return self.temp_low
    
    def TEMP_OUTSIDE(self, space):
        return self.temp_outside
    
    def TEMP_HALL(self, space):
        return self.temp_hall
    
    def PENALTY(self):
        return self.penalty
    
    def AIR_MAX(self, space):
        return self.air_max
    
    # Single state function, need map to matrix later
    def _transition(self, space, states, actions):
        
        previous_state = states[space]
        room_id = self.rooms[space]
        heating_info = actions[space]*self.CAP_AIR()*(self.TEMP_AIR()-previous_state)
        neighbor_info = self.zero
        for i,p in enumerate(self.rooms):
            if self.ADJ(room_id,p):
                neighbor_info += (states[i]-previous_state)/self.R_WALL(room_id,p)
        outside_info = self.zero
        if self.ADJ_OUTSIDE(room_id):
            outside_info=(self.TEMP_OUTSIDE(room_id)-previous_state)/self.R_OUTSIDE(room_id)
        wall_info = self.zero
        if self.ADJ_HALL(room_id):
            wall_info=(self.TEMP_HALL(room_id)-previous_state)/self.R_HALL(room_id)
            
        new_state = previous_state + self.TIME_DELTA()/self.CAP(room_id)*(heating_info + neighbor_info + outside_info + wall_info)
        return new_state
    
    # For single data point
    def _vector_trans(self, state_size, states_packed, actions_packed):
        new_states = []
        states = tf.unpack(states_packed)
        actions = tf.unpack(actions_packed)
        for i in range(state_size):
            new_states.append(self._transition(i,states,actions))
        return tf.pack(new_states)
    
    def Transition(self, states, actions):
        new_states = []
        batch_size,state_size = states.get_shape()
        states_list = tf.unpack(states)
        actions_list = tf.unpack(actions)
        for i in range(batch_size):
            new_states.append(self._vector_trans(state_size,states_list[i],actions_list[i]))
        return tf.pack(new_states)
    
    # For single data point
    def _reward(self, state_size, states_packed, actions_packed):
        reward = self.zero
        states = tf.unpack(states_packed)
        actions = tf.unpack(actions_packed)
        
        #For each room
        for i in range(state_size):
            room_id = self.rooms[i]
            #Penalty for breaking upper or lower bound constraints
            break_penalty = tf.cond(tf.logical_or(states[i] <self.TEMP_LOW(room_id), states[i] > self.TEMP_UP(room_id)), lambda: self.PENALTY(), lambda: self.zero)
                
            #Penalty for distance to centre(no bug)
            dist_penalty = tf.abs(((self.TEMP_UP(room_id)+self.TEMP_LOW(room_id))/tf.constant(2.0, dtype=tf.float32))-states[i])
            
            #Penalty for energy cost
            ener_penalty = tf.square(actions[i])*self.COST_AIR()
            
            #break_penalty+tf.constant(10.0, tf.float32)*dist_penalty
            reward -= (break_penalty+tf.constant(10.0, tf.float32)*dist_penalty+ener_penalty)
            
        return tf.pack([reward])
            
    def Reward(self, states,actions):
        new_rewards = []
        batch_size,state_size = states.get_shape()
        states_list = tf.unpack(states)
        actions_list = tf.unpack(actions)
        for i in range(batch_size):
            new_rewards.append(self._reward(state_size,states_list[i],actions_list[i]))
        return tf.pack(new_rewards)            


# In[8]:

adj_hall = [101,102,103,106,107,107,110,               201,202,203,206,207,207,210,               301,302,303,306,307,307,310,               401,402,403,406,407,407,410,               501,502,503,506,507,507,510]
adj_outside = [101,102,103,104,105,106,108,110,111,112,              201,202,203,204,205,206,208,210,211,212,              301,302,303,304,305,306,308,310,311,312,              401,402,403,404,405,406,408,410,411,412,              501,502,503,504,505,506,508,510,511,512]
adj = [[101,102],[102,103],[103,104],[104,105],[106,107],[107,108],[107,109],[108,109],[110,111],[111,112],       [201,202],[202,203],[203,204],[204,205],[206,207],[207,208],[207,209],[208,209],[210,211],[211,212],       [301,302],[302,303],[303,304],[304,305],[306,307],[307,308],[307,309],[308,309],[310,311],[311,312],       [401,402],[402,403],[403,404],[404,405],[406,407],[407,408],[407,409],[408,409],[410,411],[411,412],       [501,502],[502,503],[503,504],[504,505],[506,507],[507,508],[507,509],[508,509],[510,511],[511,512],       [101,201],[102,202],[103,203],[104,204],[105,205],[106,206],[107,207],[108,208],[109,209],[110,201],       [111,211],[112,212],[201,301],[202,302],[203,303],[204,304],[205,305],[206,306],[207,307],[208,308],       [209,309],[210,301],[211,311],[212,312],[301,401],[302,402],[303,403],[304,404],[305,405],[306,406],       [307,407],[308,408],[309,409],[310,401],[311,411],[312,412],[401,501],[402,502],[403,503],[404,504],       [405,505],[406,506],[407,507],[408,508],[409,509],[410,501],[411,511],[412,512]]
rooms = list(range(101,113))+list(range(201,213))+list(range(301,313))+list(range(401,413))+list(range(501,513))


# In[10]:

# States
states = tf.placeholder(tf.float32,[10, 60],name="States")

# Actions
actions = tf.placeholder(tf.float32,[10, 60],name="Actions")


# In[21]:

class HVACCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, adj_outside,adj_hall,adj,rooms,default_settings):
        self._num_state_units = len(rooms)
        self._num_reward_units = 1+len(rooms)
        self.hvac = HVAC(adj_outside,adj_hall,adj,rooms,default_settings)

    @property
    def state_size(self):
        return self._num_state_units

    @property
    def output_size(self):
        return self._num_reward_units

    def __call__(self, inputs, state, scope=None):
        next_state =  self.hvac.Transition(state, inputs)
        reward = self.hvac.Reward(state, inputs)      
        return tf.concat(1,[reward,next_state]), next_state
    


# In[22]:

hvac_inst_cell = HVACCell(adj_outside,adj_hall,adj,rooms,default_settings)


# In[23]:

a = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[2,60]),name="action")
initial_state = hvac_inst_cell.zero_state(2, dtype=tf.float32)+tf.constant([[random.randint(0,30) for _ in range(60)]],dtype=tf.float32)
hvac_inst_cell(a,initial_state )
#print(initial_state.get_shape())


# In[24]:

class ActionOptimizer(object):
    def __init__(self,
                a, # Actions
                num_step, # Number of RNN step, this is a fixed step RNN sequence, 12 for navigation
                batch_size,
                learning_rate=0.1): 
        self.action = tf.reshape(a,[-1,num_step,60]) #Reshape rewards
        print(self.action)
        self.num_step = num_step
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._p_create_rnn_graph()
        self._p_Q_loss()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def _p_create_rnn_graph(self):
        cell = HVACCell(adj_outside,adj_hall,adj,rooms,default_settings)
        initial_state = cell.zero_state(self.action.get_shape()[0], dtype=tf.float32)+tf.constant([RandomInitialandWriteFile(rooms)],dtype=tf.float32)
        print('action batch size:{0}'.format(array_ops.shape(self.action)[0]))
        print('Initial_state shape:{0}'.format(initial_state))
        rnn_outputs, state = tf.nn.dynamic_rnn(cell, self.action, dtype=tf.float32,initial_state=initial_state)
        #need output intermediate states as well
        concated = tf.concat(0,rnn_outputs)
        something_unpacked =  tf.unpack(concated, axis=2)
        self.outputs = tf.reshape(something_unpacked[0],[-1,self.num_step,1])
        self.intern_states = tf.pack(something_unpacked[1:61], axis=2)
        self.last_state = state
        self.pred = tf.reduce_sum(self.outputs,1)
        print("self.pred:{0}".format(self.pred))
            
    def _p_create_loss(self):

        objective = tf.reduce_mean(self.pred) 
        self.loss = objective
        print(self.loss.get_shape())
        #self.loss = -objective
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[a])
        
    def _p_Q_loss(self):
        objective = tf.constant(0.0, shape=[self.batch_size, 1])
        for i in range(self.num_step):
            Rt = self.outputs[:,i]
            SumRj=tf.constant(0.0, shape=[self.batch_size, 1])
            #SumRk=tf.constant(0.0, shape=[self.batch_size, 1])
            if i<(self.num_step-1):
                j = i+1
                SumRj = tf.reduce_sum(self.outputs[:,j:],1)
            #if i<(self.num_step-1):
                #k= i+1
                #SumRk = tf.reduce_sum(self.outputs[:,k:],1)
            objective+=(Rt*SumRj+tf.square(Rt))/(self.num_step-i)
        self.loss = tf.reduce_mean(objective)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[a])
        
    def Optimize(self,epoch=100):
        
        new_loss = self.sess.run([self.loss])
        print('Loss in epoch {0}: {1}'.format("Initial", new_loss)) 
        for epoch in range(epoch):
            training = self.sess.run([self.optimizer])
            self.sess.run(tf.assign(a, tf.clip_by_value(a, 0, 10)))
            if True:
                new_loss = self.sess.run([self.loss])
                print('Loss in epoch {0}: {1}'.format(epoch, new_loss))  
        minimum_costs_id=self.sess.run(tf.argmax(self.pred,0))
        print(minimum_costs_id)
        print('Optimal Action Squence:{0}'.format(self.sess.run(self.action)[minimum_costs_id[0]]))
        action = self.sess.run(self.action)[minimum_costs_id[0]]
        np.savetxt("HVAC_ACTION.csv",action,delimiter=",",fmt='%2.5f')
        print('Best Cost: {0}'.format(self.sess.run(self.pred)[minimum_costs_id[0]]))
        pred_list = self.sess.run(self.pred)
        pred_list=np.sort(pred_list.flatten())[::-1]
        pred_list=pred_list[:5]
        pred_mean = np.mean(pred_list)
        pred_std = np.std(pred_list)
        print('Best Cost: {0}'.format(pred_list[0]))
        print('Sorted Costs:{0}'.format(pred_list))
        print('MEAN: {0}, STD:{1}'.format(pred_mean,pred_std))
        print('The last state:{0}'.format(self.sess.run(self.last_state)))
        print('Rewards each time step:{0}'.format(self.sess.run(self.outputs)))
        reward = self.sess.run(self.outputs)[minimum_costs_id[0]]
        np.savetxt("HVAC_REWARD.csv",reward,delimiter=",",fmt='%7.5f')
        #print('Intermediate states:{0}'.format(self.sess.run(self.intern_states)[minimum_costs_id[0]]))
        interm = self.sess.run(self.intern_states)[minimum_costs_id[0]]
        np.savetxt("HVAC_INTERM.csv",interm,delimiter=",",fmt='%2.5f')


# In[25]:

a = tf.Variable(tf.constant(5.0, dtype=tf.float32,shape=[72000]),name="action")
rnn_inst = ActionOptimizer(a, 12,100)  


# In[26]:

rnn_inst.Optimize(500)


# In[ ]:



