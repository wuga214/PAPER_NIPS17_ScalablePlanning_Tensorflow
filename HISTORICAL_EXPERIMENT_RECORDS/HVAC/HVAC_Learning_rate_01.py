
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
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


# In[3]:

default_settings = {                
    "cap": 80, 
    "outside_resist" : 2.0,
    "hall_resist" : 1.3,
    "wall_resist" : 1.1,
    "cap_air" : 1.006, 
    "cost_air" : 1.0, 
    "time_delta" : 1.0,
    "temp_air" : 40.0,
    "temp_up" : 23.5,
    "temp_low" : 20.0,
    "temp_outside" : 6.0,
    "temp_hall" : 10.0,
    "penalty" : 1000.0,
    "air_max" : 10.0
   }


# In[4]:

#Matrix computation version update
class HVAC(object):
    def __init__(self, 
                 adj_outside, #Adjacent to outside 
                 adj_hall, #Adjacent to hall
                 adj, #Adjacent between rooms
                 rooms, #Room names
                 batch_size,
                 default_settings):
        self.__dict__.update(default_settings)
        self.rooms = rooms
        self.batch_size = batch_size
        self.room_size = len(rooms)
        self.zero = tf.constant(0, shape=[self.batch_size,self.room_size], dtype=tf.float32)
        self._init_ADJ_Matrix(adj)
        self._init_ADJOUT_MATRIX(adj_outside)
        self._init_ADJHALL_MATRIX(adj_hall)
    
    def _init_ADJ_Matrix(self,adj):
        np_adj = np.zeros((self.room_size,self.room_size))
        for i in adj:
            m=self.rooms.index(i[0])
            n=self.rooms.index(i[1])
            np_adj[m,n] = 1
            np_adj[n,m] = 1
        self.adj = tf.constant(np_adj,dtype=tf.float32)
        print('self.adj shape:{0}'.format(self.adj.get_shape()))
            
    def _init_ADJOUT_MATRIX(self, adj_outside):
        np_adj_outside = np.zeros((self.room_size,))
        for i in adj_outside:
            m=self.rooms.index(i)
            np_adj_outside[m] = 1
        self.adj_outside = tf.constant(np_adj_outside,dtype=tf.float32)
        
    def _init_ADJHALL_MATRIX(self,adj_hall):
        np_adj_hall = np.zeros((self.room_size,))
        for i in adj_hall:
            m=self.rooms.index(i)
            np_adj_hall[m] = 1
        self.adj_hall = tf.constant(np_adj_hall,dtype=tf.float32)
    
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
        print('state shape:{0}'.format(states.get_shape()))
        heating_info = actions*self.CAP_AIR()*(self.TEMP_AIR()-previous_state)
        neighbor_info = (tf.transpose(tf.matmul(self.ADJ(),tf.transpose(states)))                         -previous_state*tf.reduce_sum(self.ADJ(),1))/self.R_WALL()
        outside_info = (self.TEMP_OUTSIDE()-previous_state)*self.ADJ_OUTSIDE()/self.R_OUTSIDE()
        hall_info = (self.TEMP_HALL()-previous_state)*self.ADJ_HALL()/self.R_HALL()
        print('neighbor_info shape:{0}'.format(neighbor_info.get_shape()))
        print('hall_info shape:{0}'.format(hall_info.get_shape()))
        new_state = previous_state+self.TIME_DELTA()/self.CAP()*(heating_info +                                                                  neighbor_info + outside_info + hall_info)
        return new_state
            
    def Reward(self, states,actions):
        batch_size,state_size = states.get_shape()
        #break_penalty = tf.select(tf.logical_or(tf.less(states,self.TEMP_LOW()),\
        #                                        tf.greater(states,self.TEMP_UP())),self.PENALTY()+self.ZERO(),self.ZERO())
        dist_penalty = tf.abs(((self.TEMP_UP()+self.TEMP_LOW())/tf.constant(2.0, dtype=tf.float32))-states)
        ener_penalty = actions*self.COST_AIR()
        new_rewards = -tf.reduce_sum(tf.constant(10.0, tf.float32)*dist_penalty+ener_penalty,1,keep_dims=True)
        return new_rewards            


# In[5]:

adj_hall = [101,102,103,106,107,109,110,               201,202,203,206,207,209,210,               301,302,303,306,307,309,310,               401,402,403,406,407,409,410,               501,502,503,506,507,509,510]
adj_outside = [101,102,103,104,105,106,108,110,111,112,              201,202,203,204,205,206,208,210,211,212,              301,302,303,304,305,306,308,310,311,312,              401,402,403,404,405,406,408,410,411,412,              501,502,503,504,505,506,508,510,511,512]
adj = [[101,102],[102,103],[103,104],[104,105],[106,107],[107,108],[107,109],[108,109],[110,111],[111,112],       [201,202],[202,203],[203,204],[204,205],[206,207],[207,208],[207,209],[208,209],[210,211],[211,212],       [301,302],[302,303],[303,304],[304,305],[306,307],[307,308],[307,309],[308,309],[310,311],[311,312],       [401,402],[402,403],[403,404],[404,405],[406,407],[407,408],[407,409],[408,409],[410,411],[411,412],       [501,502],[502,503],[503,504],[504,505],[506,507],[507,508],[507,509],[508,509],[510,511],[511,512],       [101,201],[102,202],[103,203],[104,204],[105,205],[106,206],[107,207],[108,208],[109,209],[110,210],       [111,211],[112,212],[201,301],[202,302],[203,303],[204,304],[205,305],[206,306],[207,307],[208,308],       [209,309],[210,310],[211,311],[212,312],[301,401],[302,402],[303,403],[304,404],[305,405],[306,406],       [307,407],[308,408],[309,409],[310,410],[311,411],[312,412],[401,501],[402,502],[403,503],[404,504],       [405,505],[406,506],[407,507],[408,508],[409,509],[410,510],[411,511],[412,512]]
rooms = list(range(101,113))+list(range(201,213))+list(range(301,313))+list(range(401,413))+list(range(501,513))

batch_size = 2


# In[6]:

x=list(range(101,113))
x


# In[7]:

# hvac_inst = HVAC(adj_outside,adj_hall,adj,rooms,10,default_settings)


# In[8]:

# States
states = tf.placeholder(tf.float32,[10, 60],name="States")

# Actions
actions = tf.placeholder(tf.float32,[10, 60],name="Actions")


# In[9]:

# states_list=tf.unpack(states)
# actions_list = tf.unpack(actions)
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# new_state = hvac_inst.Transition(states, actions)
# feed_dict={states:S_A_matrix[:10,60:], actions:S_A_matrix[:10,:60]}

# print(sess.run([new_state], feed_dict=feed_dict))


# In[10]:

# new_rewards = hvac_inst.Reward(states,actions)


# In[11]:

# feed_dict={states:S_A_matrix[:10,60:], actions:S_A_matrix[:10,:60]}
# sess.run(new_rewards,feed_dict=feed_dict )


# In[12]:

class HVACCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, adj_outside,adj_hall,adj,rooms,batch_size,default_settings):
        self._num_state_units = len(rooms)
        self._num_reward_units = 1+len(rooms)
        self.hvac = HVAC(adj_outside,adj_hall,adj,rooms,batch_size,default_settings)

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
    


# In[13]:

hvac_inst_cell = HVACCell(adj_outside,adj_hall,adj,rooms,batch_size,default_settings)


# In[14]:

a = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[2,60]),name="action")
initial_state = hvac_inst_cell.zero_state(2, dtype=tf.float32)+tf.constant([[random.randint(0,30) for _ in range(60)]],dtype=tf.float32)
hvac_inst_cell(a,initial_state )
#print(initial_state.get_shape())


# In[15]:

class ActionOptimizer(object):
    def __init__(self,
                a, # Actions
                num_step, # Number of RNN step, this is a fixed step RNN sequence, 12 for navigation
                batch_size,
                loss,
                learning_rate=0.001): 
        self.action = tf.reshape(a,[-1,num_step,60]) #Reshape rewards
        print(self.action)
        self.num_step = num_step
        self.batch_size=batch_size
        self.learning_rate = learning_rate
        self.previous_output = np.zeros((batch_size,num_step))
        self.weights = np.ones((batch_size,num_step,1))
        self._p_create_rnn_graph()
        if loss == "Qloss":
            self._p_Q_loss()
        else:
            self._p_create_loss()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    
    def _p_create_rnn_graph(self):
        cell = HVACCell(adj_outside,adj_hall,adj,rooms,self.batch_size,default_settings)
        initial_state = cell.zero_state(self.action.get_shape()[0], dtype=tf.float32)                        +tf.constant(10,dtype=tf.float32)
        #+tf.constant([RandomInitialandWriteFile(rooms)],dtype=tf.float32)
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
        self.average_pred = tf.reduce_mean(self.pred)
        print("self.pred:{0}".format(self.pred))
            
    def _p_create_loss(self):
        print("MSE-loss")
        objective = tf.reduce_mean(tf.square(self.pred)) 
        self.loss = objective
        print(self.loss.get_shape())
        #self.loss = -objective
        self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss, var_list=[a])
        
    def _p_Q_loss(self):
        print("Q-loss")
        
        objective = tf.reduce_sum(self.outputs*self.weights,1)
        self.loss = tf.reduce_mean(tf.square(objective))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, var_list=[a])
        
    def softmax(self,z,dim):
        sm = np.exp(z) / np.sum(np.exp(z),axis=dim,keepdims=True)
        return sm
        
    def _p_attention(self,new_output):
        value = new_output-self.previous_output
        self.weights = self.softmax(value+np.amin(value),1).reshape(self.batch_size,self.num_step,1)
        self.previous_output = new_output
        
        
    def Optimize(self,epoch=100):
#         Time_Target_List = [15,30,60,120,240,480,960]
#         #Time_Target_List = [15,30,60,120]
#         Target = Time_Target_List[0]
#         counter = 0
#         new_loss = self.sess.run([self.average_pred])  
#         self.previous_output = np.log(-self.sess.run([self.outputs])[0].reshape((self.batch_size,self.num_step)))
#         print('Loss in epoch {0}: {1}'.format("Initial", new_loss)) 
#         print('Compile to backend complete!') 
#         start = time.time()
#         current_best = 0
#         while True:
#             training = self.sess.run([self.optimizer])
#             self.sess.run(tf.assign(a, tf.clip_by_value(a, 0, 10)))
#             new_output = np.log(-self.sess.run([self.outputs])[0])  
#             self._p_attention(new_output.reshape((self.batch_size,self.num_step)))
#             end = time.time()
#             if end-start>=Target:
#                 print('Time: {0}'.format(Target))
#                 pred_list = self.sess.run(self.pred)
#                 pred_list=np.sort(pred_list.flatten())[::-1]
#                 pred_list=pred_list[:5]
#                 pred_mean = np.mean(pred_list)
#                 pred_std = np.std(pred_list)
#                 if counter == 0:
#                     current_best = pred_list[0]
#                 if pred_list[0]>current_best:
#                     current_best=pred_list[0]
#                 print('Best Cost: {0}'.format(current_best))
#                 print('MEAN: {0}, STD:{1}'.format(pred_mean,pred_std))
#                 counter = counter+1
#                 if counter == len(Time_Target_List):
#                     print("Done!")
#                     break
#                 else:
#                     Target = Time_Target_List[counter]
        
        new_loss = self.sess.run([self.average_pred])
        print('Loss in epoch {0}: {1}'.format("Initial", new_loss)) 
        for epoch in range(epoch):
            training = self.sess.run([self.optimizer])
            self.sess.run(tf.assign(a, tf.clip_by_value(a, 0, 10)))
            if True:
                new_loss = self.sess.run([self.average_pred])
                print('Loss in epoch {0}: {1}'.format(epoch, new_loss))  
        minimum_costs_id=self.sess.run(tf.argmax(self.pred,0))
        print(minimum_costs_id)
        print('Optimal Action Squence:{0}'.format(self.sess.run(self.action)[minimum_costs_id[0]]))
        action = self.sess.run(self.action)[minimum_costs_id[0]]
        np.savetxt("HVAC_ACTION.csv",action,delimiter=",",fmt='%2.5f')
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


# In[16]:

sess = tf.InteractiveSession()
initial_a = tf.truncated_normal(shape=[576000],mean=5.0, stddev=1.0).eval() 
a = tf.Variable(initial_a,name="action")
rnn_inst = ActionOptimizer(a, 96,100,"MSE")  


# In[17]:

rnn_inst.Optimize(4000)


