# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:58:16 2018

@author: hasee
"""

import gym
import tensorflow as tf
import numpy as np
import random
from   matplotlib           import pyplot as plt

import time  

GAMMA           =0.9
INITIAL_EPSILON =1
FINAL_EPSILON   =0.1        
REPLAY_SIZE     =10000       
batch_size      =32        

ENV_NAME='MountainCar-v0'
EPISODE =1000
STEP    =200
TEST    =10
env     = gym.make(ENV_NAME) 
env     = env.unwrapped



   
    





class DeepQNetwork(object):  
    def __init__(self,  
                 n_actions,  
                 n_features,  
                 learning_rate=0.01,  
                 reward_decay=0.9,  
                 epsilon_greedy=0.9,   
                 epsilon_increment = 0.0001,  
                 replace_target_iter=300,     
                 buffer_size=10000,   
                 batch_size=32,  
                 ):  
        self.n_actions = n_actions  
        self.n_features = n_features  
        self.lr = learning_rate  
        self.gamma = reward_decay  
        self.epsilon_max = epsilon_greedy  
        self.replace_target_iter = replace_target_iter  
        self.buffer_size = buffer_size  
        self.buffer_counter = 0   
        self.batch_size = batch_size  
        self.epsilon = 0 if epsilon_increment is not None else epsilon_greedy  
        self.epsilon_max = epsilon_greedy  
        self.epsilon_increment = epsilon_increment  
        self.learn_step_counter = 0  
        self.buffer = np.zeros((self.buffer_size, n_features * 2 + 2))   
        self.build_net()  
      
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')  
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')  
        with tf.variable_scope('soft_replacement'):  
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]  
        self.sess = tf.Session()  
        tf.summary.FileWriter('logs/', self.sess.graph)  
        self.sess.run(tf.global_variables_initializer())  
  
    def build_net(self):  
        self.s     = tf.placeholder(tf.float32, [None, self.n_features])  
        self.s_    = tf.placeholder(tf.float32, [None, self.n_features])  
        self.r     = tf.placeholder(tf.float32, [None, ])  
        self.a     = tf.placeholder(tf.int32,   [None, ])  
        self.done  = tf.placeholder(tf.int32,   [None, ]) 
  
        w_initializer = tf.random_normal_initializer(0., 0.3)  
        b_initializer = tf.constant_initializer(0.1)  
       
        with tf.variable_scope('eval_net'):  
            eval_layer = tf.layers.dense(self.s, 20, tf.nn.relu, kernel_initializer=w_initializer,  
                                         bias_initializer=b_initializer, name='eval_layer')  
            self.q_eval = tf.layers.dense(eval_layer, self.n_actions, kernel_initializer=w_initializer,  
                                          bias_initializer=b_initializer, name='output_layer1')  
        with tf.variable_scope('target_net'):  
            target_layer = tf.layers.dense(self.s_, 20, tf.nn.relu, kernel_initializer=w_initializer,  
                                           bias_initializer=b_initializer, name='target_layer')  
            self.q_next = tf.layers.dense(target_layer, self.n_actions, kernel_initializer=w_initializer,  
                                          bias_initializer=b_initializer, name='output_layer2')  
        with tf.variable_scope('q_target'):  
         
            self.q_target = tf.stop_gradient(self.r + self.gamma *tf.reduce_max(self.q_next, axis=1))  
        with tf.variable_scope('q_eval'):  
            
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0]), self.a], axis=1)  
            self.q_eval_a = tf.gather_nd(params=self.q_eval, indices=a_indices)  
        with tf.variable_scope('loss'):  
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_a))  
        with tf.variable_scope('train'):  
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)  
  
           
  
    def store_transition(self, s, a, r, s_):  
        transition = np.hstack((s, a, r, s_))  
        index = self.buffer_counter % self.buffer_size  
        self.buffer[index, :] = transition  
        self.buffer_counter += 1  
  
    def choose_action_by_epsilon_greedy(self, status):  
        status = status[np.newaxis, :]  
        if random.random() < self.epsilon:  
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: status})  
            action = np.argmax(actions_value)  
        else:  
            action = np.random.randint(0, self.n_actions)  
        return action  
  
    def learn(self):  
        
        if self.learn_step_counter % self.replace_target_iter == 0:  
            self.sess.run(self.target_replace_op)  
            
        sample_index = np.random.choice(min(self.buffer_counter, self.buffer_size), size=self.batch_size)  
        batch_buffer = self.buffer[sample_index, :]

        _, cost = self.sess.run([self.train_op, self.loss], feed_dict={  
            self.s:    batch_buffer[:, :self.n_features],  
            self.a:    batch_buffer[:, self.n_features],  
            self.r:    batch_buffer[:, self.n_features+1],  
            self.s_:   batch_buffer[:, -self.n_features:],
              
        })  
        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)  
        self.learn_step_counter += 1  
        return cost  



def main():
    #initialize OpenAI Gym env and dqn agent
           
    agent  = DeepQNetwork (n_actions=env.action_space.n,n_features=env.observation_space.shape[0]) 
    
    print(env.observation_space)
    print(env.action_space)
    print(env.reward_range)
    
    cost=[]
    total_reward=[]
    test_time   =[]
    
    for episode in range(EPISODE):
        # INITIAL TASK
        state= env.reset()
        # Train
        episode_reward=0
        total_step = 0 
        start_time=time.time()
        
        t_reward=0
        while(True):
#            env.render() 
            action                    = agent.choose_action_by_epsilon_greedy(state)
            next_state,reward,done,_  = env.step(action)
            #Define reward for agent
            #print "next_state",next_state
            
            position, velocity = next_state 
            
            
            t_reward=t_reward+reward
          
            reward = abs(next_state[0]+0.5)
            
        
            
            agent.store_transition(state,action,reward,next_state)
#            print(reward,next_state[0]-state[0])
            if total_step > 200:  
                cost_ = agent.learn() 
                cost.append(cost_)  
            episode_reward += -reward 
            state=next_state
            
            
            if done:
                 print('episode:', episode,  
                       'episode_reward %.2f' % episode_reward,  
                       'epsilon %.2f'        % agent.epsilon)  
                 
                 end_time=time.time()
                 run_time=end_time-start_time
                 print("time is : ",run_time)
                 test_time.append(run_time)
                 total_reward.append(episode_reward)
                 break
            total_step = total_step+1
        if episode%100==0 and episode>10:
            for test_steps in range(10):
                _state= env.reset()
                t_reward=0
                start_time=time.time()
                
                while(True):
                    env.render()
                    action                    = agent.choose_action_by_epsilon_greedy(_state)
                    next_state,reward,done,_  = env.step(action)
                    position,velocity         = next_state
                    
                    t_reward=t_reward+reward
                    end_time=time.time()-start_time
                    if end_time>10:
                        print('episode:', test_steps,'out of time')
                        break
                    
                    _state=next_state
                    if done:
                          print('episode:', test_steps,  
                               ' episode_reward %.2f' % t_reward,  
                                'epsilon %.2f'        % agent.epsilon)  
                          break
    fig1 = plt.figure(figsize=(10,5))  
    plt.subplot(211)     
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title ("Episode Reward over Time")  
    plt.plot(np.arange(len(total_reward)), total_reward)  
    plt.show(fig1)         
    fig2 = plt.figure(figsize=(10,5))
    plt.subplot(212)
    plt.xlabel("Episode")
    plt.ylabel("Episode run time")
    plt.title ("Episode run time over Time")
    plt.plot(np.arange(len(test_time)), test_time)  
    plt.show(fig2)         
                   
            

if __name__=='__main__':
    main()





  
 


















