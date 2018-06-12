# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:27:49 2018

@author: hasee
"""

import gym
import numpy as np
 
import tensorflow as tf
import matplotlib.pyplot as plt
import time
 


class MLPStochasticPolicyAgent:
    def __init__(self,env,n_actions,n_features,learning_rate=0.01,reward_decay=0.95,output_graph=False,):
        self._env=env
        self._sess=tf.Session()
        self._states = tf.placeholder(tf.float32,(None,n_features),name="states")
        self.tf_acts = tf.placeholder(tf.int32,[None, ], name="actions_num")
        self.tf_vt   = tf.placeholder(tf.float32,[None, ], name="actions_value")
        
        
        self.n_actions  = n_actions
        self.n_features = n_features
        self.lr         = learning_rate
        self.gamma      = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._phi_hidden=10
        self._sigma_hidden=3

 

        
        self._w1=tf.get_variable("w1",[n_features,self._phi_hidden],initializer=tf.random_normal_initializer())
        self._b1=tf.get_variable("b1",[self._phi_hidden],initializer=tf.constant_initializer(0))
        self._w2=tf.get_variable("w2",[self._phi_hidden,self._sigma_hidden],initializer=tf.random_normal_initializer())
        self._b2=tf.get_variable("b2",[self._sigma_hidden],initializer=tf.constant_initializer(0))
        
        self._h1 =tf.nn.tanh(tf.matmul(self._states,self._w1)+self._b1)
        self._phi=tf.nn.tanh(tf.matmul(self._h1,self._w2)+self._b2)
        self._softmax=tf.nn.softmax(self._phi,name='phi_softmax')
        
 
            
        with tf.name_scope('loss'):
             neg_log_prob = tf.reduce_sum(-tf.log(self._softmax)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
             loss = tf.reduce_mean(neg_log_prob * self.tf_vt) 
    
        with tf.name_scope('train'):
             self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
             
        
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self._sess.run(tf.global_variables_initializer())     
             
    
    def sample_action(self,observation):
        prob_weights = self._sess.run(self._softmax, feed_dict={self._states: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
        
      

    def store_rollout(self,state,action,reward):
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(reward)
   
    def update_model(self):    
      
        r=0
        discounted_rewards= np.zeros_like(self.ep_rs)
        for t in reversed(range(0,len(self.ep_rs))):
            r=self.ep_rs[t]+self.gamma*r
            discounted_rewards[t]=r

        # reduce gradient variance by normalization
 
        discounted_rewards-=np.mean(discounted_rewards)
        discounted_rewards/=np.std(discounted_rewards)
    

        # train on episode
        self._sess.run(self.train_op, feed_dict={
             self._states:  np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt:   discounted_rewards,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_rewards
            
 
        
        
DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold
# episode: 154   reward: -10667
# episode: 387   reward: -2009
# episode: 489   reward: -1006
# episode: 628   reward: -502

RENDER = False  # rendering wastes time

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
       
        
RL = MLPStochasticPolicyAgent(
    env=env,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)    
     
    
    
if __name__=='__main__':
    rewards=[]
    for i_episode in range(1000):
        print(i_episode)
        observation = env.reset()
        
        while True:
            if RENDER: env.render()
    
            action = RL.sample_action(observation)
    
            observation_, reward, done, info = env.step(action)     # reward = -1 in all cases
    
            RL.store_rollout(observation, action, reward)
       
            if done:
                # calculate running reward
                ep_rs_sum = sum(RL.ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                rewards.append(running_reward)  
#                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
    
                print("episode:", i_episode, "  reward:", int(running_reward))
    
                vt = RL.update_model()  # train
    
                if i_episode == 30:
                    plt.plot(vt)  # plot the episode vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
    
                break
    
            observation = observation_
#        if i_episode%100==0 and i_episode>10:
#            for test_steps in range(10):
#                state= env.reset()
#                t_reward=0
#                start_time=time.time()
#                
#                while(True):
##                    env.render()
#                    action                    = RL.sample_action(observation)
#                    next_state,reward,done,_  = env.step(action)
#                    position,velocity         = next_state
#                    
#                   
#                    end_time=time.time()-start_time
##                    if end_time>20:
##                        print('episode:', test_steps,'out of time')
##                        break
#                    
#                    if done:
#                          print(end_time)
#                          print('episode:', test_steps,  
#                               ' episode_reward %.2f' % t_reward)      
#                          break
    fig1 = plt.figure(figsize=(10,5))  
    plt.subplot(211)     
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title ("Episode Reward over Time")  
    plt.plot(np.arange(len(rewards)), rewards)  
    plt.show(fig1)                      