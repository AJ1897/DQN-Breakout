#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

from environment import Environment
from test import test
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from agent import Agent
from dqn_model import DQN
from torch.utils.tensorboard import SummaryWriter
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)
Path_weights = './last_train_weights_dueldqn.tar'
Path_memory = './last_memory_dueldqn.tar'
tensor_board_dir='./logs/train_data'
writer = SummaryWriter(tensor_board_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)
class my_dataset(Dataset):
    def __init__(self,data):
        self.samples = data
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,idx):
        return self.samples[idx]

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.epochs = 10
        self.args = args
        self.n_episodes = 10000000
        self.env = env
        self.nA = self.env.action_space.n
        # self.nS = self.env.observation_space
        self.batch_size = 32
        self.eval_num=0
        self.DQN = DQN().to(device)
        self.Target_DQN = DQN().to(device)
        test_case = torch.zeros(32,4,84,84)
        writer.add_graph(DQN(),test_case)
        self.buffer_memory = 1000000
        self.train_buffer_size = 4
        self.min_buffer_size = 10000
        self.target_update_buffer =  50000
        self.learning_rate = 0.00006
        self.discount_factor = 0.99
        self.epsilon = 1
        self.min_epsilon = 0.001
        # self.decay_rate = 0.999
        self.ep_decrement = (self.epsilon - self.min_epsilon)/self.n_episodes
        self.criteria = nn.SmoothL1Loss()
        self.optimiser = optim.Adam(self.DQN.parameters(),self.learning_rate)
        self.buffer=[]
        self.Evaluation = 100000
        self.total_evaluation__episodes = 100
        self.full_train = 100000
        self.next_obs = np.transpose(self.env.reset(),(2,0,1))
        self.done = False
        self.x = 0
        self.current = 0
        self.reward_list =[]
        self.loss_list= []
        self.current_train = 0
        self.current_target = 0
        self.max_test_reward=0
        writer.add_hparams({"Learning_Rate":self.learning_rate,"Batch_Size":self.batch_size,"Discount Factor":self.discount_factor,"Min Epsilon":self.min_epsilon,"Total Episodes":self.n_episodes,"Buffer Size":self.buffer_memory},{"Max__Test_Reward":self.max_test_reward})
        if args.cont:
          print("#"*50+"Resuming Training"+"#"*50)
          dic_weights = torch.load(Path_weights,map_location=device)
          dic_memory = torch.load(Path_memory)
          self.epsilon = dic_memory['epsilon']
          #self.epsilon = 0.0001
          self.x = dic_memory['x']
          print(self.x)
          self.ep_decrement = (1 - self.min_epsilon)/(self.n_episodes)
          self.current = dic_memory['current_info'][0]
          self.current_target = dic_memory['current_info'][1]
          self.current_train = dic_memory['current_info'][2]
          self.next_obs = dic_memory['next_info'][0]
          self.done = dic_memory['next_info'][1]
          # self.buffer = dic_memory['buffer']
          # self.buffer = []
          # self.reward_list = dic_memory['reward_list']
          self.reward_list = []
          self.DQN.load_state_dict(dic_weights['train_state_dict'])
          self.Target_DQN.load_state_dict(dic_weights['target_state_dict'])
          self.DQN.train()
          self.Target_DQN.train()
          self.optimiser.load_state_dict(dic_weights['optimiser_state_dict'])
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            dic_weights = torch.load(Path_weights,map_location=device)
            self.Target_DQN.load_state_dict(dic_weights['target_state_dict'])
            self.Target_DQN.eval()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        obs = self.env.reset()
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if not test:
            p = random.random()
            if p < self.epsilon:
                action = np.random.randint(0,self.nA)
            else:
                a = self.DQN(torch.from_numpy(observation).unsqueeze(0).to(device))
                action = np.argmax(a.detach().cpu().numpy())
        else:
            a = self.Target_DQN(torch.from_numpy(observation).unsqueeze(0).to(device))
            action = np.argmax(a.detach().cpu().numpy())
        ###########################
        return action
    
    def push(self,episode):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.buffer) < self.buffer_memory:
            self.buffer.append(episode)
        else:
            self.buffer.pop(0)
            self.buffer.append(episode)
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        batch = random.sample(self.buffer,self.batch_size)
        # print(np.shape(batch[0][:]))
        batch = list(zip(*batch))
        # print(np.asarray(batch[1]))
        batch_x = torch.from_numpy(np.asarray(batch[0]))
        act = torch.from_numpy(np.vstack(batch[1]))
        rew = torch.from_numpy(np.asarray(batch[2]))
        dones = torch.from_numpy(np.asarray(batch[3]))
        batch_y = torch.from_numpy(np.asarray(batch[4]))
        # print(act.shape)
        ###########################
        return batch_x,act,rew,dones,batch_y
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        for x in range(self.x,self.n_episodes):
            # obs = np.transpose(self.env.reset(),(2,0,1))
            # print("Episode No. : %d"%x)
            obs = self.next_obs
            # print(obs[0][40][:])
            done = self.done
            accumulated_rewards = 0
            while (not done):
                # self.env.render()
                action = self.make_action(obs,False)
                next_obs,reward,done,info = self.env.step(action)
                next_obs = np.transpose(next_obs,(2,0,1))
                # print(info['ale.lives'])
                # print(np.shape(e_list[-1]))
                accumulated_rewards+=reward
                self.push([obs,action,reward,done,next_obs])
                if self.epsilon > self.min_epsilon:
                  self.epsilon-=self.ep_decrement
                self.current+=1
                self.current_train += 1
                self.current_target += 1
                obs = next_obs
                if self.current_train % self.train_buffer_size == 0 and len(self.buffer) > self.min_buffer_size:
                    batch_x,act,rew,dones,batch_y = self.replay_buffer()
                    self.optimiser.zero_grad()
                    future_return =  self.Target_DQN(batch_y.to(device)).max(1)[0].detach() * self.discount_factor
                    future_return[dones] = 0
                    y = rew.to(device) + future_return
                    c_q = self.DQN(batch_x.to(device)).gather(1,act.to(device))
                    loss = self.criteria(c_q.double(),(y.double()).unsqueeze(1))
                    # loss_list.append(loss.detach())
                    loss.backward()
                    # self.env.render()
                    self.optimiser.step()
                    self.current_train = 0

                if self.current_target > self.target_update_buffer:
                    self.Target_DQN.load_state_dict(self.DQN.state_dict())
                    self.current_target = 0

                
                if self.current % self.Evaluation == 0:
                    # print("\n Weights: \n",list(self.DQN.parameters()),"\r")
                    print("\n","#" * 40, "Evaluation number %d"%(self.current/self.Evaluation),"#" * 40)
                    self.eval_num = self.current/self.Evaluation
                    env1 = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True)
                    test(self,env1,total_episodes=100)
                    writer.add_scalar("Test/Max_test_Reward",self.max_test_reward,self.eval_num)
                    print("#" * 40, "Evaluation Ended!","#" * 40,"\n")
            self.next_obs = np.transpose(self.env.reset(),(2,0,1))
            self.done = False
            self.reward_list.append(accumulated_rewards)
            writer.add_scalar('Train/Episodic_Reward(Mean of last 30)',np.mean(self.reward_list[-30:]),x)
            if len(self.reward_list) % 200 == 0:
                self.reward_list = self.reward_list[-150:]
            if (x+1)%100 == 0:
                print("Current = %d, episode = %d, Average_reward = %0.2f, epsilon = %0.2f"%(self.current, x+1, np.mean(self.reward_list[-100:]), self.epsilon))
            if (x+1)%3000 == 0:
                print("Saving_Weights_Model")
                torch.save({
                  'target_state_dict':self.Target_DQN.state_dict(),
                  'train_state_dict':self.DQN.state_dict(),
                  'optimiser_state_dict':self.optimiser.state_dict()
                  },Path_weights)
                print("Saving_Memory_Info")
                torch.save({
                  'current_info':[self.current,self.current_target,self.current_train],
                  'x':x+1,
                  'next_info':[self.next_obs,self.done],
                  'epsilon':self.epsilon,
                  # 'buffer':self.buffer,
                  #'reward_list':self.reward_list
                  }
                  ,Path_memory)






        
        ###########################
