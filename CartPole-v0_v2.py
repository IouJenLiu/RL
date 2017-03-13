#cart_pole-v0
#action = {0, 1}
#observation is a 4 * 1 vector
#solve: average reward > 200 for 100 consecutive episode

from collections import namedtuple
import argparse
import math
import sys
import gym
from gym import wrappers
import torch
import torch.nn as nn
import numpy as np
import random
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = 'DQN for cartPole')
parser.add_argument('--gamma', type = float, default = 0.99)
parser.add_argument('--max_episode', type = int, default = 100000)
parser.add_argument('--lr', type = float, default = 0.005)
parser.add_argument('--lr_decay', type = float, default = 1)
parser.add_argument('--mem_capacity', type = int, default = 10000)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--D_h', type = int, default = 30)
parser.add_argument('--max_step', type = int, default = 700)
parser.add_argument('--epsilon', type = float, default = 0.95)
parser.add_argument('--important_factor', type = int, default = 3)
args = parser.parse_args()

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, 'video/cartpole_v0-experiment-1')
D_in = env.observation_space.shape[0]
D_out = env.action_space.n
D_h = args.D_h
dtype = torch.FloatTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(D_in, D_h)
        self.bn1 = nn.BatchNorm1d(D_h)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(D_h, D_h)
        self.bn2 = nn.BatchNorm1d(D_h)
        self.relu2 = nn.ReLU()
        #self.fc3 = nn.Linear(D_h, D_h)
        #self.bn3 = nn.BatchNorm1d(D_h)
        #self.relu3 = nn.ReLU()
        self.fc_out = nn.Linear(D_h, D_out)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        #x = self.fc3(x)
        #x = self.bn3(x)
        #x = self.relu3(x)
        q = self.fc_out(x)
        return q

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def Sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

model = Model()
memory = ReplayMemory(args.mem_capacity)
loss_fn = nn.SmoothL1Loss()
learning_rate = args.lr



def SelectAction(s, i_epi):
    model.eval()
    s = Variable(s, volatile = True).float()
    all_Q = model(s)
    #print("s", s)
    #print("all_Q", all_Q)
    a = all_Q.data.max(1)[1]
    if random.random() > args.epsilon:
        a = torch.LongTensor(1, 1)
        a[0, 0] = random.randrange(D_out)
    return a


def OptModel():
    if len(memory) < args.batch_size:
        return
    global learning_rate
    model.train()
    transition = memory.Sample(args.batch_size)
    batch = Transition(*zip(*transition))
    action_batch = Variable(torch.cat(batch.action))

    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
    non_final_next_states_t = torch.cat(tuple(s for s in batch.next_state if s is not None)).type(dtype)
    non_final_next_states = Variable(non_final_next_states_t, volatile = True)
    state_batch = Variable(torch.cat(batch.state)).float()
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    state_action_values = model(state_batch).gather(1, action_batch)
    next_state_values = Variable(torch.zeros(args.batch_size))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    target_state_action_values = args.gamma * next_state_values + reward_batch
    loss = loss_fn(state_action_values, target_state_action_values)
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)
    learning_rate *= args.lr_decay
    optimizer.zero_grad()
    loss.backward()
    #print("action", batch.action)
    #print("next_state", batch.next_state)
    #print("reward", reward_batch)
    #print("state_action_values", state_action_values)
    #print("target_state_action_values", target_state_action_values)
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def Train():
    R = []
    hundred_reward = 0
    solved = False
    for i_epi in range(args.max_episode):
        s = env.reset()
        s = np.reshape(s, (1, 4))
        s = torch.from_numpy(s)
        eps_r = 0
        t = 0
        while t < args.max_step:
            env.render()
            a = SelectAction(s, i_epi)
            next_s, reward, done, info = env.step(a[0, 0])
            next_s = torch.from_numpy(np.reshape(next_s, (1, 4)))
            if done:
                next_s = None
                for i in range(args.important_factor):
                    memory.Push(s, a, torch.Tensor([reward]), next_s)     
            memory.Push(s, a, torch.Tensor([reward]), next_s)      
            s = next_s
            eps_r += reward
            hundred_reward += reward
            t += 1
            if done or t == args.max_step - 1:
                OptModel()
                R.append(eps_r)
                print("Episode {}, acc avg reward {}".format(i_epi, sum(R) / (i_epi + 1) ))
                if i_epi % 100 == 0:
                    print("100 epi avg reward {}".format(hundred_reward / 100))
                    if hundred_reward >= 10000:
                        solved = True
                    hundred_reward = 0
                break
        if solved == True:
            break

def main():
    Train()

if __name__ == "__main__":
    main()
