from argparse import Action
from select import select
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Dqn(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Dqn, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.opt = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions  = self.fc3(x)

        return actions

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size = 100000,
    eps_end = 0.01, eps_dec = 0.01):
    
        self.gamma = gamma
        
        try:
            self.epsilon = float(open('/home/pedro/catkin_ws/src/tcc/scripts/weights/epsilon.txt','r').read()[:-1])
            print('O ultimo epsilon foi carregado, com valor de: ',self.epsilon)
        except:
            print('Epsilon não carregado')
            self.epsilon = epsilon

        self.eps_min  = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.fc1 = 256
        self.fc2 = 256
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = Dqn(self.lr, self.input_dims, self.fc1, self.fc2,self.n_actions)
        self.Q_target = Dqn(self.lr, self.input_dims, self.fc1, self.fc2,self.n_actions)
        
        try:
            self.Q_eval =T.load('/home/pedro/catkin_ws/src/tcc/scripts/weights/model_eval.pt')
            print('Pesos eval carregados!')
        except:
            print('Pesos eval não carregados!')

        try:
            self.Q_target =T.load('/home/pedro/catkin_ws/src/tcc/scripts/weights/model_targ.pt')
            print('Pesos target carregados!')
        except:
            print('Pesos target não carregados!')
        
        self.Q_target.eval()
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype =np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype =np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype = np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype =np.bool)

    
    def save_transitions(self,state ,action, reward, state_, done):
        i = self.mem_cntr % self.mem_size
       
        self.state_memory[i] = state
        self.new_state_memory[i] = state_
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.terminal_memory[i] = done
        self.mem_cntr +=1
    
    def get_an_action(self, state):
        if np.random.random()>self.epsilon:
            state = T.tensor(np.array([state]))
            action = T.argmax(self.Q_eval.forward(state)).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        if self.mem_cntr <self.batch_size:
            return

        self.Q_eval.opt.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem,self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype = np.int32)

        state_batch = T.tensor(self.state_memory[batch])
        new_state_batch = T.tensor(self.new_state_memory[batch])
        reward_batch = T.tensor(self.reward_memory[batch])
        terminal_batch = T.tensor(self.terminal_memory[batch])
        
        action_batch = self.action_memory[batch]
        
        

        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch+self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target,q_eval)
        loss.backward()
        self.Q_eval.opt.step()

        if self.epsilon>self.eps_min:
            self.epsilon = self.epsilon-self.eps_dec
        else:
            self.epsilon = self.eps_min

    def save(self):
        T.save(self.Q_eval, '/home/pedro/catkin_ws/src/tcc/scripts/weights/model_eval.pt')
        T.save(self.Q_target, '/home/pedro/catkin_ws/src/tcc/scripts/weights/model_targ.pt')
        np.savetxt('/home/pedro/catkin_ws/src/tcc/scripts/weights/epsilon.txt', [self.epsilon])

    def update_target(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        

    

