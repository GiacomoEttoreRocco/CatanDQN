import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from torch_geometric.nn import GCNConv, GraphConv, Sequential, global_mean_pool
import torch
import random
from torch_geometric.data import Data, Batch #aggiunto di recente, forse da togliere
from Classes.MoveTypes import TurnMoveTypes
#f
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class L2DQNagent():
    # def __init__(self, nInputs, nOutputs, criterion = torch.nn.SmoothL1Loss(), device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
    def __init__(self, name, nInputs, nOutputs, eps, criterion = torch.nn.SmoothL1Loss(), device = torch.device("cpu")) -> None:
        # print("DQNgent CONSTRUCTOR")
        self.name = name # per ora non serve a un cazzo, serve in caso di debug
        self.BATCH_SIZE = 16 
        self.GAMMA = 0.99
        self.EPS = eps
        self.TAU = 0.005 
        self.LearningRate = 1e-3
        self.device = device

        self.policy_net = DQN(nInputs, nOutputs).to(device) # 54*11+72
        self.target_net = DQN(nInputs, nOutputs).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LearningRate)
        self.memory = ReplayMemory(1000)

        self.decay = 0.99

    def epsDecay(self):
        self.EPS = self.EPS * self.decay
        # print("decay invoked.")
        if(self.EPS < 0.0001):
            self.EPS = 0

    def saveInMemory(self, state, action, reward, nextState): # qui è da inserire una transition
        if(reward != None):
            self.memory.push(state, torch.tensor([action]), torch.tensor([reward]), nextState)

    def selectMove(self, state, availableMoves):
        sample = random.random()
        if sample < self.EPS:
            action = self.explorationAction(availableMoves)
        else:
            action = self.greedyAction(state, availableMoves)
        return action
    
    def step(self, state, availableMoves, fatherDQN):   
        action = self.selectMove(state, availableMoves) 
        if(self.EPS > 0.005):
            self.optimize_model(fatherDQN)
            self.softUpdate()
        return action

    def greedyAction(self, state, availableMoves):
        with torch.no_grad():
            q_values = self.policy_net.forward(state) 
            valid_q_values = q_values[0][availableMoves]  
            max_q_value, max_index = valid_q_values.max(0) 
            action = availableMoves[max_index.item()]  
        return action

    def explorationAction(self, availableMoves):
        random_action = random.choice(availableMoves)
        return random_action

    def optimize_model(self, fatherDQN):
        print("optimizing")
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions)) 
        state_batch = torch.cat(batch.state)
        # print("Riga 76, DQN: ", state_batch.size())
        reward_batch = torch.cat(batch.reward)  # prendi tutti i rewards
        # print("Riga 78, DQN: ", reward_batch.size())
        action_batch = torch.cat(batch.action)  # prendi tutte le actions
        # print("Riga 80, DQN: ", action_batch.unsqueeze(1).size())
        action_batch = action_batch.unsqueeze(1)

        next_state = torch.cat(batch.next_state)

        with torch.no_grad():
            expected_state_action_values = self.GAMMA * (self.target_net.forward(next_state).max(1)[0]* fatherDQN.target_net.forward(next_state).max(1)[0]) + reward_batch
            # print("Dimensione di expected_state_action_values:", expected_state_action_values.size()) # 16 x 16 

        state_action_values = self.policy_net.forward(state_batch).gather(1, action_batch)
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def softUpdate(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

class ReplayMemory():
    def __init__(self, capacity=1000) -> None:
        self.memory = deque([], capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, observationLenght, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(observationLenght, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath))
  

