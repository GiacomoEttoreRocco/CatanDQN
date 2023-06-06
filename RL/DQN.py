import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import torch
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class DQNagent():
    def __init__(self, nInputs, nOutputs, criterion, device) -> None:
        self.BATCH_SIZE = 64 # 64 # 256
        self.GAMMA = 0.99
        self.fixed_EPS = 0.1
        self.TAU = 0.005 # 0.005
        self.LR = 1e-3
        self.device = device
        self.previousState = None
        self.policy_net = DQNetwork(nInputs, nOutputs).to(device)
        self.target_net = DQNetwork(nInputs, nOutputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayMemory(10000)
        self.previousState = None
        self.previousAction = None

        self.decay = 0.9

    def epsDecay(self):
        self.fixed_EPS = self.fixed_EPS * self.decay
        if(self.fixed_EPS < 0.0001):
            self.fixed_EPS = 0

    def saveInMemory(self, state, action, reward, nextState): # qui è da inserire una transition
        if(reward != None):
            self.memory.push(state, action, reward, nextState)

    def selectMove(self, state):
        sample = random.random()
        if sample < self.fixed_EPS:
            action = self.explorationAction()
        else:
            action = self.greedyAction(state)
        return action
    
    def step(self, state, previousReward):                                                 
        self.saveInMemory(self.previousState, self.previousAction, previousReward, state)
        action = self.selectMove(state)
        self.previousState = state
        self.previousAction = action
        if(self.fixed_EPS > 0.005):
            self.optimize_model()
            self.softUpdate()
        return action
    
    def greedyAction(self, state):
        with torch.no_grad():
            action = self.policy_net(state).max(1)[1].view(1, 1)
        return action

    def explorationAction(self):
        random_action = random.randint(0, 1)
        return torch.tensor([[random_action]], dtype=torch.long, device=self.device)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions)) # batch.state mi passa un array di tutti gli stati
        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward) # prendi tutti i rewards
        action_batch = torch.cat(batch.action) # prendi tutte le actions
        next_state = torch.cat(batch.next_state)  # ...

        with torch.no_grad():
            expected_state_action_values = self.GAMMA * self.target_net.forward(next_state).max(1)[0] + reward_batch
        state_action_values = self.policy_net.forward(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)
        self.optimizer.zero_grad()
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        #print("Loss: ", loss)
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

class DQNetwork(nn.Module):
    def __init__(self, observationLenght, n_actions):
        super(DQNetwork, self).__init__()
        self.layer1 = nn.Linear(observationLenght, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
