import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from torch_geometric.nn import GCNConv, GraphConv, Sequential, global_mean_pool
import torch
import random
from torch_geometric.data import Data, Batch #aggiunto di recente, forse da togliere
from Classes.MoveTypes import TurnMoveTypes
#f
Transition = namedtuple('Transition', ('graph', 'glob', 'action', 'reward', 'next_graph', 'next_glob'))

class DQGNNagent():
    # def __init__(self, nInputs, nOutputs, criterion = torch.nn.SmoothL1Loss(), device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:
    def __init__(self, nInputs, nOutputs, eps, criterion = torch.nn.SmoothL1Loss(), device = torch.device("cpu")) -> None:
        # print("DQGNNAgent CONSTRUCTOR")
        self.BATCH_SIZE = 64 # 16  # 256
        self.GAMMA = 0.99
        self.EPS = eps
        self.TAU = 0.005 # 0.005
        self.LearningRate = 1e-3
        self.device = device
        self.policy_net = DQGNN(nInputs, 8, 4, 9, nOutputs).to(device)
        self.target_net = DQGNN(nInputs, 8, 4, 9, nOutputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LearningRate)
        self.memory = ReplayMemory(1000)

        # self.decay = 0.995
        self.decay = 0.999


    def epsDecay(self):
        self.EPS = self.EPS * self.decay
        # print("decay invoked.")
        if(self.EPS < 0.0001):
            self.EPS = 0

    def saveInMemory(self, graph, glob, action, reward, nextGraph, nextGlob): # qui è da inserire una transition
        if(reward != None):
            self.memory.push(graph, glob, torch.tensor([action]), torch.tensor([reward]), nextGraph, nextGlob)

    def selectMove(self, graph, glob, availableMoves):
        sample = random.random()
        if sample < self.EPS:
            action = self.explorationAction(availableMoves)
        else:
            action = self.greedyAction(graph, glob, availableMoves)
        return action
    
    def step(self, graph, glob, availableMoves):   
        action = self.selectMove(graph, glob, availableMoves) 
        if(self.EPS > 0.005):
            self.optimize_model()
            self.softUpdate()
        return action

    def greedyAction(self, graph, glob, availableMoves):
        with torch.no_grad():
            q_values = self.policy_net.forward(graph, glob) 
            valid_q_values = q_values[0][availableMoves]  
            # print("Print riga 62 DQGNN, valid_q_values: ", valid_q_values)
            max_q_value, max_index = valid_q_values.max(0) 
            action = availableMoves[max_index.item()]  
        # print("Greedy move!")
        return action

    def explorationAction(self, availableMoves):
        # print("Available moves: ", availableMoves)
        random_action = random.choice(availableMoves)
        return random_action

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        # print("RIGA 77 DQGNN: ", len(self.memory))
        print("optimizing...")
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions)) 
        graph_batch = Batch.from_data_list(batch.graph)
        glob_batch = torch.cat(batch.glob)
        reward_batch = torch.cat(batch.reward) # prendi tutti i rewards
        action_batch = torch.cat(batch.action) # prendi tutte le actions
        next_graph = Batch.from_data_list(batch.next_graph)
        next_glob = torch.cat(batch.next_glob)
        with torch.no_grad():
            expected_state_action_values = self.GAMMA * self.target_net.forward(next_graph, next_glob).max(1)[0] + reward_batch
        state_action_values = self.policy_net.forward(graph_batch, glob_batch)
        state_action_values = state_action_values.gather(1, action_batch.unsqueeze(1))
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

class DQGNN(nn.Module):
  def __init__(self, gnnInputDim, gnnHiddenDim, gnnOutputDim, globInputDim, nActions):
    super().__init__()

    self.Gnn = Sequential('x, edge_index, edge_attr', [
        (GraphConv(gnnInputDim, gnnHiddenDim), 'x, edge_index, edge_attr -> x'), nn.ReLU(inplace=True),
        (GraphConv(gnnHiddenDim, 4), 'x, edge_index, edge_attr -> x'), nn.ReLU(inplace=True), # (GCNConv(gnnHiddenDim, gnnOutputDim), 'x, edge_index, edge_attr -> x'), # nn.ReLU(inplace=True)
    ])
    
    self.GlobalLayers = nn.Sequential(
        nn.Linear(globInputDim, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, 8),
        nn.ReLU(inplace=True),
        nn.Linear(8, globInputDim),
        nn.ReLU(inplace=True)
    )

    self.OutLayers = nn.Sequential(
        nn.Linear(54*4+globInputDim, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, nActions)
    )
  
  def forward(self, graph, glob):
    embeds = self.Gnn(graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr)
    embeds = torch.reshape(embeds, (graph.num_graphs, 54 * 4))
    glob = self.GlobalLayers(glob)
    output = torch.cat([embeds, glob], dim=-1)
    output = self.OutLayers(output)
    return output
  
  def save_weights(self, filepath):
    torch.save(self.state_dict(), filepath)

  def load_weights(self, filepath):
    self.load_state_dict(torch.load(filepath))
  

