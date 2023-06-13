import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from torch_geometric.nn import GCNConv, GraphConv, Sequential, global_mean_pool
import torch
import random
#f
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class DQGNNagent():
    def __init__(self, nInputs, nOutputs, criterion = torch.nn.SmoothL1Loss(), device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")) -> None:

        self.BATCH_SIZE = 64 # 64 # 256
        self.GAMMA = 0.99
        self.fixed_EPS = 0.1
        self.TAU = 0.005 # 0.005
        self.LearningRate = 1e-3
        self.device = device
        self.previousState = None
        self.policy_net = DQGNN(nInputs, 8, 4, 9, nOutputs).to(device)
        self.target_net = DQGNN(nInputs, 8, 4, 9, nOutputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.criterion = criterion
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.LearningRate)
        self.memory = ReplayMemory(1000)
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
        action = self.selectMove(state) # la differenza sta nel fatto che può essere scelta la mossa random
        self.previousState = state
        self.previousAction = action
        if(self.fixed_EPS > 0.005):
            self.optimize_model()
            self.softUpdate()
        return action
    
    # def greedyAction(self, state, availableMoves):
    #     with torch.no_grad():
    #         action = self.policy_net(state).max(1)[1].view(1, 1)
    #     return action

    def greedyAction(self, state, availableMoves):
        with torch.no_grad():
            q_values = self.policy_net(state)  # Calcola i valori Q per tutte le azioni
            valid_q_values = q_values[0][availableMoves]  # Filtra i valori Q validi
            max_q_value, max_index = valid_q_values.max(0)  # Trova il massimo tra i valori Q validi
            action = availableMoves[max_index.item()]  # Seleziona l'azione corrispondente all'indice massimo
            # action = torch.tensor([[action]], dtype=torch.long)
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
            expected_state_action_values = self.GAMMA * self.target_net.forward(next_state).max(1)[0] + reward_batch # scelta progettuale: per il next_step, considerare tutte le mosse potenziali
        state_action_values = self.policy_net.forward(state_batch)
        state_action_values = state_action_values.gather(1, action_batch) # vengono selezionati gli elementi indicati da action_batch, nel tensore "state_action_values"
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

# class DQNetwork(nn.Module):
#     def __init__(self, observationLenght, n_actions):
#         super(DQNetwork, self).__init__()
#         self.layer1 = nn.Linear(observationLenght, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)
#     def forward(self, x):
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         return self.layer3(x)

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
        # nn.Linear(54*4+globInputDim, 128),
        # nn.ReLU(inplace=True),
        # nn.Linear(128, 1),
        # nn.Sigmoid()
        nn.Linear(54*4, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, nActions)
    )

    def forward(self, graph, globalFeats, isTrain):
        batch_size, batch, x, edge_index, edge_attr = graph.num_graphs, graph.batch, graph.x, graph.edge_index, graph.edge_attr
        embeds = self.Gnn(x, edge_index=edge_index, edge_attr=edge_attr)
        embeds = torch.reshape(embeds, (batch_size, 54*4))
        globalFeats = self.GlobalLayers(globalFeats)
        output = torch.cat([embeds, globalFeats], dim=-1)
        output = torch.dropout(output, p = 0.2, train = isTrain)
        output = self.OutLayers(output)
        return output
