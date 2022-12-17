import torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.nn import GraphConv
import numpy as np
import pandas as pd
import os as os
import Classes.Board as Board



class Gnn():
    instance = None
    def __new__(cls, epochs=10, learningRate=0.001):
        if cls.instance is None:
            cls.instance = super(Gnn, cls).__new__(cls)
            cls.moves = None
            cls.epochs = epochs
            cls.learningRate = learningRate
            cls.model = Net(11, 8, 3, 8)
            if os.path.exists('./AI/model_weights.pth'):
                #cls.model.load_state_dict(torch.load('./AI/model_weights.pth')
                cls.model.load_state_dict(torch.load('./AI/model_weights.pth', map_location=torch.device('cpu')))
                print('Weights loaded..')
                
        return cls.instance

    def trainModel(cls):
        cls.moves = pd.read_json('./json/game.json')
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=cls.learningRate)
        permutationIndexMoves = np.random.permutation([x for x in range(len(cls.moves))])
        for epoch in range(cls.epochs):
            print('epoch: ', epoch+1)
            for i, idx in enumerate(permutationIndexMoves):
                g = cls.extractInputFeaturesMove(cls.moves, idx)
                glob = torch.tensor(list(cls.moves.iloc[idx].globals.values())[:-1]).float()
                labels = torch.tensor(list(cls.moves.iloc[idx].globals.values())[-1])-1
                optimizer.zero_grad()
                outputs = cls.model(g, glob)
                outputs = loss(outputs, labels)
                outputs.backward()
                optimizer.step()
        
        cls.saveWeights()           
    

    def evaluatePosition(cls, player):
        globalFeats = player.globalFeaturesToDict()
        graph = cls.fromDictsToGraph(Board.Board().placesToDict(), Board.Board().edgesToDict())
        glob = torch.tensor(list(globalFeats.values())[:-1]).float()
        
        return cls.model(graph, glob)[globalFeats['player_id']-1]

    def extractInputFeaturesMove(cls, moveIndex):
        places = cls.moves.iloc[moveIndex].places
        edges = cls.moves.iloc[moveIndex].edges
        return cls.fromDictsToGraph(places, edges)

    def fromDictsToGraph(cls, places, edges):
        u = torch.tensor(edges['place_1'])
        v = torch.tensor(edges['place_2'])
        w = torch.tensor(edges['edge_owner'])
        g = dgl.graph((torch.cat([u, v], dim=0) , torch.cat([v, u], dim=0)))
        g.edata['weight'] = torch.cat([w, w], dim=0).float()
        g.ndata['feat'] = torch.tensor(np.transpose(list(places.values()))).float()
        return g

    def saveWeights(cls):
        torch.save(cls.model.state_dict(), './AI/model_weights.pth')

class Net(nn.Module):
  def __init__(self, gnnInputDim, gnnHiddenDim, gnnOutputDim, globInputDim):
    super().__init__()
    self.GNN1 = GraphConv(gnnInputDim, gnnHiddenDim)
    self.GNN2 = GraphConv(gnnHiddenDim, gnnHiddenDim)
    self.GNN3 = GraphConv(gnnHiddenDim, gnnOutputDim)

    self.GlobalLayer1 = nn.Linear(globInputDim, 16)
    self.GlobalLayer2 = nn.Linear(16, 16)
    self.GlobalLayer3 = nn.Linear(16, globInputDim)


    self.OutputLayer1 = nn.Linear(54*gnnOutputDim+globInputDim, 85)
    self.OutputLayer2 = nn.Linear(85, 4)

  def forward(self, graph, globalFeats):
    graph.ndata['feat'] = F.relu(self.GNN1(graph, graph.ndata['feat']))
    graph.ndata['feat'] = F.relu(self.GNN2(graph, graph.ndata['feat']))
    embeds = F.relu(self.GNN3(graph, graph.ndata['feat']))
    embeds = torch.flatten(embeds)
    
    globalFeats = F.relu(self.GlobalLayer1(globalFeats))
    globalFeats = F.relu(self.GlobalLayer2(globalFeats))
    globalFeats = F.relu(self.GlobalLayer3(globalFeats))


    output = F.relu(self.OutputLayer1(torch.cat([embeds, globalFeats])))
    output = self.OutputLayer2(output)
    return nn.Softmax(dim=0)(output)
        