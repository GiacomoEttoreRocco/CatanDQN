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
from statistics import mean



class Gnn():
    instance = None
    def __new__(cls, epochs=250, learningRate=0.0001): # precedente 0.03
        if cls.instance is None:
            cls.instance = super(Gnn, cls).__new__(cls)
            cls.moves = None
            cls.epochs = epochs
            cls.learningRate = learningRate
            cls.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            cls.model = Net(9, 8, 3, 9).to(cls.device)
            if os.path.exists('./AI/model_weights.pth'):
                cls.model.load_state_dict(torch.load('./AI/model_weights.pth', map_location=cls.device))
                print('Weights loaded..')
        return cls.instance

    def trainModel(cls, validate = False):
        if os.path.exists('./AI/model_weights.pth'):
            cls.model.load_state_dict(torch.load('./AI/model_weights.pth', map_location=cls.device))
            print('Weights loaded..')
        trainingDataFrame = pd.read_json('./json/training_game.json')
        testingDataFrame = pd.read_json('./json/testing_game.json')
        trainingSetLength = len(trainingDataFrame)
        testingSetLength = len(testingDataFrame)
        cls.moves = pd.concat([trainingDataFrame, testingDataFrame], ignore_index=True)
        lossFunction = nn.MSELoss()
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=cls.learningRate)
        trainingSet = np.random.permutation([x for x in range(trainingSetLength)])
        testingSet = np.random.permutation([x for x in range(trainingSetLength, trainingSetLength+testingSetLength)])

        previousTestingLoss = []
        with torch.no_grad():
            for i, idx in enumerate(testingSet):
                g = cls.extractInputFeaturesMove(idx).to(cls.device)
                glob = torch.tensor(list(cls.moves.iloc[idx].globals.values())[:-1]).to(cls.device).float()
                labels = torch.tensor([list(cls.moves.iloc[idx].globals.values())[-1]], device=cls.device).float()
                outputs = cls.model(g, glob, isTrain = False)
                loss = lossFunction(outputs, labels)
                previousTestingLoss.append(loss.item())
        previousTestingLossMean = mean(previousTestingLoss)
        print(f'Training loss: - Testing loss: {previousTestingLossMean}')

        counter = 0
        for epoch in range(cls.epochs):
            trainingLoss = []
            print('epoch: ', epoch+1, "/", cls.epochs)
            for i, idx in enumerate(trainingSet):
                g = cls.extractInputFeaturesMove(idx).to(cls.device)
                glob = torch.tensor(list(cls.moves.iloc[idx].globals.values())[:-1]).to(cls.device).float()
                labels = torch.tensor([list(cls.moves.iloc[idx].globals.values())[-1]], device=cls.device).float()
                optimizer.zero_grad()
                outputs = cls.model(g, glob, isTrain=True)
                loss = lossFunction(outputs, labels)
                loss.backward()
                optimizer.step()
                trainingLoss.append(loss.item())
            if validate:
                testingLoss = []
                with torch.no_grad():
                    for i, idx in enumerate(testingSet):
                        g = cls.extractInputFeaturesMove(idx).to(cls.device)
                        glob = torch.tensor(list(cls.moves.iloc[idx].globals.values())[:-1]).to(cls.device).float()
                        labels = torch.tensor([list(cls.moves.iloc[idx].globals.values())[-1]], device=cls.device).float()
                        outputs = cls.model(g, glob, isTrain = False)
                        loss = lossFunction(outputs, labels)
                        testingLoss.append(loss.item())
                
                testingLossMean = mean(testingLoss)
                print(f'Training loss: {mean(trainingLoss)} Testing loss: {mean(testingLoss)}')
                if testingLossMean < previousTestingLossMean:
                    previousTestingLossMean = testingLossMean
                    cls.saveWeights()
                    counter = 0
                elif counter<2:
                    counter+=1
                else:
                    return
            else:
                cls.saveWeights()

    def evaluatePositionForPlayer(cls, player):
        globalFeats = player.globalFeaturesToDict()
        del globalFeats['player_id']
        graph = cls.fromDictsToGraph(Board.Board().placesToDict(player), Board.Board().edgesToDict(player)).to(cls.device)
        glob = torch.tensor(list(globalFeats.values())[:-1]).float().to(cls.device)
        return cls.model(graph, glob, isTrain=False).item()

    # def evaluatePosition(cls, player):
    #     globalFeats = player.globalFeaturesToDict()
    #     del globalFeats['player_id']
    #     graph = cls.fromDictsToGraph(Board.Board().placesToDict(player), Board.Board().edgesToDict(player)).to(cls.device)
    #     glob = torch.tensor(list(globalFeats.values())[:-1]).float().to(cls.device)
    #     return cls.model(graph, glob)

    def extractInputFeaturesMove(cls, moveIndex):
        places = cls.moves.iloc[moveIndex].places
        edges = cls.moves.iloc[moveIndex].edges
        return cls.fromDictsToGraph(places, edges)

    def fromDictsToGraph(cls, places, edges):
        u = torch.tensor(edges['place_1'])
        v = torch.tensor(edges['place_2'])
        w = torch.tensor(edges['is_owned_edge'])
        g = dgl.graph((torch.cat([u, v], dim=0) , torch.cat([v, u], dim=0)))
        g.edata['weight'] = torch.cat([w, w], dim=0).float()
        g.ndata['feat'] = torch.tensor(np.transpose(list(places.values()))).float()
        return g

    def saveWeights(cls):
        torch.save(cls.model.state_dict(), './AI/model_weights.pth')
        print("weights correctly updated.") 

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
    self.OutputLayer2 = nn.Linear(85, 1)

  def forward(self, graph, globalFeats, isTrain):
    graph.ndata['feat'] = F.relu(self.GNN1(graph, graph.ndata['feat']))
    graph.ndata['feat'] = F.relu(self.GNN2(graph, graph.ndata['feat']))
    embeds = F.relu(self.GNN3(graph, graph.ndata['feat']))
    for i, x in enumerate(embeds):
        print("Place: ", i, x)
    embeds = torch.flatten(embeds)
    
    globalFeats = F.relu(self.GlobalLayer1(globalFeats))
    globalFeats = F.relu(self.GlobalLayer2(globalFeats))
    globalFeats = F.relu(self.GlobalLayer3(globalFeats))
    output = torch.cat([embeds, globalFeats])
    output = torch.dropout(output, p = 0.33, train = isTrain)
    output = F.relu(self.OutputLayer1(output))
    output = self.OutputLayer2(output)
    return torch.sigmoid(output)
        