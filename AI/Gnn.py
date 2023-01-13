import torch
import torch.nn as nn
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.loader as ld
from torch_geometric.nn import GCNConv, Sequential
import numpy as np
import pandas as pd
import os as os
import Classes.Board as Board
from statistics import mean



class Gnn():
    instance = None
    def __new__(cls, epochs=250, batch=16, learningRate=0.0001):
        if cls.instance is None:
            cls.instance = super(Gnn, cls).__new__(cls)
            cls.moves = None
            cls.epochs = epochs
            cls.batch = batch
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
        trainingDataset = MyDataset(train=True)
        testingDataset = MyDataset(train=False)

        lossFunction = nn.MSELoss()
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=cls.learningRate)
        trainingSetLoader = ld.DataLoader(trainingDataset, batch_size=cls.batch)
        testingSetLoader = ld.DataLoader(testingDataset)

        previousTestingLossMean = cls.test(testingSetLoader, lossFunction=lossFunction)
        print(f'Training loss: - Testing loss: {previousTestingLossMean}')

        counter = 0
        for epoch in range(cls.epochs):
            print('epoch: ', epoch+1, "/", cls.epochs)
            
            trainingLossMean = cls.train(trainingSetLoader, lossFunction, optimizer)
            testingLossMean = cls.test(testingSetLoader, lossFunction)
            print(f'Training loss: {trainingLossMean} Testing loss: {testingLossMean}')
            if testingLossMean < previousTestingLossMean:
                previousTestingLossMean = testingLossMean
                cls.saveWeights()
                counter = 0
            elif counter<2:
                counter+=1
            else:
                return
    
    def test(cls, loader, lossFunction):
        previousTestingLoss = []
        with torch.no_grad():
            for data in loader:
                graphs, globs, labels = data[0].to(cls.device), data[1].to(cls.device), data[2].to(cls.device)
                outputs = cls.model(graphs, globs, isTrain = False)
                loss = lossFunction(outputs, labels)
                previousTestingLoss.append(loss.item())
        return mean(previousTestingLoss)
    
    def train(cls, loader, lossFunction, optimizer):
        trainingLoss = []
        for data in loader:
            graphs, globs, labels = data[0].to(cls.device), data[1].to(cls.device), data[2].to(cls.device)
            optimizer.zero_grad()
            outputs = cls.model(graphs, globs, isTrain=True)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()
            trainingLoss.append(loss.item())
        return mean(trainingLoss)

    def evaluatePositionForPlayer(cls, player):
        globalFeats = player.globalFeaturesToDict()
        del globalFeats['player_id']
        graph = Batch.from_data_list([cls.fromDictsToGraph(Board.Board().placesToDict(player), Board.Board().edgesToDict(player)).to(cls.device)])
        glob = torch.tensor([list(globalFeats.values())[:-1]], dtype=torch.float, device=cls.device)
        return cls.model(graph, glob, isTrain=False).item()

    def fromDictsToGraph(cls, places, edges):
        w = torch.abs(torch.tensor(edges['is_owned_edge'], dtype=torch.float))
        x = torch.tensor(np.transpose(list(places.values())), dtype=torch.float)
        edge_index = torch.tensor([edges['place_1'],edges['place_2']], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=w)

    def saveWeights(cls):
        torch.save(cls.model.state_dict(), './AI/model_weights.pth')
        print("weights correctly updated.") 

class Net(nn.Module):
  def __init__(self, gnnInputDim, gnnHiddenDim, gnnOutputDim, globInputDim):
    super().__init__()
    self.Gnn = Sequential('x, edge_index, edge_attr', [
        (GCNConv(gnnInputDim, gnnHiddenDim), 'x, edge_index, edge_attr -> x'),
        nn.ReLU(inplace=True),
        (GCNConv(gnnHiddenDim, gnnHiddenDim), 'x, edge_index, edge_attr -> x'),
        nn.ReLU(inplace=True),
        (GCNConv(gnnHiddenDim, gnnOutputDim), 'x, edge_index, edge_attr -> x'),
        nn.ReLU(inplace=True)
    ])
    
    self.GlobalLayers = nn.Sequential(
        nn.Linear(globInputDim, 16),
        nn.ReLU(inplace=True),
        nn.Linear(16, 16),
        nn.ReLU(inplace=True),
        nn.Linear(16, globInputDim),
        nn.ReLU(inplace=True)
    )

    self.OutLayers = nn.Sequential(
        nn.Linear(54*gnnOutputDim+globInputDim, 85),
        nn.ReLU(inplace=True),
        nn.Linear(85, 1),
        nn.Sigmoid(),
    )

  def forward(self, graph, globalFeats, isTrain):
    batch_size, x, edge_index, edge_attr = graph.num_graphs, graph.x, graph.edge_index, graph.edge_attr
    
    embeds = self.Gnn(x, edge_index=edge_index, edge_attr=edge_attr)
    embeds = torch.reshape(embeds, (batch_size, 54*3))

    globalFeats = self.GlobalLayers(globalFeats)
    output = torch.cat([embeds, globalFeats], dim=-1)
    output = torch.dropout(output, p = 0.5, train = isTrain)
    output = self.OutLayers(output)
    return output

class MyDataset(torch.utils.data.IterableDataset):
    def __init__(self, train=False):
        super(MyDataset).__init__()
        if train:
            self.dataFrame = pd.read_json('./json/training_game.json')
        else:
            self.dataFrame = pd.read_json('./json/testing_game.json')

        self.data = [[self.extractInputFeaturesMove(i), self.extractGlobalFeatures(i), self.extractLabels(i)] for i in range(len(self.dataFrame))]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def extractInputFeaturesMove(self, moveIndex):
        places = self.dataFrame.iloc[moveIndex].places
        edges = self.dataFrame.iloc[moveIndex].edges
        return Gnn().fromDictsToGraph(places, edges)
    
    def extractGlobalFeatures(self, moveIdx):
        return torch.tensor(list(self.dataFrame.iloc[moveIdx].globals.values())[:-1], dtype=torch.float)

    def extractLabels(self, moveIdx):
        return torch.tensor([list(self.dataFrame.iloc[moveIdx].globals.values())[-1]], dtype=torch.float)
