import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.loader as ld
from torch_geometric.nn import GCNConv, GraphConv, Sequential, global_mean_pool
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
            # cls.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            cls.device = 'cpu'
            cls.model = Net(9, 8, 4, 9).to(cls.device) # 9,8,3,9
            # cls.modelWeightsPath = "AI/best_model_weights.pth"
            cls.modelWeightsPath = "AI/best_model_SL.pth"

            if os.path.exists(cls.modelWeightsPath):
                cls.model.load_state_dict(torch.load(cls.modelWeightsPath, map_location=cls.device))
                print('Weights loaded..')
        return cls.instance

    def reset(cls):
        cls.instance = None

    def trainModel(cls, validate = False):
        cls.loadWeights()
        trainingDataset = MyDataset(train=True)
        testingDataset = MyDataset(train=False)
        lossFunction = nn.MSELoss()
        optimizer = torch.optim.Adam(cls.model.parameters(), lr=cls.learningRate)
        trainingSetLoader = ld.DataLoader(trainingDataset, batch_size=cls.batch)
        testingSetLoader = ld.DataLoader(testingDataset, batch_size=cls.batch)
        previousTestingLossMean = cls.test(testingSetLoader, lossFunction=lossFunction)
        actualModelWeights = cls.model.state_dict()
        print(f'Training loss: - Testing loss: {previousTestingLossMean}')
        counter = 0
        for epoch in range(cls.epochs):
            print('epoch: ', epoch+1, "/", cls.epochs)
            trainingLossMean = cls.train(trainingSetLoader, lossFunction, optimizer)
            testingLossMean = cls.test(testingSetLoader, lossFunction)
            print(f'Training loss: {trainingLossMean} Testing loss: {testingLossMean}')
            if testingLossMean < previousTestingLossMean:
                previousTestingLossMean = testingLossMean
                actualModelWeights = cls.model.state_dict()
                counter = 0
            elif counter<2:
                counter+=1
            else:
                cls.saveWeights(actualModelWeights)
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

    # def evaluatePositionForPlayer(cls, player):
    #     globalFeats = player.globalFeaturesToDict()
    #     del globalFeats['player_id'] # rimuove la colonna player_id che è inutile
    #     graph = Batch.from_data_list([cls.fromDictsToGraph(Board.Board().placesToDict(player), Board.Board().edgesToDict(player)).to(cls.device)])
    #     # print("Graph gnn, riga 88 GNN: ", graph)
    #     # DataBatch(x=[54, 9], edge_index=[2, 72], edge_attr=[72], batch=[54], ptr=[2])
    #     glob = torch.tensor([list(globalFeats.values())[:-1]], dtype=torch.float, device=cls.device)
    #     return cls.model(graph, glob, isTrain=False).item()
    
    # def fromDictsToGraph(cls, places, edges):
    #     w = torch.tensor(edges['is_owned_edge'], dtype=torch.float)
    #     x = torch.tensor(np.transpose(list(places.values())), dtype=torch.float)
    #     return Data(x=x, edge_index=Board.Board().hardEdgeIndex, edge_attr=w)

    def evaluatePositionForPlayer(cls, player):
        graph = Batch.from_data_list([Data(x=Board.Board().placesToTensor(player), edge_index=Board.Board().hardEdgeIndex, edge_attr=Board.Board().edgesToTensor(player)).to(cls.device)])
        glob = player.globalFeaturesToTensor()
        return cls.model(graph, glob, isTrain=False).item()

    def saveWeights(cls, weights):
        torch.save(weights, cls.modelWeightsPath)
        print("weights correctly updated.") 
    
    def loadWeights(cls):
        if os.path.exists(cls.modelWeightsPath):
            cls.model.load_state_dict(torch.load(cls.modelWeightsPath, map_location=cls.device))
            print('Weights loaded..')

class Net(nn.Module):
  def __init__(self, gnnInputDim, gnnHiddenDim, gnnOutputDim, globInputDim):
    # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    self.device = 'cpu'
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
        # nn.Linear(54*gnnOutputDim+globInputDim, 85),
        nn.Linear(54*4+globInputDim, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

#   def forward(self, graph, globalFeats, isTrain):
#     batch_size, batch, x, edge_index, edge_attr = graph.num_graphs, graph.batch, graph.x, graph.edge_index, graph.edge_attr
#     embeds = self.Gnn(x, edge_index=edge_index, edge_attr=edge_attr)
#     embeds = torch.reshape(embeds, (batch_size, 54*4))
#     globalFeats = self.GlobalLayers(globalFeats)
#     output = torch.cat([embeds, globalFeats], dim=-1)
#     output = torch.dropout(output, p = 0.2, train = isTrain)
#     output = self.OutLayers(output)
#     return output
  
  def forward(self, graph, globalFeats, isTrain): 
    batch_size, batch, x, edge_index, edge_attr = graph.num_graphs, graph.batch, graph.x, graph.edge_index, graph.edge_attr
    embeds = self.Gnn(x.to(self.device), edge_index=edge_index.to(self.device), edge_attr=edge_attr.to(self.device))
    embeds = torch.reshape(embeds, (batch_size, 54 * 4))
    globalFeats = self.GlobalLayers(globalFeats.to(self.device))
    output = torch.cat([embeds, globalFeats], dim=-1)
    output = torch.dropout(output, p=0.2, train=isTrain)
    output = self.OutLayers(output.to(self.device))
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
        np.random.shuffle(self.data)
        return iter(self.data)

    def __len__(self):
        return len(self.data)
    
    def extractInputFeaturesMove(self, moveIndex):
        places = self.dataFrame.iloc[moveIndex].places
        edges = self.dataFrame.iloc[moveIndex].edges
        return Gnn().fromDictsToGraph(places, edges)
    
    def extractGlobalFeatures(self, moveIdx):
        return torch.tensor(list(self.dataFrame.iloc[moveIdx].globals.values())[:-1], dtype=torch.float)

    def extractLabels(self, moveIdx):
        return torch.tensor([list(self.dataFrame.iloc[moveIdx].globals.values())[-1]], dtype=torch.float)
