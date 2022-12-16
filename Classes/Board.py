import random
import numpy as np
import Classes.CatanGraph as CatanGraph
import csv
from csv import QUOTE_NONE
import pandas as pd

dictCsvResources = {None: -2, "desert": -1, "crop": 0, "iron": 1, "wood": 2, "clay": 3, "sheep": 4}
dictCsvHarbor = {"" : 0, "3:1" : 1, "2:1 crop" : 2, "2:1 iron" : 3, "2:1 wood" : 4, "2:1 clay" : 5, "2:1 sheep" : 6}


class Board: # deve diventare un singleton

    instance = None
    
    def __new__(cls, doPlacement=True):

        if cls.instance is None: 
            cls.instance = super(Board, cls).__new__(cls)
            cls.robberTile = 0
            cls.deck = ["knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight",
                        "victory_point","victory_point","victory_point","victory_point","victory_point",
                        "year_of_plenty","year_of_plenty","monopoly","monopoly", "road_building","road_building"]
            #   SHUFFLE DECK
            cls.deck = np.random.permutation(cls.deck)

            cls.graph = CatanGraph.CatanGraph()
            cls.tiles = cls.graph.tiles
            cls.places = cls.graph.places
            cls.edges = cls.graph.edges

            #   PERMUTATIONS: 
            cls.numbers = np.random.permutation(cls.graph.numbers)
            cls.resources = np.random.permutation(cls.graph.resources)
            cls.harbors = np.random.permutation(cls.graph.harbors)
            cls.EdgesOnTheSea = np.random.permutation(cls.graph.EdgesOnTheSea)

            if(doPlacement):
                print("\n Tiles placement...\n")
                cls.tilesPlacement(cls)

        return cls.instance

    def reset(cls):
        Board.instance = None

    def availableForHarbor(cls, edge):
        p1 = edge[0]
        p2 = edge[1]
        for pAdj in cls.graph.listOfAdj[p1]:
            if(cls.places[pAdj].harbor != ""):
                return False
        for pAdj in cls.graph.listOfAdj[p2]:
            if(cls.places[pAdj].harbor != ""):
                return False
        return True

    def chooseTileHarbor(cls):
        #for harbor in cls.harbors:
        #    print(harbor)
        i = 0
        for edge in cls.EdgesOnTheSea:
            if(cls.availableForHarbor(cls, edge) and i < len(cls.harbors)):
                harbor = cls.harbors[i]
                cls.places[edge[0]].harbor = harbor
                cls.places[edge[1]].harbor = harbor
                i += 1

    def tilesPlacement(cls):
        number_index = 0
        for index, res in enumerate(cls.resources): 
            if(res == "desert"):
                cls.robberTile = index
                tile = CatanGraph.Tile(res, 7, index)
                cls.tiles.append(tile)
            else:
                tile = CatanGraph.Tile(res, cls.numbers[number_index], index)
                number_index = number_index+1
                cls.tiles.append(tile)

        cls.chooseTileHarbor(cls)

        for t in cls.tiles:
            for p in t.associatedPlaces:
                cls.places[p].touchedResourses.append(t.resource)

    def __repr__(cls):
        s = ""
        for t in cls.tiles:
            s = s + "Tile: " + str(t) + "\n"
        for p in cls.places: 
            s = s + "Place: "+ str(p) + "\n"
        return s

    def actualEvaluation(cls):
        return 2

    def dicesOfPlace(cls, place):
        numbers = []
        for tile in cls.tiles:
            if place.id in CatanGraph.tilePlaces[tile.identificator]:
                numbers.append(tile.number)
        if(len(numbers) < 1):
            return [-1, -1, -1]
        elif(len(numbers) < 2):
            numbers.append(-1)
            numbers.append(-1)
        elif(len(numbers) < 3):
            numbers.append(-1)
        return numbers

    def robberOfPlace(cls, place):
        number = 0
        for tile in cls.tiles:
            if place.id in CatanGraph.tilePlaces[tile.identificator]:
                if(cls.robberTile == tile.identificator):
                    return number
                else:
                    number += 1
        return number

###########################################################################################################################################################################################################################

    def placesToDict(cls) :
        data={'id':[], 'place_owner':[], 'type':[], 'resource_1':[],'dice_1':[],'resource_2':[],'dice_2':[],'resource_3':[],'dice_3':[], 'harbor':[], 'robber_tile':[]}
        for p in cls.places:
            data['id'].append(p.id)
            data['place_owner'].append(p.owner)
            if(p.isCity):
                data['type'].append(2)
            elif(p.isColony):
                data['type'].append(1)
            else:
                data['type'].append(0)
            
            dices = cls.dicesOfPlace(p)
            if(len(p.touchedResourses) < 1):
                data['resource_1'].append(dictCsvResources[None])
                data['dice_1'].append(-1)
            else:
                data['resource_1'].append(dictCsvResources[p.touchedResourses[0]])
                data['dice_1'].append(dices[0])

            if(len(p.touchedResourses) < 2):
                data['resource_2'].append(dictCsvResources[None])
                data['dice_2'].append(-1)
            else:
                data['resource_2'].append(dictCsvResources[p.touchedResourses[1]])
                data['dice_2'].append(dices[1])

            if(len(p.touchedResourses) < 3):
                data['resource_3'].append(dictCsvResources[None])
                data['dice_3'].append(-1)
            else:
                data['resource_3'].append(dictCsvResources[p.touchedResourses[2]])
                data['dice_3'].append(dices[2])
            data['harbor'].append(dictCsvHarbor[p.harbor])
            data['robber_tile'].append(cls.robberOfPlace(p))    
        return data

    def edgesToDict(cls):
        data={'place_1':[],'place_2':[], 'edge_owner':[]}
        for edge in cls.edges.keys():
            data['place_1'].append(edge[0])
            data['place_2'].append(edge[1])
            data['edge_owner'].append(cls.edges[edge])
        return data
            
# Nodes: 

# ID: 		{0,...,53}
# Owner: 		one hot econding: {0000,1000,0100,0010,0001}
# Type: 		{Nothing : 0, Colony: 1, City: 2}
# ResTile1: 	{None: -1, Crop: 0, Iron: 1, Wood: 2, Clay: 3, Sheep: 4} 
# DiceTile1: 	{None: -1, 2,3,4,5,6,8,9,10,11,12} 
# ResTile2: 	{None: -1, Crop: 0, Iron: 1, Wood: 2, Clay: 3, Sheep: 4}
# DiceTile2: 	{None: -1, 2,3,4,5,6,8,9,10,11,12}
# ResTile3: 	{None: -1, Crop: 0, Iron: 1, Wood: 2, Clay: 3, Sheep: 4}
# DiceTile3: 	{None: -1, 2,3,4,5,6,8,9,10,11,12}
# Harbor: {None: 0, Harbor31: 1, Harbor21Crop: 2, Harbor21Iron: 3, Harbor21Wood: 4, Harbor21Clay: 5, Harbor21Sheep: 6}
# RobberTile: 	{No: 0, Tile1: 1, Tile2: 2, Tile3: 3}

#Altra opzione:

# Harbor31: 	{0, 1}
# Harbor21C: 	{0, 1}
# Harbor21I: 	{0, 1}
# Harbor21W: 	{0, 1}
# Harbor21CL: 	{0, 1}
# Harbro21S: 	{0, 1}

