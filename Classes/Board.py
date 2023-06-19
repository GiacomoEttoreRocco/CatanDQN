import numpy as np
import Classes.CatanGraph as CatanGraph
import torch

from torch_geometric.data import Data, Batch

dictCsvResources = {None: 0, "desert": 0, "crop": 1, "iron": 2, "wood": 3, "clay": 4, "sheep": 5}
dictCsvHarbor = {"" : 0, "3:1" : 1, "2:1 crop" : 2, "2:1 iron" : 3, "2:1 wood" : 4, "2:1 clay" : 5, "2:1 sheep" : 6} 

class Board: 
    instance = None
    hardEdgeIndex = torch.tensor([[ 0,  0,  1,  2,  2,  3,  4,  4,  5,  6,  7,  7,  8,  9,  9, 10, 11, 11,
        12, 13, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 24,
        24, 25, 26, 27, 28, 28, 29, 30, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36,
        38, 39, 40, 41, 41, 42, 43, 43, 44, 45, 45, 39, 47, 48, 49, 50, 51, 52],
        [ 1,  8,  2,  3, 10,  4,  5, 12,  6, 14,  8, 17,  9, 10, 19, 11, 12, 21,
        13, 14, 23, 15, 25, 17, 27, 18, 19, 29, 20, 21, 31, 22, 23, 33, 24, 25,
        35, 26, 37, 28, 29, 38, 30, 31, 40, 32, 33, 42, 34, 35, 44, 36, 37, 46,
        39, 40, 41, 42, 49, 43, 44, 51, 45, 46, 53, 47, 48, 49, 50, 51, 52, 53]])
    
    def __new__(cls, doPlacement=True):
        if cls.instance is None: 
            cls.instance = super(Board, cls).__new__(cls)
            # cls.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            cls.device = 'cpu'
            cls.robberTile = 0
            cls.deck = ["knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight",
                        "victory_point","victory_point","victory_point","victory_point","victory_point","victory_point","victory_point","victory_point",
                        "year_of_plenty","year_of_plenty","monopoly","monopoly", "road_building","road_building"]
            #   SHUFFLE DECK
            cls.deck = np.random.permutation(cls.deck)
            cls.graph = CatanGraph.CatanGraph()
            cls.tiles = cls.graph.tiles
            cls.places = cls.graph.places
            cls.edges = cls.graph.edges
            cls.numbers = np.random.permutation(cls.graph.numbers)
            cls.resources = np.random.permutation(cls.graph.resources)
            cls.harbors = np.random.permutation(cls.graph.harbors)
            cls.EdgesOnTheSea = np.random.permutation(cls.graph.EdgesOnTheSea)

            if(doPlacement):
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
                if(t.resource != None and t.resource != "desert"):
                    cls.places[p].touchedResourses.append(t.resource)

    def __repr__(cls):
        s = ""
        for t in cls.tiles:
            s = s + "Tile: " + str(t) + "\n"
        for p in cls.places: 
            s = s + "Place: "+ str(p) + "\n"
        return s

    def dicesOfPlace(cls, place):
        numbers = []
        for tile in cls.tiles:
            if place.id in CatanGraph.tilePlaces[tile.identificator]:
                numbers.append(tile.number)
        if(len(numbers) < 1):
            return [0, 0, 0]
        elif(len(numbers) < 2):
            numbers.append(0)
            numbers.append(0)
        elif(len(numbers) < 3):
            numbers.append(0)
        return numbers

    def placesOnRobber(cls):
        return CatanGraph.tilePlaces[cls.robberTile]
    
    def idTileBlocked(cls, place):
        id = 1
        for tile in CatanGraph.placeTiles[place.id]:
            if(tile == cls.robberTile):
                return id
            else:
                id += 1
        return 0

    def placesToDict(cls, playerInTurn):
        data={'is_owned_place': [], 'type':[], 'resource_1':[],'dice_1':[],'resource_2':[],'dice_2':[],'resource_3':[],'dice_3':[], 'harbor':[]} #, 'robber_tile':[]}
    
        for p in cls.places:
            resourceBlockedId = cls.idTileBlocked(p)  
            data['is_owned_place'].append(p.ownedByThisPlayer(playerInTurn))
            data['type'].append(p.placeType()) 
            dices = cls.dicesOfPlace(p)

            if(len(p.touchedResourses) < 1):
                data['resource_1'].append(dictCsvResources[None])
                data['dice_1'].append(0) 
            else:
                if(resourceBlockedId != 0):
                    data['resource_1'].append(dictCsvResources[p.touchedResourses[0]])
                else:
                    data['resource_1'].append(dictCsvResources[None])
                data['dice_1'].append(dices[0])

            if(len(p.touchedResourses) < 2):
                data['resource_2'].append(dictCsvResources[None])
                data['dice_2'].append(0) 
            else:
                if(resourceBlockedId != 1):
                    data['resource_2'].append(dictCsvResources[p.touchedResourses[1]])
                else:
                    data['resource_2'].append(dictCsvResources[None])
                data['dice_2'].append(dices[1])

            if(len(p.touchedResourses) < 3):
                data['resource_3'].append(dictCsvResources[None])
                data['dice_3'].append(0)  
            else:
                if(resourceBlockedId != 2):
                    data['resource_3'].append(dictCsvResources[p.touchedResourses[2]])
                else:
                    data['resource_3'].append(dictCsvResources[None])
                data['dice_3'].append(dices[2])
            data['harbor'].append(dictCsvHarbor[p.harbor])
        return data

    def placesToTensor(cls, playerInTurn):
        is_owned_place = []
        place_type = []
        resource_1 = []
        dice_1 = []
        resource_2 = []
        dice_2 = []
        resource_3 = []
        dice_3 = []
        harbor = []
        for p in cls.places:
            resourceBlockedId = cls.idTileBlocked(p)  
            is_owned_place.append(p.ownedByThisPlayer(playerInTurn))
            place_type.append(p.placeType()) 
            dices = cls.dicesOfPlace(p)
            if len(p.touchedResourses) < 1:
                resource_1.append(dictCsvResources[None])
                dice_1.append(0) 
            else:
                if resourceBlockedId != 0:
                    resource_1.append(dictCsvResources[p.touchedResourses[0]])
                else:
                    resource_1.append(dictCsvResources[None])
                dice_1.append(dices[0])

            if len(p.touchedResourses) < 2:
                resource_2.append(dictCsvResources[None])
                dice_2.append(0) 
            else:
                if resourceBlockedId != 1:
                    resource_2.append(dictCsvResources[p.touchedResourses[1]])
                else:
                    resource_2.append(dictCsvResources[None])
                dice_2.append(dices[1])

            if len(p.touchedResourses) < 3:
                resource_3.append(dictCsvResources[None])
                dice_3.append(0)  
            else:
                if resourceBlockedId != 2:
                    resource_3.append(dictCsvResources[p.touchedResourses[2]])
                else:
                    resource_3.append(dictCsvResources[None])
                dice_3.append(dices[2])
            harbor.append(dictCsvHarbor[p.harbor])
        tensor = torch.Tensor([is_owned_place, place_type, resource_1, dice_1, resource_2, dice_2, resource_3, dice_3, harbor])
        return tensor.t()

    def placesStateTensor(cls, playerInTurn):
        ownedType = []
        resource_1 = []
        dice_1 = []
        underRobber1 = []
        resource_2 = []
        dice_2 = []
        underRobber2 = []
        resource_3 = []
        dice_3 = []
        underRobber3 = []
        harbor = []

        for p in cls.places:
            resourceBlockedId = cls.idTileBlocked(p)  
            owned = p.ownedByThisPlayer(playerInTurn)

            if p.isCity:
                ownedType.append(2 * owned) 
            elif p.isColony:
                ownedType.append(1 * owned) 
            else:
                ownedType.append(0 * owned)

            harbor.append(dictCsvHarbor[p.harbor])
            dices = cls.dicesOfPlace(p)

            if len(p.touchedResourses) < 1:
                resource_1.append(dictCsvResources[None])
                dice_1.append(0) 
                underRobber1.append(0)
            else:
                resource_1.append(dictCsvResources[p.touchedResourses[0]])

                if resourceBlockedId == 1:
                    underRobber1.append(1)
                else:
                    underRobber1.append(0)

                dice_1.append(dices[0])

            if len(p.touchedResourses) < 2:
                resource_2.append(dictCsvResources[None])
                dice_2.append(0) 
                underRobber2.append(0)
            else:
                resource_2.append(dictCsvResources[p.touchedResourses[1]])

                if resourceBlockedId == 2:
                    underRobber2.append(1)
                else:
                    underRobber2.append(0)

                dice_2.append(dices[1])

            if len(p.touchedResourses) < 3:
                resource_3.append(dictCsvResources[None])
                dice_3.append(0) 
                underRobber3.append(0)
            else:
                resource_3.append(dictCsvResources[p.touchedResourses[2]])

                if resourceBlockedId == 3:
                    underRobber3.append(1)
                else:
                    underRobber3.append(0)

                dice_3.append(dices[2])

        tensor = torch.Tensor([ownedType, resource_1, dice_1, underRobber1, resource_2, dice_2, underRobber2, resource_3, dice_3, underRobber3, harbor])
        # print("Riga 285 board: ", tensor)
        return tensor.t()

    def edgesToDict(cls, playerInTurn):
        data={'is_owned_edge': []}
        for edge in cls.edges.keys():
            if cls.edges[edge] == playerInTurn.id:
                data['is_owned_edge'].append(1)
            elif cls.edges[edge] == 0:
                data['is_owned_edge'].append(-1)
            else:
                data['is_owned_edge'].append(0)
        return data
    
    def edgesToTensor(cls, playerInTurn):
        is_owned_edge = []
        for edge, owner in cls.edges.items():
            if owner == playerInTurn.id:
                is_owned_edge.append(1)
            elif owner == 0:
                is_owned_edge.append(-1)
            else:
                is_owned_edge.append(0)
        tensor = torch.Tensor(is_owned_edge)
        return tensor
    
    def boardStateGraph(cls, player):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Definisci il dispositivo
        return Batch.from_data_list([Data(x=cls.placesStateTensor(player), edge_index= cls.hardEdgeIndex, edge_attr= cls.edgesToTensor(player)).to(cls.device)])
            
    def boardStateTensor(cls, player):
            places_state_tensor = cls.placesStateTensor(player).flatten()
            # print("Riga 317 Board: ", places_state_tensor)
            # edge_index = cls.hardEdgeIndex
            edges_tensor = cls.edgesToTensor(player)
            # print("Riga 320 Board: ", edges_tensor)
            # print("Riga 320 Board: ", len(edges_tensor))
            return torch.cat([places_state_tensor, edges_tensor], dim=0).to(cls.device)
    