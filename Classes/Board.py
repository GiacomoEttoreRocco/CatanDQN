import random
import numpy as np
import Classes.CatanGraph as CatanGraph

class Board: # deve diventare un singleton

    instance = None
    
    def __new__(cls, doPlacement=True):

        if cls.instance is None: 
            cls.instance = super(Board, cls).__new__(cls)
            cls.robberTile = 7
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

    def actualEvaluation(self):
        return 2


b = Board()
#for p in Board().places:
#    print(p.owner)
