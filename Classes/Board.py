import random
import numpy as np
import CatanGraph as cg

class Board: # deve diventare un singleton

    instance = None
    
    def __new__(cls, doPlacement=True):

        if cls.instance is None: 
            cls.instance = super(Board, cls).__new__(cls)
            cls.robberTile = 7
            cls.deck = ["knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight","knight",
                        "victory_point","victory_point","victory_point","victory_point","victory_point",
                        "road_building","road_building",
                        "year_of_plenty","year_of_plenty",
                        "monopoly","monopoly"]
            #   SHUFFLE DECK
            cls.deck = np.random.permutation(cls.deck)

            cls.graph = cg.CatanGraph()
            cls.tiles = cls.graph.tiles
            cls.places = cls.graph.places
            cls.edges = cls.graph.edges

            #   PERMUTATIONS: 
            cls.numbers = np.random.permutation(cls.graph.numbers)
            cls.resources = np.random.permutation(cls.graph.resources)
            cls.harbors = np.random.permutation(cls.graph.harbors)
            cls.tilesOnTheSea = np.random.permutation(cls.graph.tilesOnTheSea)

            if(doPlacement):
                print("\n Tiles placement...\n")
                cls.tilesPlacement(cls)

        return cls.instance

    def tilesPlacement(cls):
        number_index = 0
        for index, res in enumerate(cls.resources): 
            if(res == "desert"):
                tile = cg.Tile(res, 7, index)
                cls.tiles.append(tile)
            else:
                tile = cg.Tile(res, cls.numbers[number_index], index)
                number_index = number_index+1
                cls.tiles.append(tile)

        alreadyHarborPlaceInEdge = []

        for i in range(0, 9):
            alreadyHarborPlaceInEdge = cls.chooseTileHarbor(cls, alreadyHarborPlaceInEdge, i)

        for t in cls.tiles:
            for p in t.associated_places:
                cls.places[p].touchedResourses.append(t.resource)

    def chooseTileHarbor(cls, ahpe, i):
        tos = cls.tilesOnTheSea[random.randint(0, len(cls.tilesOnTheSea)-1)]
        edges = []
        for e in cls.graph.CoupleOfPlaceOnTheSea[tos]: 
            edges.append(e)
        choosenEdge = edges[random.randint(0, len(edges)-1)]
        if(cls.edgeAlreadyHarbor(cls, choosenEdge, ahpe)):
            return cls.chooseTileHarbor(cls, ahpe, i)  # RICORSIONE, occhio che ci va la i non tos.
        else:
            p1 = choosenEdge[0]
            #print("p1:", p1)
            p2 = choosenEdge[1]
            #print("p2:", p2)
            for p in range(0, len(cls.tiles[tos].associated_places)):
                if((cls.tiles[tos].associated_places[p] == p1) or (cls.tiles[tos].associated_places[p] == p2)):
                    cls.places[cls.tiles[tos].associated_places[p]].harbor = cls.harbors[i]
                    #print("INSERITO HARBOR" , cls.harbors[i], "IN PLACE: ", cls.tiles[tos].associated_places[p]) 
            ahpe.append(p1)
            ahpe.append(p2)
            return ahpe

    def edgeAlreadyHarbor(cls, edge, places):
        if((edge[0] in places) or (edge[1] in places)):
            return True
        return False

    def __repr__(cls):
        s = ""
        for t in cls.tiles:
            s = s + "Tile: " + str(t) + "\n"
        for p in cls.places: 
            s = s + "Place: "+ str(p) + "\n"
        return s


b = Board()
#for p in Board().places:
#    print(p.owner)
