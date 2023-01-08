import Graphics.geomlib as geomlib

def getCoords(view, i):
    tile, place = placeCoordinates[i]
    hexLayout = geomlib.Layout(geomlib.layout_pointy, geomlib.Point(view.height//12, view.height//12), geomlib.Point(view.gameWidth/2, view.height/2))
    return geomlib.polygon_corners(hexLayout, view.graphicTileList[tile].hex)[place]

placeCoordinates = { 
    0 : (0,3),
    1 : (0,2),
    2 : (0,1),
    3 : (1,2),
    4 : (1,1),
    5 : (2,2),
    6 : (2,1),
    7 : (3,3), 
    8 : (3,2),
    9 : (3,1),
    10 : (4,2),
    11 : (4,1),
    12 : (5,2),
    13 : (5,1),
    14 : (6,2),
    15 : (6,1),
    16 : (7,3),
    17 : (7,2),
    18 : (7,1),
    19 : (8,2),
    20 : (8,1),
    21 : (9,2),
    22 : (9,1),
    23 : (10,2),
    24 : (10,1),
    25 : (11,2),
    26 : (11,1),
    27 : (7,4),
    28 : (12,3),
    29 : (12,2),
    30 : (12,1),
    31 : (13,2),
    32 : (13,1),
    33 : (14,2),
    34 : (14,1),
    35 : (15,2),
    36 : (15,1),
    37 : (11,0),
    38 : (12,4),
    39 : (16,3),
    40 : (16,2),
    41 : (16,1),
    42 : (17,2),
    43 : (17,1),
    44 : (18,2),
    45 : (18,1),
    46 : (15,0),
    47 : (16,4),
    48 : (16,5),
    49 : (16,0),
    50 : (17,5),
    51 : (17,0),
    52 : (18,5),
    53 : (18,0)
    }












