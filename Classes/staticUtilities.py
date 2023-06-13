def availableResourcesForColony(resDict):
    return resDict["wood"] >= 1 and resDict["clay"] >= 1 and resDict["sheep"] >= 1 and resDict["crop"] >= 1

def availableResourcesForCity(resDict):
    return resDict["iron"] >= 3 and resDict["crop"] >= 2

def availableResourcesForStreet(resDict):
    return resDict["wood"] >= 1 and resDict["clay"] >= 1 

def availableResourcesForDevCard(resDict):
    return resDict["crop"] >= 1 and resDict["iron"] >= 1 and resDict["sheep"] >= 1

