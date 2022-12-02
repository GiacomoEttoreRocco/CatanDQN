import Player
class Bank:
    instance = None
    def __new__(cls):
        if cls.instance is None: 
            cls.instance = super(Bank, cls).__new__(cls)
            cls.resources = {"wood" : 19, "crop" : 19, "sheep" : 19, "iron" : 19, "clay" : 19}  
        return cls.instance
    
    def resourceToAsk(cls, player, resource):
        harborToCheck = "2:1 " + resource
        if (harborToCheck in player.harbors):
            return 2
        elif("3:1" in player.harbors):
            return 3
        else: 
            return 4

    def giveResource(cls, player: Player, resource):
        if(resource != "desert"):
            player.resources[resource] += 1
            cls.resources[resource] -= 1
    
    def trial(cls):
        print("ciao")
