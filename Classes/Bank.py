import Classes.Player as Player
import Classes.Game as Game


class Bank:
    instance = None
    def __new__(cls):
        if cls.instance is None: 
            cls.instance = super(Bank, cls).__new__(cls)
            cls.resources = {"wood" : 19, "clay" : 19, "crop" : 19, "sheep" : 19, "iron" : 19}  
        return cls.instance
    
    def resourceToAsk(cls, player, resource):
        harborToCheck = "2:1 " + resource
        if (harborToCheck in player.ownedHarbors):
             return 2
        elif("3:1" in player.ownedHarbors):
             return 3
        else: 
             return 4

    def printBank(cls):
        print("Bank situation: \n")
        for res in cls.resources:
            print(res + " " + str(cls.resources[res]) + "\n")

    def reset(cls):
        Bank.instance = None


