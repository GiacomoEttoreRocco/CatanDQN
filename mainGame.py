import Classes as c

if __name__ == "__main__":
    whoWon = []
    for i in range(0, 1):
        g = c.Game.Game()
        whoWon.append(g.playGame().id)
        c.Board.Board().reset()
        c.Bank.Bank().reset()
        #print(Board.Board().places)
        #print(Board.Board().edges)
        #time.sleep(4043)

    print("Who won? ", whoWon)