from time import sleep
from Game import GameState

game = GameState()
sleep(3)

while True:
    game.update()
    game.in_battle = True
    input()


