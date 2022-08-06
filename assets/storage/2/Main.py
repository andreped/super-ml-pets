from time import sleep
from random import randrange
from Game import GameState


EPISODE_NUM = 1000
START_NUM = 0

if __name__ == "__main__":
    game = GameState()

    wins = 0

    print("starting")

    sleep(2)

    for e in range(START_NUM+1, EPISODE_NUM):
        game.start_game()

        sleep(1)

        end = False
        game.update()
        last_turn = game.turn
        last_wins = game.wins
        last_losses = game.losses

        while not end:
            for j in range(min(4, game.turn)):
                for i in range(min(5, game.turn)):
                    for j in range(min(7, game.turn*2)):
                        game.action_buy(j, i)()

                game.action_roll()()

            game.action_end()()

            game.update()

            if game.end_result == 1:
                wins += 1

                end = True

            elif last_turn < game.turn:
                if game.wins > last_wins:
                    print("won")
                elif game.losses > last_losses:
                    print("lost")
                else:
                    print("draw")

                last_turn = game.turn
                last_wins = game.wins
                last_losses = game.losses

            if end:
                game.action_cancel()
                game.action_cancel()

        print('episode: {}/{}, score: {}'.format(e, EPISODE_NUM, game.wins))
        print('wins: {}/{}'.format(wins, e-START_NUM))

        sleep(4)
