from time import sleep
from Game import GameState, PX_RESHAPE
from AI import DQNAgent

EPISODE_NUM = 1000
START_NUM = 450

if __name__ == "__main__":
    game = GameState()
    agent = DQNAgent(PX_RESHAPE[0]*PX_RESHAPE[1], len(game.actions))
    agent.load(START_NUM)
    wins = 0

    print("starting")

    sleep(2)

    for e in range(START_NUM+1, EPISODE_NUM):
        game.start_game()

        sleep(1)

        end = False
        game.update()
        last_screen = game.screen_px
        last_turn = game.turn
        last_wins = game.wins
        last_losses = game.losses
        reward = 0
        action_counter = 0

        while not end:
            action = agent.act(last_screen)
            game.actions[action]()
            game.update()

            action_counter += 1
            if action_counter > 20:
                reward -= .5

            if game.end_result != 0:
                if game.end_result == 1:
                    print("win")
                    wins += 1
                    reward += 10000
                else:
                    print("loss")
                    reward -= 150
                end = True

            elif last_turn < game.turn:
                if game.wins > last_wins:
                    print("won")
                    reward += 60*game.turn
                elif game.losses > last_losses:
                    print("lost")
                    reward -= 30

                    if action_counter < 3:
                        game.action_forfeit()
                        reward -= 80
                        end = True
                else:
                    print("draw")
                    reward += 10

                action_counter = 0

                last_turn = game.turn
                last_wins = game.wins
                last_losses = game.losses

            agent.remember(last_screen, action,
                           reward, game.screen_px, end)

            last_screen = game.screen_px

            if end:
                game.action_cancel()
                game.action_cancel()

        agent.replay(32)

        print('episode: {}/{}, score: {}'.format(e, EPISODE_NUM, reward))
        print('wins: {}/{}'.format(wins, e-START_NUM))

        if e % 10 == 0:
            agent.save(e)

        sleep(4)

    agent.save()
