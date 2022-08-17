from src.game_interaction.train_agent import train_with_masks
from argparse import ArgumentParser
import sys

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--nb_steps', metavar='--nt', type=int, nargs='?', default=10000,
                        help="number of time steps when training RL model.")
    parser.add_argument('--nb_games', metavar='--ng', type=int, nargs='?', default=10000,
                        help="number of games to run when evaluating model after training.")
    parser.add_argument('--finetune', metavar='--fn', type=str, nargs='?', default=None,
                        help="whether to finetune using pretrained model. Provide path to model if yes.")
    parser.add_argument('--model_name', metavar='--m', type=str, nargs='?', default=None,
                        help="which name to use for the model.")
    ret = parser.parse_args(sys.argv[1:])
    print(ret)

    train_with_masks(
        nb_timesteps=ret.nb_steps,
        nb_games=ret.nb_games,
        finetune=ret.finetune,
        model_name=ret.model_name,
        )
