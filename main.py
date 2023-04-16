from argparse import ArgumentParser
import sys
import os
from smp.utils import define_logger
import logging as log


def main(virtual_screen=False):
    parser = ArgumentParser()
    parser.add_argument('-t', '--task', type=str, nargs='?', default="train",
                        help="which task to perform. Either 'train' or 'eval'.")
    parser.add_argument('-nt', '--nb_steps', type=int, nargs='?', default=1024,
                        help="number of time steps when training RL model.")
    parser.add_argument('-ng', '--nb_games', type=int, nargs='?', default=1024,
                        help="number of games to run when evaluating model after training.")
    parser.add_argument('-fn', '--finetune', type=str, nargs='?', default=None,
                        help="whether to finetune using pretrained model. Provide path to model if yes.")
    parser.add_argument('-m', '--model_name', type=str, nargs='?', default="rl_model",
                        help="which name to use for the model.")
    parser.add_argument('-nr', '--nb_retries', type=int, nargs='?', default=1,
                        help="number of times training is restarted (continued) if it crashes. Set to -1 to train forever.")
    parser.add_argument('-im', '--infer_model', type=str, nargs='?', default=None,
                        help="which model to use for deployment. Full path excluding '.zip' extension.")
    parser.add_argument('-py', '--infer_pversion', type=str, nargs='?', default="3.7",
                        help="define which python version the current deployment model is trained with.")
    parser.add_argument('-bs', '--batch_size', type=int, nargs='?', default=512,
                        help="set which batch size to use for training.")
    parser.add_argument('-lr', '--learning_rate', type=float, nargs='?', default=0.0003,
                        help="set which learning rate to use for training.")
    parser.add_argument('-sf', '--save_freq', type=int, nargs='?', default=1024,
                        help="set frequency of how often models are saved using checkpoint callback.")
    parser.add_argument('-gm', '--gamma', type=float, nargs='?', default=0.99,
                        help="set which gamma to use for MaskablePPO training.")
    parser.add_argument('-v', '--verbose', type=int, nargs='?', default=1,
                        help="sets the verbose level.")
    ret = parser.parse_known_args(sys.argv[1:])[0]

    # set verbose handler
    define_logger(verbose=ret.verbose)

    log.debug(ret)

    if ret.task == "train":
        from smp.train_agent import train_with_masks
        train_with_masks(ret)
    elif ret.task == "deploy":
        if ret.infer_model is None:
            raise ValueError("Please, provide the path to the model to use for deployment, by setting 'infer_model'.")
        elif not os.path.exists(ret.infer_model + ".zip"):
            raise ValueError("The model chosen for deployment does not exist. Chosen model:", ret.infer_model)

        from smp.deploy_agent import run, pause
        
        if virtual_screen:
            log.info("Pausing...")
            pause()

        log.info("Running...")
        run(ret)
    else:
        raise ValueError("Unknown task specified. Available tasks include {'train', 'deploy'}, but used:", ret.task)


if __name__ == "__main__":
    main()  # pragma: no cover
