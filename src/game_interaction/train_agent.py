from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from sapai_gym import SuperAutoPetsEnv
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator
from tqdm import tqdm
import numpy as np

def opponent_generator(num_turns):
    # Returns teams to fight against in the gym 
    opponents = biggest_numbers_horizontal_opp_generator(25)
    return opponents

def train_with_masks(nb_timesteps: int, nb_games: int, finetune: bool): #,
    #gamma: int):
    # initialize environment
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    # setup logger
    logger = configure("./history/sb3_log/")

    # setup model checkpoint callback, to save model after a specific #iters
    checkpoint_callback = CheckpointCallback(save_freq=1000, 
        save_path='./models/', name_prefix='model_sap_gym_sb3_070822_checkpoint')

    if finetune:
        model = MaskablePPO.load("./models/model_sap_gym_sb3_070822")
        model.set_env(env)
    else:
        model = MaskablePPO("MlpPolicy", env, verbose=1)

# train
    print("\nTraining...")
    training_flag = True
    while training_flag:
        try:
            model.set_logger(logger)
            model.learn(total_timesteps=nb_timesteps)
            evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=1, warn=False)
            obs = env.reset()

            # if we reach 1M iterations, then training can stop, else, restart!
            training_flag = False
        except AssertionError as e1:
            print(e1)
        except TypeError as e2:
            print(e2)
            print("Model stopped training...")
        except ValueError as e3:
            print(e3)

        # load previous checkpoint
        model = MaskablePPO.load("./models/model_sap_gym_sb3_070822_checkpoint")
        model.set_env(env)

    # save best model
    model.save("./models/model_sap_gym_sb3_070822_checkpoint")

    del model

    # load model
    trained_model = MaskablePPO.load("./models/model_sap_gym_sb3_070822_checkpoint")

    print("\nPredicting...")
    
    # predict
    obs = env.reset()
    rewards = []
    for i in tqdm(range(nb_games), "Games:"):
        # Predict outcome with model
        action_masks = get_action_masks(env)
        action, _states = trained_model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        rewards.append(reward)
    print(sum(rewards), len(rewards), np.mean(rewards))
    env.close()

if __name__ == "__main__":
    train_with_masks(nb_timesteps=10000000, nb_games=10000, 
        finetune=True)#, gamma=0.99)
