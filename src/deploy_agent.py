"""
Methods for agent interaction using sapai-gym and stable basline 3
"""

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sapai_gym import SuperAutoPetsEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator
from sapai_gym.ai import baselines
from sapai import *
from sapai.shop import *
from .image_detection import *
from .actions import *
import pynput
import matplotlib.pyplot as plt
import pyautogui as gui
import sys
import logging as log
from .utils import opponent_generator


# global variable
stop_program = False


def kill_process(key):
    """
    method to stop agent from running if 'escape' key is pressed
    """
    global stop_program
    if key == pynput.keyboard.Key.esc:
        print("\nEscape pressed, stopping agent...")
        stop_program = True
        return False


def pause():
    """
    method which pauses until 'space' keyboard key is pressed
    """
    while True:
        #if keyboard.read_key() == 'space':
        if key == pynput.keyboard.Key.space:
            break


def time_pause(time: int):
    """
    method which pauses a specified time (in ms)
    """
    plt.pause(time)


def get_action_name(k: int) -> str:
    """
    translated action name from integer
    """
    name_val = list(SuperAutoPetsEnv.ACTION_BASE_NUM.items())
    assert k >= 0
    for (start_name, _), (end_name, end_val) in zip(name_val[:-1], name_val[1:]):
        if k < end_val:
            return start_name
    else:  # @TODO: this can't possibly be the corrct placement, or?
        return end_name


def remove_nothing(pet_list):
    """
    removes all occurrences of 'nothing' in pet list
    """
    pets = []
    for i in pet_list:
        if i != 'nothing':
            pets.append(i)
    return pets


def run(ret):
    """
    method to use pretrained RL model with the real game (deployment)
    """
    interface = SuperAutoPetsMouse()
    action_dict = interface.get_action_dict()

    # custom object relevant for supporting using model trained using a different python version than the one used now
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,  # @TODO: Is this needed? Probably not for MaskablePPO
        "clip_range": lambda _: 0.0,
    }

    log.info("INITIALIZATION [self.run]: Loading Model")
    model = MaskablePPO.load(ret.infer_model, custom_objects=custom_objects)

    log.info("INITIALIZATION [self.run]: Create SuperAutoPetsEnv Object")
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
    obs = env.reset()

    with pynput.keyboard.Listener(on_press=kill_process) as listener:
        while not stop_program:
            time_pause(0.5)
            log.info("CV SYSTEM [self.run]: Detect the Pets and Food" +
                                  " in the Shop Slots")
            log.info("CV SYSTEM [self.run]: Calls " +
                                  "[image_detection.find_the_animals]")
            pets, _ = find_the_animals(
                directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "SAP_res/").replace("\\", "/"))
            pets = remove_nothing(pets)
            log.info("CV SYSTEM [self.run]: The detected Pets and Food in the Shop is : {}".format(pets))
            log.info("GAME ENGINE [self.run]: Set Environment Shop = " +
                                  "detected Pets and Food")
            env.player.shop = Shop(pets)
            if env.player.lives <= 3:
                log.info("GAME ENGINE [self.run]: Increment number of " +
                                      "remaining lives by 3")
                env.player.lives += 3
            action_masks = get_action_masks(env)
            obs = env._encode_state()
            log.info("GAME ENGINE [self.run]: Get the best action" +
                                  " to make for the given state from the loaded model")
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            s = env._avail_actions()

            time_pause(1.0)  # 0.5
            log.info("GAME ENGINE [self.run]:" +
                                  " Current Team and Shop \n{}".format(s[action][0]))
            log.info("GAME ENGINE [self.run]:" +
                                  " Best Action = {} {}".format(action, get_action_name(action)))
            log.info("GAME ENGINE [self.run]: Instruction given " +
                                  "by the model = {}".format(s[action][1:]))
            # log.info("GAME ENGINE [self.en")
            if env._is_valid_action(action):
                log.info("GAME ENGINE [self.run]: Action is valid")
                if get_action_name(action) == 'buy_food':
                    num_pets = 0
                    num_food = 0
                    for shop_slot in env.player.shop:
                        if shop_slot.slot_type == "pet":
                            num_pets += 1
                        if shop_slot.slot_type == "food":
                            num_food += 1
                    log.info("GAME ENGINE [self.run]:" + " Calls {}".format(get_action_name(action)) +
                             " with parameters {}, {}".format(s[action][1:], num_pets - num_food % 2))
                    action_dict[get_action_name(action)](s[action][1:], num_pets - num_food % 2)
                elif get_action_name(action) == 'buy_team_food':  # same behaviour as for buy_food for single animal
                    num_pets = 0
                    num_food = 0
                    for shop_slot in env.player.shop:
                        if shop_slot.slot_type == "pet":
                            num_pets += 1
                        if shop_slot.slot_type == "food":
                            num_food += 1
                    log.info("GAME ENGINE [self.run]:" + " Calls {}".format(get_action_name(action)) +
                             " with parameters {}, {}".format(s[action][1:], num_pets - num_food % 2))
                    action_dict[get_action_name(action)](s[action][1:], num_pets - num_food % 2)
                else:
                    if get_action_name(action) == 'roll':
                        log.info("GAME ENGINE [self.run]: " +
                                 "Calls {}".format(get_action_name(action)) + " with no parameters")
                        action_dict[get_action_name(action)]()
                    else:
                        log.info("GAME ENGINE [self.run]: " + "Calls {}".format(get_action_name(action)) +
                                 " with parameters {}".format(s[action][1:]))
                        action_dict[get_action_name(action)](s[action][1:])
            log.info("GAME ENGINE [self.run]: Implements the action" +
                     " in the Environment\n\n\n")
            obs, reward, done, info = env.step(action)
            if get_action_name(action) == 'end_turn':
                # time_pause(1.5)
                # end turned press, start clicking until end turn button shows again (game is over)
                time_pause(3.0)
                battle_finished = False
                while not battle_finished:
                    # click event
                    log.info("TRIVIAL MOUSE ACTION [self.run]: clicking to skip the battle")
                    gui.click(1780, 200)

                    # check if battle is done
                    if find_paw():
                        log.info("GAME ENGINE [self.run]: Battle is over")
                        battle_finished = True
                    else:
                        # check if game is over
                        if find_arena():
                            time_pause(0.2)
                            log.info("GAME ENGINE [self.run]: Game is over! Start new game")
                            gui.click(600, 400)

                gui.click(1780, 200)

        listener.join()

    # print(s[action][0])
    env.close()
