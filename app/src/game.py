"""Game Interaction Module

Allows for reading of the screen and clicking on elements
"""
from calendar import c
from time import sleep
import pickle
from typing import List, Callable
import numpy as np
from mss import mss
from pynput.mouse import Button, Controller
from PIL import Image
import cv2
import pytesseract

# Pixel stuff
PX_WINDOW = {"left": 0, "top": 35, "width": 1920, "height": 985}
PX_START = (960, 400)

PX_LIVES_BOX = ((245, 60), (60, 50))
PX_WINS_BOX = ((400, 60), (35, 50))

PX_ITEM_SHOP = (560, 665)
PX_ITEM_TEAM = (560, 400)
PX_ITEM_WIDTH = 132

PX_ROLL = (210, 930)
PX_ROLL_CHECK = (247, 106, 0)
PX_FREEZE = (1200, 930)
PX_SELL = (1200, 930)
PX_END = (1750, 930)
PX_CANCEL = (1850, 280)
PX_SKIP_TURN = (1250, 640)

PX_BATTLE_END = (960, 400)
PX_BATTLE_END_CHECK = (254, 205, 77)

PX_WIN = (1470, 410)
PX_WIN_CHECK = (42, 46, 52)
PX_LOSS = (1530, 530)
PX_LOSS_CHECK = (101, 69, 0)

PX_FF = ((1860, 45), (950, 590), (1250, 650))

PX_ERROR_MARGIN = 10

PX_RESHAPE = (400, 400)

SHOP_SIZE = 7
TEAM_SIZE = 5


class GameState:
    def __init__(self) -> None:
        self.reset()

        self.mouse = Controller()

        self.actions = self.get_actions()

    def reset(self) -> None:
        self.turn = 1
        self.wins = 0
        self.losses = 0
        self.in_battle = False

        self.last_lives = None
        self.last_win = None
        self.screen_px = None

        # 1 = win, 2 = loss
        self.end_result = 0

    def start_game(self) -> None:
        self.reset()
        print("Start Game")

        self.mouse.position = PX_START
        self.mouse.click(Button.left)

        sleep(.5)

    def update(self) -> None:
        ready = False
        while not ready:
            if not self.in_battle:
                self.action_cancel()

            if self.end_result != 0:
                return

            sleep(1.5)

    def get_actions(self) -> List[int]:
        # buy shop item and place
        # freeze shop item
        # roll shop
        # sell pet
        # move pet to place
        # end turn
        actions = []
        for i in range(SHOP_SIZE):
            for j in range(TEAM_SIZE):
                actions.append(self.action_buy(i, j))

        for i in range(SHOP_SIZE):
            actions.append(self.action_freeze(i))

        actions.extend([self.action_roll()])

        for i in range(TEAM_SIZE):
            actions.append(self.action_sell(i))

        for i in range(TEAM_SIZE):
            for j in range(TEAM_SIZE):
                if i != j:
                    actions.append(self.action_move(i, j))

        actions.extend([self.action_end()]*10)

        return actions

    def action_buy(self, shop_idx: int, team_idx: int) -> Callable:
        def buy():
            print("buy position "+str(shop_idx)+" to "+str(team_idx))
            self.mouse.position = (
                PX_ITEM_SHOP[0]+PX_ITEM_WIDTH*shop_idx, PX_ITEM_SHOP[1])
            self.mouse.click(Button.left)
            sleep(.5)
            self.mouse.position = (
                PX_ITEM_TEAM[0]+PX_ITEM_WIDTH*team_idx, PX_ITEM_TEAM[1])
            self.mouse.click(Button.left)

            sleep(.5)
            self.action_cancel()

        return buy

    def action_freeze(self, shop_idx: int) -> Callable:
        def freeze():
            print("freeze position "+str(shop_idx))
            self.mouse.position = (
                PX_ITEM_SHOP[0]+PX_ITEM_WIDTH*shop_idx, PX_ITEM_SHOP[1])
            self.mouse.click(Button.left)
            sleep(.5)
            self.mouse.position = PX_FREEZE
            self.mouse.click(Button.left)
            sleep(.5)

            self.action_cancel()

        return freeze

    def action_roll(self) -> Callable:
        def roll():
            print("roll")
            self.mouse.position = PX_ROLL
            self.mouse.click(Button.left)
            sleep(.5)

            self.action_cancel()

        return roll

    def action_sell(self, team_idx: int) -> Callable:
        def sell():
            print("sell position "+str(team_idx))
            self.mouse.position = (
                PX_ITEM_TEAM[0]+PX_ITEM_WIDTH*team_idx, PX_ITEM_TEAM[1])
            self.mouse.click(Button.left)
            sleep(.5)
            self.mouse.position = PX_SELL
            self.mouse.click(Button.left)
            sleep(.5)

            self.action_cancel()

        return sell

    def action_move(self, team_idx1: int, team_idx2: int) -> Callable:
        def move():
            print("move position "+str(team_idx1)+" to "+str(team_idx2))
            self.mouse.position = (
                PX_ITEM_TEAM[0]+PX_ITEM_WIDTH*team_idx1, PX_ITEM_TEAM[1])
            self.mouse.click(Button.left)
            sleep(.5)
            self.mouse.position = (
                PX_ITEM_TEAM[0]+PX_ITEM_WIDTH*team_idx2, PX_ITEM_TEAM[1])
            self.mouse.click(Button.left)
            sleep(.5)

            self.action_cancel()

        return move

    def action_end(self) -> Callable:
        def end():
            print("end turn")
            self.mouse.position = PX_END
            self.mouse.click(Button.left)
            sleep(.5)

            self.action_cancel()
            sleep(.5)

            self.mouse.position = PX_SKIP_TURN
            self.mouse.click(Button.left)

            self.in_battle = True

        return end

    def action_cancel(self) -> None:
        self.mouse.position = PX_CANCEL
        self.mouse.click(Button.left)
        sleep(.5)

    def action_forfeit(self) -> None:
        print("ff")
        for i in range(3):  
            self.mouse.position = PX_FF[i]
            self.mouse.click(Button.left)
            sleep(1)

        sleep(2)
