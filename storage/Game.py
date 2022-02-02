"""Game Interaction Module

Allows for reading of the screen and clicking on elements
"""
from time import sleep
import pickle
from typing import List, Callable
import numpy as np
from mss import mss
from pynput.mouse import Button, Controller
from PIL import Image
import pytesseract

# Pixel stuff (1920x1080 windowed)
PX_START = (960, 400)

PX_ITEM_SHOP = (560, 665)
PX_ITEM_TEAM = (560, 400)
PX_ITEM_WIDTH = 132

PX_FILL_SHOP = (560, 760)
PX_FILL_TEAM = (560, 490)
PX_FILL_CHECK = (255, 255, 255)
PX_STAT_SHOP = (500, 735)
PX_STAT_TEAM = (500, 470)
PX_HEALTH_OFFSET_X = 65
PX_STAT_READ = (60, 40)

PX_MONEY = (85, 60)
PX_MONEY_READ = (60, 50)
PX_LIVES = (245, 60)
PX_LIVES_READ = (60, 50)
PX_WINS = (400, 60)
PX_WINS_READ = (35, 50)
PX_TURNS = (620, 60)
PX_TURNS_READ = (85, 50)

PX_ROLL = (210, 930)
PX_ROLL_CHECK = (255, 106, 0)
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
PX_LOSS_CHECK = (101, 69)

PX_ERROR_MARGIN = 10


class GameState:
    _sct = mss()
    _pokedex = {}

    class Item:
        def __init__(self, id: int) -> None:
            self.id = id
            self.health = 0
            self.attack = 0

    def __init__(self) -> None:
        self.team = [self.Item(-1) for i in range(5)]
        self.lives = 10
        self.money = 10
        self.turn = 1
        self.wins = 0
        self.in_battle = False
        self.shop = [self.Item(-1) for i in range(7)]

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        self.update_info()

        self.mouse = Controller()

        self.actions = self.get_actions()

        # 1 = win, 2 = loss
        self.end_result = 0

    def start_game(self) -> None:
        self.__init__()
        self.mouse.position = PX_START
        self.mouse.click(Button.left)

    def update_info(self) -> None:
        info = []
        for i in self.team:
            info.extend([i.id, i.attack, i.health])

        info.extend([self.lives, self.money, self.turn, self.wins])

        for i in self.shop:
            info.extend([i.id, i.attack, i.health])

        self.info = info

    def update(self) -> None:
        ready = False
        while not ready:
            self.action_cancel()
            sct_img = GameState._sct.grab(GameState._sct.monitors[1])
            screen = np.array(Image.frombytes(
                "RGB", sct_img.size, sct_img.bgra, "raw", "BGRX"))

            sleep(.5)

            ready = self.is_ready(screen)

        if self.end_result != 0:
            return

        self.update_shop(screen)
        self.update_team(screen)
        self.update_money(screen)
        self.update_lives(screen)
        self.update_wins(screen)
        self.update_turns(screen)

        self.update_info()
        print(self.money, self.lives, self.wins, self.turn)
        print([str(a.id)+": "+str(a.attack)+", "+str(a.health)
              for a in self.team])
        print([str(a.id)+": "+str(a.attack)+", "+str(a.health)
              for a in self.shop])

    def update_shop(self, screen: np.array) -> None:
        for i in range(7):
            item_px = tuple(
                screen[PX_ITEM_SHOP[1], PX_ITEM_SHOP[0]+PX_ITEM_WIDTH*i])
            if item_px not in GameState._pokedex:
                GameState._pokedex[item_px] = len(GameState._pokedex) + 1

            item = GameState.Item(GameState._pokedex[item_px])

            if self.pixel_check(tuple(screen[PX_FILL_SHOP[1], PX_FILL_SHOP[0]+PX_ITEM_WIDTH*i]), PX_FILL_CHECK):
                attack_px = (PX_STAT_SHOP[0]+PX_ITEM_WIDTH*i, PX_STAT_SHOP[1])
                health_px = (PX_STAT_SHOP[0] +
                            PX_HEALTH_OFFSET_X+PX_ITEM_WIDTH*i, PX_STAT_SHOP[1])

                attack = self.read_value(screen, (attack_px, PX_STAT_READ))
                health = self.read_value(screen, (health_px, PX_STAT_READ))

                item.attack = attack
                item.health = health

            self.shop[i] = item

    def update_team(self, screen: np.array) -> None:
        for i in range(5):
            item_px = tuple(
                screen[PX_ITEM_TEAM[1], PX_ITEM_TEAM[0]+PX_ITEM_WIDTH*i])
            if item_px not in GameState._pokedex:
                GameState._pokedex[item_px] = len(GameState._pokedex) + 1

            if GameState._pokedex[item_px] != self.team[i]:
                item = GameState.Item(GameState._pokedex[item_px])
            else:
                item = self.team[i]

            if self.pixel_check(tuple(screen[PX_FILL_TEAM[1], PX_FILL_TEAM[0]+PX_ITEM_WIDTH*i]), PX_FILL_CHECK):
                attack_px = (PX_STAT_TEAM[0]+PX_ITEM_WIDTH*i, PX_STAT_TEAM[1])
                health_px = (PX_STAT_TEAM[0] +
                            PX_HEALTH_OFFSET_X+PX_ITEM_WIDTH*i, PX_STAT_TEAM[1])

                attack = self.read_value(screen, (attack_px, PX_STAT_READ))
                health = self.read_value(screen, (health_px, PX_STAT_READ))

                item.attack = attack
                item.health = health

            self.team[i] = item

    def update_money(self, screen: np.array) -> None:
        self.money = self.read_value(screen, (PX_MONEY, PX_MONEY_READ))

    def update_lives(self, screen: np.array) -> None:
        self.lives = self.read_value(screen, (PX_LIVES, PX_LIVES_READ))

    def update_wins(self, screen: np.array) -> None:
        self.wins = self.read_value(screen, (PX_WINS, PX_WINS_READ))

    def update_turns(self, screen: np.array) -> None:
        self.turn = self.read_value(screen, (PX_TURNS, PX_TURNS_READ))

    def read_value(self, screen: np.array, box: tuple) -> None:
        # box defined as ((top left x, top left y), (width, height))
        ss_crop = screen[box[0][1]:box[0][1]+box[1][1],
                         box[0][0]:box[0][0]+box[1][0]]
        (Image.fromarray(np.uint8(ss_crop))).save(
            str(box[0][0])+str(box[0][1])+str(self.turn)+str(self.money)+'help.png')
        value = pytesseract.image_to_string(
            ss_crop, config='-l sap --psm 8 --oem 3 -c tessedit_char_whitelist=0123456789')
        try:
            result = int(value)
        except:
            result = 0

        return result

    def is_ready(self, screen: np.array) -> bool:
        if self.in_battle:
            if self.finished_battle(screen):
                self.in_battle = False
                self.action_cancel()
                return True
            else:
                return False
                
        return self.pixel_check(tuple(screen[PX_ROLL[1], PX_ROLL[0]]), PX_ROLL_CHECK)

    def finished_battle(self, screen: np.array) -> bool:
        if self.pixel_check(tuple(screen[PX_BATTLE_END[1], PX_BATTLE_END[0]]), PX_BATTLE_END_CHECK):
            return True

        else:
            if self.pixel_check(tuple(screen[PX_WIN[1], PX_WIN[0]]), PX_WIN_CHECK):
                self.end_result = 1
                return True
            elif self.pixel_check(tuple(screen[PX_LOSS[1], PX_LOSS[0]]), PX_LOSS_CHECK):
                self.end_result = 2
                return True

        return False

    def pixel_check(self, screen_px: tuple, comparison: tuple) -> bool:
        return abs(sum([screen_px[i]-comparison[i] for i in range(len(screen_px))])) < PX_ERROR_MARGIN

    def get_actions(self) -> List[int]:
        # buy shop item and place
        # freeze shop item
        # roll shop
        # sell pet
        # move pet to place
        # end turn
        actions = []
        for i in range(len(self.shop)):
            for j in range(len(self.team)):
                actions.append(self.action_buy(i, j))

        for i in range(len(self.shop)):
            actions.append(self.action_freeze(i))

        actions.append(self.action_roll())

        for i in range(len(self.team)):
            actions.append(self.action_sell(i))

        for i in range(len(self.team)):
            for j in range(len(self.team)):
                actions.append(self.action_move(i, j))

        actions.append(self.action_end())

        return actions

    def action_buy(self, shop_idx: int, team_idx: int) -> Callable:
        def buy():
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
            self.mouse.position = PX_ROLL
            self.mouse.click(Button.left)
            sleep(.5)

            self.action_cancel()

        return roll

    def action_sell(self, team_idx: int) -> Callable:
        def sell():
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
            self.mouse.position = (
                PX_ITEM_SHOP[0]+PX_ITEM_WIDTH*team_idx1, PX_ITEM_SHOP[1])
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

    def save_pokedex(self) -> None:
        with open('pokedex.pkl', 'wb') as f:
            pickle.dump(GameState._pokedex, f)

    def load_pokedex(self) -> None:
        with open('pokedex.pkl', 'rb') as f:
            GameState._pokedex = pickle.load(f)
