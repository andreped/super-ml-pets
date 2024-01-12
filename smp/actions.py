"""
Methods for interacting with the game through mouse movements
"""

import pyautogui as gui
from .utils import *
import time
import logging as log
import matplotlib.pyplot as plt


class SuperAutoPetsMouse:
    """
    A Class that implements the interface between reinforcement agent and game
    convention:
        - when the parameters for a method has 2 inputs(team slot, shop slot) then the team slot
        always comes in the first
    """

    def __init__(self):
        self.position = get_position()
        self.team_position = [1 for _ in range(5)]
        self.logger = log.getLogger()

    def _shop2team(self, n1, n2):
        """
        helper method to click to locations
        """
        # print("_shop2team")
        # print(n1, n2)
        self.logger.info("MOUSE ACTION [self._shop2team]: Buys Pet from" +
                         "{}th Shop Slot to {}th Team Slot".format(n1, n2))
        print("_SHOP2TEAM EVENT:")
        print(self.position[str(n1) + '_slot'])
        print(self.position[str(n2) + '_team_slot'])
        gui.click(self.position[str(n1) + '_slot'])
        plt.pause(2)
        gui.click(self.position[str(n2) + '_team_slot'])

        #gui.dragTo(self.position[str(n1) + '_slot'],
        #           self.position[str(n2) + '_team_slot'],
        #           duration=1.0,  # how long drag event should take
        #           tween=gui.easeOutQuad,  # move_drag_tween
        #           button='left')

    def _click(self, first_click):
        """
        helper method to click
        """
        self.logger.info("MOUSE ACTION [self._click]: Clicks the position {}".format(first_click))
        gui.click(self.position[first_click])

    def move_pet(self, n1, n2):
        """
        a helper method to move the pets in the team
        n1 to n2
        """
        if n1 == n2:
            self.logger.warning("MOUSE ACTION [self.move_pet]: (CAUSE) The Pet"+
            " Slots to move Pet is same = {}".format(n1))
            return
        self.logger.info("MOUSE ACTION [self.move_pet]: Moving the Pet from"+
        " {}th Team Slot to {}th Team Slot".format(n1,n2))
        self.logger.info("MOUSE ACTION [self.move_pet]: Calls [self._click]")
        self._click(str(n1) + '_team_slot')
        #gui.mouseDown(button="left")
        gui.dragTo(self.position[str(n2) + '_team_slot'][0],
                   self.position[str(n2) + '_team_slot'][1],
                   duration=3.0,  # how long drag event should take
                   tween=gui.easeOutQuad,  # move_drag_tween
                   button='left')
        #gui.moveTo(self.position[str(n2) + '_team_slot'][0],
        #           self.position[str(n2) + '_team_slot'][1], duration=0.2)
        #time.sleep(2.5)
        #gui.mouseUp(button="left")

    def buy(self, nth_slot):
        """
        method to buy pets from shop
        """
        # print("buy")
        # print(nth_slot)
        # if type(nth_slot[0]) == type(()):
        #     if len(nth_slot[0]) == 2:
        #         nth_slot1 = nth_slot[0][0]
        #         nth_slot2 = nth_slot[0][1]
        #         self._shop2team(nth_slot1, nth_slot2)
        #         return
        nth_slot = nth_slot[0]
        self.logger.info("MOUSE ACTION [self.buy]: Buy Pet from {}th Shop Slot".format(nth_slot))
        for j, i in enumerate(self.team_position):
            if i:
                self.logger.info("MOUSE ACTION [self.buy]: Calls [self._shop2team]")
                self._shop2team(nth_slot, j)
                self.team_position[j] = 0
                return

    def buy_food(self, nth_slot, num_pets):
        """
        method to buy food
        """
        target = 3  # @TODO: Is this correct? Aren't we buying food to the same animal at all times here?
        if len(nth_slot) != 1:
            target = nth_slot[1]
        nth_slot = nth_slot[0]
        nth_slot = nth_slot - num_pets + 5
        if nth_slot == 5 or nth_slot == 6:
            self.logger.info("MOUSE ACTION [self.buy_food]: Buy Food from {}th Shop Slot to target {}".format(nth_slot, target))
            self.logger.info("MOUSE ACTION [self.buy_food]: Calls [_shop2team]")
            self._shop2team(nth_slot, target)
        else:
            self.logger.critical("MOUSE ACTION [self.buy_food]: Failed!!!")
            self.logger.debug("MOUSE ACTION [self.buy_food]: The nth_slot is incorrect, nth_slot = ".format(nth_slot))
            raise Exception("Invalid buy_food: The nth_slot is incorrect, nth_slot = ", nth_slot)

    def buy_team_food(self, nth_slot, num_pets):
        """
        method for buying team food
        """
        target = 3  # when buying for team, it doesnt matter which team member we aim at
        nth_slot = nth_slot[0]
        nth_slot = nth_slot - num_pets + 5
        if nth_slot == 5 or nth_slot == 6:
            self.logger.info("MOUSE ACTION [self.buy_team_food]: Buy Food from {}th Shop Slot".format(nth_slot))
            self.logger.info("MOUSE ACTION [self.buy_team_food]: Calls [self._shop2team]")
            self._shop2team(nth_slot, target)
        else:
            self.logger.critical("MOUSE ACTION [self.buy_team_food]: Failed!!!")
            self.logger.debug("MOUSE ACTION [self.buy_team_food]: The nth_slot is incorrect, nth_slot = ".format(nth_slot))
            raise Exception("Invalid buy_food: The nth_slot is incorrect, nth_slot = ", nth_slot)

    def sell_buy(self, nth_slot, nth_team_slot):
        """
        method to buy and place the pet in a specified team slot
        """
        if self.team_position[nth_team_slot] == 0:
            self.logger.info("MOUSE ACTION [self.sell_buy]: Sell Pet in the {}th ".format(nth_team_slot)+
            "Team Slot and Replace with Pet in the {}th Shop Slot".format(nth_slot))
            self.logger.info("MOUSE ACTION [self.sell_buy]: Calls [self.sell]")
            self.sell(nth_team_slot)
            self.logger.info("MOUSE ACTION [self.sell_buy]: Calls [self._shop2team]")
            self._shop2team(nth_slot, nth_team_slot)
            self.team_position[nth_team_slot] = 0
        else:
            self.logger.debug("MOUSE ACTION [self.sell_buy]: No Pet present in" +
            "in the {}th Team Slot".format(nth_team_slot))
            raise Exception("Invalid sell_buy: No pet present in the Team Slot to sell...")

    def sell(self, nth_team_slot):
        """
        method to sell a pet from the team
        """
        nth_team_slot = nth_team_slot[0]
        if self.team_position[nth_team_slot] == 0:
            self.logger.info("MOUSE ACTION [self.sell]: Sells Pet in" + 
            "the {}th Team Slot".format(nth_team_slot))
            self.logger.info("MOUSE ACTION [self.sell]: Calls [self._click]")
            self._click(str(nth_team_slot) + '_team_slot')
            gui.click(self.position['sell'])
            self.team_position[nth_team_slot] = 1
        else:
            raise Exception("Invalid sell: No pet present in the slot to sell...")

    def combine_in_team(self, n):
        """
        method to combine 2 pets (n1 and n2) in the team
        """
        n1 = n[0]
        n2 = n[1]
        if self.team_position[n1] == 1 or self.team_position[n2] == 1:
            self.logger.debug("MOUSE ACTION [self.combine_in_team]: FAILED!!!")
            self.logger.debug("MOUSE ACTION [self.combine_in_team]: (Invalid Combine)" + 
            " Pets in Team Slot is not present")
            raise Exception("Invalid Combine: Pets in Team Slot is not present...")
        if 0 <= n1 < 5 and 0 <= n2 < 5:
            self.logger.info("MOUSE ACTION [self.combine_in_team]: Combines Pets" + 
            " in {}th and {}th Team Slots".format(n1, n2))
            self.logger.info("MOUSE ACTION [self.combine_in_team]:"+
            " Calls [self._click]")
            self._click(str(n1) + '_team_slot')
            self.logger.info("MOUSE ACTION [self.combine_in_team]:"+
            " Calls [self._click]")
            self._click(str(n2) + '_team_slot')
            self.team_position[n2] = 0
        else:
            self.logger.debug("MOUSE ACTION [self.combine_in_team]: FAILED!!!")
            self.logger.debug("MOUSE ACTION: (CAUSE)"+
            " Indices {} and {} out of range".format(n1, n2))
            raise Exception("Invalid Combine: index out of range")

    def buy_combine(self, n):
        """
        method to buy and combine 2 pets
        """
        nth_slot = n[0]
        nth_team_slot = n[1]
        # print("\n134 actions.py - self.team_position[nth_team_slot], nth_team_slot:", self.team_position[nth_team_slot], nth_team_slot)
        if self.team_position[nth_team_slot] == 1:
            self.logger.debug("MOUSE ACTION [self.buy_combine]: FAILED!!!")
            self.logger.debug("MOUSE ACTION [self.buy_combine]: (CAUSE)"+
            " There is no Pet in Team Slot {}".format(nth_team_slot))
            raise Exception("Invalid buy_combine: pet in team slot not present...")
        self.logger.info("MOUSE ACTION [self.buy_combine]: Buys and Combines Pet "+
        "from {}th Shop Slot to {}th Team Slot".format(nth_slot, nth_team_slot))
        self.logger.info("MOUSE ACTION [self.buy_combine]: Calls [_shop2team]")
        self._shop2team(nth_slot, nth_team_slot)

    def reorder(self, order):
        """
        method to reorder the team
        """
        order_str = " ".join(map(str, order[0]))
        self.logger.info("MOUSE ACTION [self.reorder]: Reorders The Team "+
        " to the order ({})".format(order_str))
        # self.logger.info("current order:", order_str)
        order = order[0]
        order = list(order)
        # copy_order = list(order)

        orig_order = list(range(len(order)))  # (0, 1, 2, 3, 4) - for len = 5
        curr_order = orig_order.copy()

        # tried to fix reordering - left to right
        for i, j in enumerate(order):
            if curr_order == order:
                break
            loc = curr_order.index(j)  # get location of value of interest
            curr_order.pop(loc)
            curr_order.insert(i, j)
            self.logger.info("MOUSE ACTION [self.reorder]: Calls [self.move_pet]")
            self.move_pet(loc, i)  # move value to position i
        return order

        #for i, j in enumerate(order):
        #    if i != j:
        #        self.move_pet(i, j)
        #        del copy_order[i]
        #        copy_order.insert(j, j)
        #        final_order = self.reorder((tuple(copy_order), ))  # recursively solves
        #        return final_order
        #return copy_order  # when all the elements are already sorted

    # @TODO: Note that this method is never used, as sapai-gym don't currently support freezing/unfreezing
    def freeze_unfreeze(self, nth_slot):
        """
        a method to freeze or unfreeze pets
        """
        self._click(str(nth_slot) + '_slot')
        self._click('freeze')

    def end_turn(self, _):
        """
        a method to end turn
        """
        self.logger.info("MOUSE ACTION [self.end_turn]: Ends the Round")
        self._click('end_turn')

    def roll(self):
        """
        a method to roll
        """
        self.logger.info("MOUSE ACTION [self.roll]: Rolls to refresh Shop Pets and Food")
        self.logger.info("MOUSE ACTION [self.roll]: Calls [self._click]")
        self._click('roll')

    def get_action_dict(self):
        """
        returns a dictionary of all the methods
        """
        return {
            'roll': self.roll,
            'end_turn': self.end_turn,
            'reorder': self.reorder,
            'buy_pet': self.buy,
            'buy_food': self.buy_food,
            'buy_food_team': self.buy_team_food,
            'sell': self.sell,
            'buy_combine': self.buy_combine,
            'combine': self.combine_in_team
            }
