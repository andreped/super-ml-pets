"""
General settings and implementation of the SAP Game
"""

from numpy import argmax, roll
from sapai import Player
from sapai import data
from sapai import Food


class SAP(object):
    score = 0
    wins = 0

    def __init__(self):
        self.player = Player(pack="StandardPack") 

    def step(self, action):
        """
        Update the system state using the best of action
        """
        action = argmax(action) + 1

        if action < 35:
            buyshop
        elif action < 55:
            moveteam
        elif action < 60:
            sellteam
        elif action < 67:
            freezeshop
        elif action == 68:
            rollshop
        else:
            endturn


    def get_scaled_state(self):
        """
        Get full state, scaled into (approximately) [0, 1].
        State is: 
        team states {id, atk, def, food},
        shop states {id, atk, def},
        money, turn, lives, wins
        """

        state = []
        for teamslot_state in range(self.player.team.state["team"]):
            pet = teamslot_state["pet"]
            state.extend([list(data["pets"].keys()).index(pet.name), pet.attack, pet.health, pet.status])

        for shopslot_state in range(self.player.shop.state["shop_slots"]):
            item = shopslot_state["item"]
            if item is Food:
                state.extend([list(data["foods"].keys()).index(item.name), item.attack, item.health])
            else:
                state.extend([list(data["pets"].keys()).index(item.name), item.attack, item.health])

        state.extend([self.player.gold, self.player.turn, self.player.lives, self.wins])
        
        return state


