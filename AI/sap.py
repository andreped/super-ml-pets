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
    turns = 0
    actions_taken_this_turn = 0

    def __init__(self):
        self.player = Player(pack="StandardPack")

    def step(self, action):
        """
        Update the system state using the best of action
        """
        action = argmax(action)
        
        try:
            if action < 35:
                # buyshop
                # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34
                tm_idx = action/7
                shp_idx = action%7
                tm_slot = self.player.team[tm_idx]
                shp_slot = self.player.shop[shp_idx]
                
                # buy pet (always puts in last slot), buy combine 
                if shp_slot.slot_type == "pet":
                    if tm_slot.empty:
                        self.player.buy_pet(shp_slot)
                        self.player.team.move(len(self.player.team)-1, tm_idx)
                    else:
                        self.player.buy_combine(shp_slot, tm_slot)
                    
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

        except:
            score -= 10

    def get_scaled_state(self):
        """
        Get full state, scaled into (approximately) [0, 1].
        State is: 
        team states {id, atk, def, food},
        shop states {id, atk, def},
        money, turn, lives, wins
        """

        state = []
        for teamslot_state in self.player.team.state["team"]:
            pet = teamslot_state["pet"]
            if pet["name"] == "pet-none":
                state.extend([89, 0, 0, 0])
            else:
                state.extend([(list(data["pets"].keys()).index(
                    pet["name"]))/len(data["foods"]), pet["attack"]/50, pet["health"]/50,
                    (list(data["statuses"].keys()).index(pet["status"]))/len(data["statuses"])])

        for shopslot_state in self.player.shop.state["shop_slots"]:
            item = shopslot_state["item"]
            if item["name"] == "pet-none" or item["name"] == "food-none":
                state.extend([89/len(data["pets"]), 0, 0])
            elif item["type"] == "Food":
                state.extend([(list(data["foods"].keys()).index(
                    item["name"]))/len(data["foods"]), item["attack"]/50, item["health"]/50])
            else:
                state.extend([(list(data["pets"].keys()).index(
                    item["name"]))/len(data["pets"]), item["attack"]/50, item["health"]/50])

        for i in range(7-len(self.player.shop)):
            state.extend([89/len(data["pets"]), 0, 0])

        state.extend([self.player.gold, self.player.turn,
                     self.player.lives, self.wins])

        return state
