"""
General settings and implementation of the SAP Game
"""

import random
from numpy import argmax, roll
from sapai import Player
from sapai import data
from sapai import Food
from sapai.battle import Battle

past_teams = []

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

        self.actions_taken_this_turn += 1
        
        try:
            if action < 35:
                # buyshop
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
                else:
                    self.player.buy_food(shp_slot, tm_slot)
                    
            elif action < 55:
                # moveteam
                action -= 35
                tm1_idx = action/5
                tm2_idx = action%5
                
                if self.player.team[tm1_idx].name == self.player.team[tm2_idx].name and not self.player.team[tm1_idx].empty:
                    self.player.combine(tm2_idx, tm1_idx)
                else:
                    self.player.team.move(tm1_idx, tm2_idx)
            elif action < 60:
                # sellteam
                action -= 55
                tm_slot = self.player.team[action]
                
                self.player.sell(tm_slot)
            elif action < 67:
                # freezeshop
                action -= 60
                shp_slot = self.player.shop[action]

                self.player.freeze(shp_slot)
            elif action == 68:
                # rollshop
                self.player.roll()
            else:
                # endturn
                self.actions_taken_this_turn = 0
                self.player.end_turn()

                battle = Battle(self.player.team, past_teams[random.randint(0, len(past_teams)-1)])
                winner = battle.battle()

                if winner == 0:
                    self.wins += 1
                    self.score += 50
                elif winner == 1:
                    self.score -= 10
                else:
                    self.score += 20

                past_teams.append(self.player.team)

        except:
            self.score -= 10

    def battle():
        print("BATTLE")

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
