"""
General settings and implementation of the SAP Game
"""

import random
import traceback

import numpy as np

import sapai
from sapai import Player
from sapai import Food
from sapai import Team
from sapai.battle import Battle


class SAP(object):
    def __init__(self, data):
        self.player = Player(pack="StandardPack")
        self.score = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.turns = 1
        self.actions_taken_this_turn = 0
        self.past_teams = data.past_teams
        self.logs = data.logs

    def step(self, action):
        """
        Update the system state using the best of action (0-68)
        """
        action = np.argmax(action)

        self.actions_taken_this_turn += 1

        self.score = 0

        if self.actions_taken_this_turn > 20:
            self.score -= 10

        try:
            if action < 35:
                # buyshop
                tm_idx = int(action/7)
                shp_idx = action % 7
                tm_slot = self.player.team[tm_idx]
                shp_slot = self.player.shop[shp_idx]

                self.score += 1

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
                tm1_idx = int(action/5)
                tm2_idx = action % 5

                self.score -= 1

                if not self.player.team[tm1_idx].empty and self.player.team[tm1_idx].pet.name == self.player.team[tm2_idx].pet.name:
                    self.player.combine(tm2_idx, tm1_idx)
                else:
                    self.player.team.move(tm1_idx, tm2_idx)
            elif action < 60:
                # sellteam
                action -= 55
                tm_slot = self.player.team[action]

                self.score -= 1

                self.player.sell(tm_slot)
            elif action < 67:
                # freezeshop
                action -= 60
                shp_slot = self.player.shop[action]

                self.score -= 1

                self.player.freeze(shp_slot)
            elif action < 68:
                # rollshop
                self.player.roll()

                self.score += 1
            else:
                # endturn
                self.actions_taken_this_turn = 0
                self.player.end_turn()

                prev_team = Team([])
                while len(self.past_teams) <= self.turns:
                    self.past_teams.append([])

                if len(self.past_teams[self.turns]) == 0:
                    self.past_teams[self.turns].append(Team([]))


                if len(self.past_teams[self.turns]) > 500:
                    self.past_teams[self.turns] = self.past_teams[self.turns][150:]

                prev_team = self.past_teams[self.turns][random.randint(
                    0, len(self.past_teams[self.turns])-1)]

                battle = Battle(self.player.team, prev_team)
                winner = battle.battle()

                if winner == 0:
                    self.wins += 1
                    self.score += 50
                elif winner == 1:
                    self.losses += 1
                    self.score += 5
                    if self.turns <= 2:
                        self.player.lives -= 1
                    elif self.turns <= 4:
                        self.player.lives -= 2
                    else:
                        self.player.lives -= 3
                else:
                    self.draws += 1
                    self.score += 20

                self.past_teams[self.turns].append(self.player.team)
                self.turns += 1
        
        except Exception:
            self.logs.append(traceback.format_exc())

            self.score -= 5

    def get_scaled_state(self):
        """
        Get full state, scaled into (approximately) [0, 1].
        State is: 
        team states {id, exp, atk, def, food},
        shop states {id, atk, def},
        money, turn, lives, wins
        """

        state = []
        for teamslot_state in self.player.team.state["team"]:
            pet = teamslot_state["pet"]
            if pet["name"] == "pet-none":
                state.extend([89/len(sapai.data["pets"]), 0, 0, 0, 1])
            else:
                exp = pet["experience"]
                lvl = pet["level"]
                if lvl == 2:
                    exp += 2
                elif lvl == 3:
                    exp = 5

                if pet["status"] != 'none':
                    state.extend([(list(sapai.data["pets"].keys()).index(
                        pet["name"]))/len(sapai.data["pets"]), exp/6, pet["attack"]/50, pet["health"]/50,
                        (list(sapai.data["statuses"].keys()).index(pet["status"]))/(len(sapai.data["statuses"])+1)])

                else:
                    state.extend([(list(sapai.data["pets"].keys()).index(
                        pet["name"]))/len(sapai.data["pets"]), exp/6, pet["attack"]/50, pet["health"]/50,
                        (11)/(len(sapai.data["statuses"])+1)])

        for shopslot_state in self.player.shop.state["shop_slots"]:
            item = shopslot_state["item"]
            if item["name"] == "pet-none" or item["name"] == "food-none":
                state.extend([89/len(sapai.data["pets"]), 0, 0])
            elif item["type"] == "Food":
                state.extend([(list(sapai.data["foods"].keys()).index(item["name"])+len(sapai.data["pets"]))
                              / (len(sapai.data["foods"])+len(sapai.data["pets"])), item["attack"]/50, item["health"]/50])
            else:
                state.extend([(list(sapai.data["pets"].keys()).index(
                            item["name"]))/len(sapai.data["pets"]), item["attack"]/50, item["health"]/50])

        for i in range(7-len(self.player.shop)):
            state.extend([89/len(sapai.data["pets"]), 0, 0])

        state.extend([self.player.gold, self.player.turn,
                     self.player.lives, self.wins])

        return np.array(state)

    def isGameOver(self):
        if self.player.lives <= 0 or self.wins >= 10 or self.turns >= 30 or self.actions_taken_this_turn >= 30:
            return True
        
        return False