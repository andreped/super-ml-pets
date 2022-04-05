"""
General settings and implementation of the SAP Game
"""

import random
from numpy import argmax, roll
from sapai import Player
from sapai import data
from sapai import Food
from sapai import Team
from sapai.battle import Battle

# Save the teams from every level, refresh every generation to fight against
past_teams = [[]]

total_wins = 0
total_losses = 0
total_draws = 0 

class SAP(object):
    def __init__(self):
        self.player = Player(pack="StandardPack")
        self.score = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.turns = 1
        self.actions_taken_this_turn = 0

    def step(self, action):
        """
        Update the system state using the best of action
        """
        action = argmax(action)

        self.actions_taken_this_turn += 1

        if self.actions_taken_this_turn > 20:
            self.score -= .1

        try:
            if action < 35:
                # buyshop
                tm_idx = action/7
                shp_idx = action % 7
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
                tm2_idx = action % 5
                self.score -= .5

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
            elif action < 68:
                # rollshop
                self.player.roll()
            else:
                # endturn
                self.actions_taken_this_turn = 0
                self.player.end_turn()

                prev_team = Team([])
                if len(past_teams[self.turns]) == 0:
                    past_teams[self.turns].append(Team([]))

                prev_team = past_teams[self.turns][random.randint(
                    0, len(past_teams[self.turns])-1)]

                battle = Battle(self.player.team, prev_team)
                winner = battle.battle()

                if winner == 0:
                    self.wins += 1
                    self.score += 50
                    print("won battle")
                elif winner == 1:
                    self.losses += 1
                    self.score += 5
                    print("lost battle")
                else:
                    self.draws += 1
                    self.score += 20
                    print("drew battle")

                past_teams[self.turns].append(self.player.team)
                self.turns += 1

            return True

        except:
            return False

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
                state.extend([89/len(data["pets"]), 0, 0, 1])
            else:
                state.extend([(list(data["pets"].keys()).index(
                    pet["name"]))/len(data["pets"]), pet["attack"]/50, pet["health"]/50,
                    (list(data["statuses"].keys()).index(pet["status"]))/(len(data["statuses"])+1)])

        for shopslot_state in self.player.shop.state["shop_slots"]:
            item = shopslot_state["item"]
            if item["name"] == "pet-none" or item["name"] == "food-none":
                state.extend([89/len(data["pets"]), 0, 0])
            elif item["type"] == "Food":
                state.extend([(list(data["foods"].keys()).index(item["name"])+len(data["pets"]))
                              / (len(data["foods"])+len(data["pets"])), item["attack"]/50, item["health"]/50])
            else:
                state.extend([(list(data["pets"].keys()).index(
                            item["name"]))/len(data["pets"]), item["attack"]/50, item["health"]/50])

        for i in range(7-len(self.player.shop)):
            state.extend([89/len(data["pets"]), 0, 0])

        state.extend([self.player.gold, self.player.turn,
                     self.player.lives, self.wins])

        return state

    def isGameOver(self):
        if self.player.lives <= 0 or self.player.wins >= 10 or self.turns >= 30:
            return True
        
        return False